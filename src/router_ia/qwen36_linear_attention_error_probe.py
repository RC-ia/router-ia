from __future__ import annotations

"""Stage-by-stage diagnostic for Qwen3.6 linear attention fidelity."""

import argparse
import gc
from pathlib import Path

import torch

from . import qwen36_linear_attention_hf as candidate
from . import qwen36_40layer_loop as base
from .qwen36_linear_attention_stateful_probe import _make_reference_cache, _patch_official_conv
from .qwen36_layer_fidelity_probe import _build_meta_model, _find_layers, _load_config, _materialize_layer, _module_input_dtype
from .qwen36_op_probe import load_embedding_row, rmsnorm


def stats(ref, got):
    r = ref.detach().float()
    g = got.detach().float()
    d = (r - g).abs()
    denom = r.norm().clamp_min(1e-12)
    cos = torch.nn.functional.cosine_similarity(r.reshape(1, -1), g.reshape(1, -1)).item()
    return float(d.max()), float(d.mean()), float(d.norm() / denom), cos


def report(name, ref, got):
    a, m, rel, cos = stats(ref, got)
    print(f"    {name:<22} max={a:.7g} mean={m:.7g} rel={rel:.7g} cos={cos:.9f} ref={ref.dtype} got={got.dtype}")


def clone(x):
    return None if x is None else x.detach().clone()


def unwrap(fn):
    depth = 0
    while hasattr(fn, "__wrapped__") and depth < 16:
        fn = fn.__wrapped__
        depth += 1
    return fn, depth


def layout(x):
    if x is None:
        return "none"
    return f"shape={tuple(x.shape)} stride={x.stride()} contiguous={x.is_contiguous()} offset={x.storage_offset()}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--layer", type=int, default=1)
    p.add_argument("--tokens", type=int, default=4)
    p.add_argument("--token-id", type=int, default=0)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")

    root = args.root.resolve()
    config = _load_config(root)
    meta = _build_meta_model(config)
    layers = _find_layers(meta)
    layer = layers[args.layer]
    loaded, total = _materialize_layer(root, layer, args.layer, args.device)
    input_dtype = _module_input_dtype(layer)
    input_norm = base.load_layer_weight(root, args.layer, "input_layernorm.weight", args.device)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32

    tokens = [load_embedding_row(root, args.token_id + i).reshape(1, base.HIDDEN).to(args.device).to(dtype) for i in range(args.tokens)]

    qwen, conv_originals = _patch_official_conv()
    ref_cache = _make_reference_cache(config)

    original_ref_recurrent = qwen.torch_recurrent_gated_delta_rule
    original_candidate_recurrent = candidate.gated_delta_recurrent
    raw_ref_recurrent, raw_depth = unwrap(original_ref_recurrent)
    ref_cap = {}
    cand_cap = {}

    def ref_wrap(query, key, value, g, beta, initial_state=None, **kwargs):
        out, state = original_ref_recurrent(query, key, value, g=g, beta=beta, initial_state=initial_state, **kwargs)
        ref_cap["query"] = clone(query)
        ref_cap["key"] = clone(key)
        ref_cap["value"] = clone(value)
        ref_cap["g"] = clone(g)
        ref_cap["beta"] = clone(beta)
        ref_cap["initial_state"] = clone(initial_state)
        ref_cap["core"] = clone(out)
        ref_cap["state"] = clone(state)
        return out, state

    def cand_wrap(query, key, value, g, beta, state):
        out, new_state = original_candidate_recurrent(query, key, value, g, beta, state)
        cand_cap["query"] = clone(query)
        cand_cap["key"] = clone(key)
        cand_cap["value"] = clone(value)
        cand_cap["g"] = clone(g)
        cand_cap["beta"] = clone(beta)
        cand_cap["initial_state"] = clone(state)
        cand_cap["core"] = clone(out)
        cand_cap["state"] = clone(new_state)
        return out, new_state

    qwen.torch_recurrent_gated_delta_rule = ref_wrap
    candidate.gated_delta_recurrent = cand_wrap

    print(f"op=linear-attention-error-probe layer={args.layer} tokens={args.tokens} device={args.device} materialized={loaded}/{total}")
    print(f"reference_recurrent={original_ref_recurrent.__module__}.{getattr(original_ref_recurrent, '__name__', type(original_ref_recurrent).__name__)} raw_depth={raw_depth} raw_recurrent={raw_ref_recurrent.__module__}.{getattr(raw_ref_recurrent, '__name__', type(raw_ref_recurrent).__name__)}")

    try:
        candidate_conv = None
        candidate_rec = None
        for pos, raw in enumerate(tokens):
            token = raw.to(dtype=input_dtype)
            normed = rmsnorm(token, input_norm)
            ref_cap.clear(); cand_cap.clear()

            reference = layer.linear_attn(hidden_states=normed.unsqueeze(1), cache_params=ref_cache, attention_mask=None)
            if isinstance(reference, tuple):
                reference = reference[0]
            reference = reference.reshape(1, base.HIDDEN)

            candidate_out, candidate_conv, candidate_rec = candidate.linear_attention_step(root, args.layer, token, candidate_conv, candidate_rec, args.device)

            print(f"\nTOKEN {pos}: final")
            report("linear_output", reference, candidate_out)

            if ref_cap and cand_cap:
                print("  recurrent boundary")
                for key in ("query", "key", "value", "g", "beta", "initial_state", "core", "state"):
                    if ref_cap.get(key) is not None and cand_cap.get(key) is not None:
                        report(key, ref_cap[key], cand_cap[key])
                    else:
                        print(f"    {key:<22} unavailable")

                print("  recurrent layouts")
                for key in ("query", "key", "value", "g", "beta", "initial_state", "core", "state"):
                    ref_x = ref_cap.get(key)
                    cand_x = cand_cap.get(key)
                    if ref_x is not None and cand_x is not None:
                        print(f"    {key:<22} ref={layout(ref_x)}")
                        print(f"    {'':<22} cand={layout(cand_x)}")

                if pos == 1:
                    print("  raw fallback vs candidate/reference")
                    try:
                        raw_kwargs = {
                            "g": ref_cap["g"],
                            "beta": ref_cap["beta"],
                            "initial_state": ref_cap["initial_state"],
                            "output_final_state": True,
                            "use_qk_l2norm_in_kernel": True,
                        }
                        raw_core, raw_state = raw_ref_recurrent(
                            ref_cap["query"],
                            ref_cap["key"],
                            ref_cap["value"],
                            **raw_kwargs,
                        )
                        report("raw_core-vs-ref", ref_cap["core"], raw_core)
                        report("raw_state-vs-ref", ref_cap["state"], raw_state)
                        report("raw_core-vs-cand", cand_cap["core"], raw_core)
                        report("raw_state-vs-cand", cand_cap["state"], raw_state)
                        print(
                            f"    ref_state_stride={ref_cap['state'].stride()} cand_state_stride={cand_cap['state'].stride()} raw_state_stride={raw_state.stride()}"
                        )
                        print(
                            f"    ref_state_contig={ref_cap['state'].is_contiguous()} cand_state_contig={cand_cap['state'].is_contiguous()} raw_state_contig={raw_state.is_contiguous()}"
                        )
                    except Exception as exc:
                        print(f"    raw_fallback ERROR {type(exc).__name__}: {exc}")
            else:
                print("  recurrent boundary     not used (chunk path)")

    finally:
        qwen.torch_recurrent_gated_delta_rule = original_ref_recurrent
        candidate.gated_delta_recurrent = original_candidate_recurrent
        candidate_conv = None
        candidate_rec = None
        qwen.causal_conv1d_fn, qwen.causal_conv1d_update = conv_originals
        layer.to_empty(device="meta")
        del meta
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
