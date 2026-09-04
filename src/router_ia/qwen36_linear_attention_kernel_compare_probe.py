from __future__ import annotations

"""Compare HF recurrent paths and sweep state dtype conversion."""

import argparse
import gc
import inspect
from pathlib import Path

import torch

from . import qwen36_linear_attention_hf as candidate
from . import qwen36_40layer_loop as base
from .qwen36_linear_attention_recurrence_probe import hf_literal, report, clone
from .qwen36_linear_attention_stateful_probe import _make_reference_cache, _patch_official_conv
from .qwen36_layer_fidelity_probe import _build_meta_model, _find_layers, _load_config, _materialize_layer, _module_input_dtype
from .qwen36_op_probe import load_embedding_row, rmsnorm


def run_path(fn, cap, state):
    return fn(
        cap["query"], cap["key"], cap["value"],
        g=cap["g"], beta=cap["beta"], initial_state=state,
        output_final_state=True, use_qk_l2norm_in_kernel=True,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--layer", type=int, default=1)
    p.add_argument("--tokens", type=int, default=4)
    p.add_argument("--token-id", type=int, default=0)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
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
    tokens = [
        load_embedding_row(root, args.token_id + i).reshape(1, base.HIDDEN).to(args.device).to(dtype)
        for i in range(args.tokens)
    ]

    qwen, conv_originals = _patch_official_conv()
    ref_cache = _make_reference_cache(config)
    decorated = qwen.torch_recurrent_gated_delta_rule
    fallback = inspect.unwrap(decorated)
    old_cand = candidate.gated_delta_recurrent
    cap = {}

    def ref_wrap(query, key, value, g, beta, initial_state=None, **kwargs):
        cap.clear()
        cap.update({
            "query": clone(query), "key": clone(key), "value": clone(value),
            "g": clone(g), "beta": clone(beta), "initial": clone(initial_state),
        })
        out, st = decorated(query, key, value, g=g, beta=beta, initial_state=initial_state, **kwargs)
        cap.update({"decorated_out": clone(out), "decorated_state": clone(st)})
        return out, st

    def cand_wrap(query, key, value, g, beta, state):
        out, st = old_cand(query, key, value, g, beta, state)
        cap.update({"candidate_out": clone(out), "candidate_state": clone(st)})
        return out, st

    qwen.torch_recurrent_gated_delta_rule = ref_wrap
    candidate.gated_delta_recurrent = cand_wrap

    print(
        f"op=linear-attention-kernel-compare layer={args.layer} tokens={args.tokens} "
        f"device={args.device} materialized={loaded}/{total}"
    )
    print(f"decorated: {decorated!r}")
    print(f"fallback : {fallback!r}")
    print(f"same object: {decorated is fallback}")

    try:
        cand_conv = cand_state = None
        with torch.no_grad():
            for pos, raw in enumerate(tokens):
                token = raw.to(dtype=input_dtype)
                normed = rmsnorm(token, input_norm)
                cap.clear()
                reference = layer.linear_attn(hidden_states=normed.unsqueeze(1), cache_params=ref_cache, attention_mask=None)
                if isinstance(reference, tuple):
                    reference = reference[0]
                cand_out, cand_conv, cand_state = candidate.linear_attention_step(
                    root, args.layer, token, cand_conv, cand_state, args.device
                )

                if "decorated_out" not in cap:
                    print(f"\nTOKEN {pos}: chunk path")
                    print(f"  candidate state dtype: {cand_state.dtype}")
                    continue

                initial = cap["initial"].clone() if cap["initial"] is not None else None
                print(f"\nTOKEN {pos}")
                print(f"  input value dtype={cap['value'].dtype} initial dtype={initial.dtype if initial is not None else None}")
                print(f"  candidate state dtype={cand_state.dtype}")

                fb_out, fb_state = run_path(fallback, cap, initial)
                lit_out, lit_state, _ = hf_literal(
                    cap["query"], cap["key"], cap["value"],
                    cap["g"], cap["beta"], cap["initial"],
                )

                report("decorated_vs_fallback_out", cap["decorated_out"], fb_out)
                report("decorated_vs_literal_out", cap["decorated_out"], lit_out)
                report("fallback_vs_literal_out", fb_out, lit_out)
                report("candidate_vs_fallback_out", cap["candidate_out"], fb_out)
                report("decorated_vs_fallback_state", cap["decorated_state"], fb_state)
                report("decorated_vs_literal_state", cap["decorated_state"], lit_state)
                report("fallback_vs_literal_state", fb_state, lit_state)
                report("candidate_vs_fallback_state", cap["candidate_state"], fb_state)

                if pos == 1 and initial is not None:
                    print("  STATE DTYPE SWEEP (same captured token-1 inputs)")
                    candidates = {
                        "initial_as_is": initial,
                        "initial_to_value": initial.to(cap["value"]),
                        "initial_to_bf16": initial.to(dtype=torch.bfloat16),
                        "initial_to_fp32": initial.to(dtype=torch.float32),
                        "initial_clone": initial.clone(),
                    }
                    for name, st_in in candidates.items():
                        o, s = run_path(fallback, cap, st_in)
                        report(f"{name}_vs_decorated_out", cap["decorated_out"], o)
                        report(f"{name}_vs_decorated_state", cap["decorated_state"], s)

                report("model_vs_candidate_final", reference.reshape_as(cand_out), cand_out)
    finally:
        qwen.torch_recurrent_gated_delta_rule = decorated
        candidate.gated_delta_recurrent = old_cand
        qwen.causal_conv1d_fn, qwen.causal_conv1d_update = conv_originals
        layer.to_empty(device="meta")
        del meta
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
