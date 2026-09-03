from __future__ import annotations

"""Diagnose Qwen3.6 linear-attention state divergence."""

import argparse
import gc
from pathlib import Path

import torch

from . import qwen36_attention_cache as attention
from . import qwen36_40layer_loop as base
from .qwen36_layer_fidelity_probe import _build_meta_model, _find_layers, _load_config, _materialize_layer, _module_input_dtype, _pure_torch_causal_conv1d, _stage_stats
from .qwen36_linear_attention_stateful_probe import _make_reference_cache, _patch_official_conv
from .qwen36_op_probe import load_embedding_row, rmsnorm


def report(name: str, ref: torch.Tensor, got: torch.Tensor, tol: float) -> None:
    s = _stage_stats(ref, got)
    print(f"  {name:<28} {'PASS' if s[0] <= tol else 'FAIL'} max_abs={s[0]:.6g} mean_abs={s[1]:.6g} rel={s[2]:.6g} cosine={s[3]:.9f}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--layer", type=int, default=1)
    p.add_argument("--tokens", type=int, default=4)
    p.add_argument("--token-id", type=int, default=0)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--tolerance", type=float, default=1e-3)
    args = p.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")

    root = args.root.resolve()
    config = _load_config(root)
    meta = _build_meta_model(config)
    layers = _find_layers(meta)
    if base.attention_type(root, args.layer) != "linear_attention":
        raise SystemExit(f"Layer {args.layer} is not linear_attention")

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    tokens = [load_embedding_row(root, args.token_id + i).reshape(1, base.HIDDEN).to(args.device).to(dtype) for i in range(args.tokens)]
    layer = layers[args.layer]
    loaded, total = _materialize_layer(root, layer, args.layer, args.device)
    print(f"op=linear-attention-state-compare layer={args.layer} tokens={args.tokens} device={args.device} tolerance={args.tolerance}")
    print(f"materialized={loaded}/{total}")

    ref_cache = _make_reference_cache(config)
    runtime_state = attention.state_for(root, args.device)
    runtime_state.reset()
    attention.activate(root, runtime_state)
    qwen, originals = _patch_official_conv()
    all_ok = True
    input_norm = base.load_layer_weight(root, args.layer, "input_layernorm.weight", args.device)

    try:
        for pos, token in enumerate(tokens):
            token = token.to(dtype=_module_input_dtype(layer))
            normed = rmsnorm(token, input_norm)
            ref = layer.linear_attn(hidden_states=normed.unsqueeze(1), cache_params=ref_cache, attention_mask=None)
            if isinstance(ref, tuple):
                ref = ref[0]
            ref = ref.reshape(1, base.HIDDEN)

            got_res = attention.step_attention(root, args.layer, token, args.device)
            got = got_res - token.float()
            print(f"\nTOKEN {pos}")
            report("linear_output", ref, got, args.tolerance)

            ref_layer_cache = ref_cache.layers[args.layer]
            ref_conv = ref_layer_cache.conv_states.get(0) if isinstance(ref_layer_cache.conv_states, dict) else None
            ref_rec = ref_layer_cache.recurrent_states.get(0) if isinstance(ref_layer_cache.recurrent_states, dict) else None
            got_conv = runtime_state.linear_conv_states.get(args.layer)
            got_rec = runtime_state.linear_states.get(args.layer)

            print(f"  ref_conv_shape={tuple(ref_conv.shape) if torch.is_tensor(ref_conv) else None} runtime_conv_shape={tuple(got_conv.shape) if torch.is_tensor(got_conv) else None}")
            print(f"  ref_rec_shape={tuple(ref_rec.shape) if torch.is_tensor(ref_rec) else None} runtime_rec_shape={tuple(got_rec.shape) if torch.is_tensor(got_rec) else None}")

            if torch.is_tensor(ref_conv) and torch.is_tensor(got_conv):
                ref_conv_tail = ref_conv[..., -got_conv.shape[-1]:]
                print("  conv_state_tail")
                report("conv_state_tail", ref_conv_tail, got_conv, args.tolerance)
            else:
                print("  conv_state_tail              UNAVAILABLE")
                all_ok = False

            if torch.is_tensor(ref_rec) and torch.is_tensor(got_rec):
                print("  recurrent_state")
                report("recurrent_state", ref_rec, got_rec, args.tolerance)
            else:
                print("  recurrent_state              UNAVAILABLE")
                all_ok = False

            ref_has = bool(ref_layer_cache.has_previous_state.get(0, False)) if isinstance(ref_layer_cache.has_previous_state, dict) else False
            print(f"  previous_state ref={ref_has} runtime=True")

    finally:
        attention.deactivate(root)
        qwen.causal_conv1d_fn, qwen.causal_conv1d_update = originals
        layer.to_empty(device="meta")
        del meta
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    print(f"\nRESULT status={'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
