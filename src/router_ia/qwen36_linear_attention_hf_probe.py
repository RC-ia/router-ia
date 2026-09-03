from __future__ import annotations

"""Compare the standalone HF-style linear-attention implementation to Transformers."""

import argparse
import gc
from pathlib import Path

import torch

from . import qwen36_linear_attention_hf as candidate
from . import qwen36_40layer_loop as base
from .qwen36_linear_attention_stateful_probe import _make_reference_cache, _patch_official_conv
from .qwen36_layer_fidelity_probe import (
    _build_meta_model,
    _find_layers,
    _load_config,
    _materialize_layer,
    _module_input_dtype,
    _stage_stats,
)
from .qwen36_op_probe import load_embedding_row, rmsnorm


def report(name: str, ref: torch.Tensor, got: torch.Tensor, tol: float) -> bool:
    s = _stage_stats(ref, got)
    ok = s[0] <= tol
    print(
        f"  {name:<34} {'PASS' if ok else 'FAIL'} "
        f"max_abs={s[0]:.6g} mean_abs={s[1]:.6g} rel={s[2]:.6g} cosine={s[3]:.9f}"
    )
    return ok


def _cache_tensors(cache, layer_idx: int):
    if layer_idx >= len(cache.layers):
        return None, None
    lc = cache.layers[layer_idx]
    conv = getattr(lc, "conv_states", None)
    rec = getattr(lc, "recurrent_states", None)
    return (conv if torch.is_tensor(conv) else None), (rec if torch.is_tensor(rec) else None)


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
    layer = layers[args.layer]
    loaded, total = _materialize_layer(root, layer, args.layer, args.device)
    input_dtype = _module_input_dtype(layer)
    input_norm = base.load_layer_weight(root, args.layer, "input_layernorm.weight", args.device)

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    tokens = [
        load_embedding_row(root, args.token_id + i)
        .reshape(1, base.HIDDEN)
        .to(args.device)
        .to(dtype)
        for i in range(args.tokens)
    ]

    qwen, conv_originals = _patch_official_conv()
    ref_cache = _make_reference_cache(config)
    candidate_conv = None
    candidate_rec = None
    all_ok = True

    print(
        f"op=linear-attention-hf-candidate layer={args.layer} tokens={args.tokens} "
        f"device={args.device} tolerance={args.tolerance} materialized={loaded}/{total}"
    )

    try:
        for pos, raw in enumerate(tokens):
            token = raw.to(dtype=input_dtype)
            normed = rmsnorm(token, input_norm)

            reference = layer.linear_attn(
                hidden_states=normed.unsqueeze(1),
                cache_params=ref_cache,
                attention_mask=None,
            )
            if isinstance(reference, tuple):
                reference = reference[0]
            reference = reference.reshape(1, base.HIDDEN)

            candidate_out, candidate_conv, candidate_rec = candidate.linear_attention_step(
                root,
                args.layer,
                token,
                candidate_conv,
                candidate_rec,
                args.device,
            )

            print(f"\nTOKEN {pos}")
            all_ok &= report("linear_output", reference, candidate_out, args.tolerance)

            ref_conv, ref_rec = _cache_tensors(ref_cache, args.layer)
            if ref_conv is not None:
                all_ok &= report("conv_state", ref_conv, candidate_conv, args.tolerance)
            else:
                print("  conv_state                     UNAVAILABLE")

            if ref_rec is not None:
                all_ok &= report("recurrent_state", ref_rec, candidate_rec, args.tolerance)
            else:
                print("  recurrent_state                UNAVAILABLE")

            print(
                f"  candidate_conv_dtype={candidate_conv.dtype} "
                f"candidate_recurrent_dtype={candidate_rec.dtype}"
            )

    finally:
        candidate_conv = None
        candidate_rec = None
        qwen.causal_conv1d_fn, qwen.causal_conv1d_update = conv_originals
        layer.to_empty(device="meta")
        del meta
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    print(f"\nRESULT status={'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
