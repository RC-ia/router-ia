from __future__ import annotations

"""Test whether Qwen3.6 Linear Attention QKV mismatch is caused by FP16 materialization."""

import argparse
import gc
from pathlib import Path

import torch
import torch.nn.functional as F

from . import qwen36_attention_cache as attention
from . import qwen36_40layer_loop as base
from .qwen36_layer_fidelity_probe import (
    _build_meta_model,
    _find_layers,
    _load_config,
    _materialize_layer,
    _module_input_dtype,
    _stage_stats,
)
from .qwen36_linear_attention_stateful_probe import _make_reference_cache, _patch_official_conv
from .qwen36_op_probe import load_embedding_row, rmsnorm

LINEAR_CONV_DIM = int(base.LINEAR_KEY_DIM * 2 + base.LINEAR_VALUE_DIM)


def report(name: str, ref: torch.Tensor, got: torch.Tensor, tol: float) -> bool:
    s = _stage_stats(ref, got)
    print(
        f"  {name:<34} {'PASS' if s[0] <= tol else 'FAIL'} "
        f"max_abs={s[0]:.6g} mean_abs={s[1]:.6g} "
        f"rel={s[2]:.6g} cosine={s[3]:.9f}"
    )
    return s[0] <= tol


def _run_reference(layer, tokens, input_norm, input_dtype, config, upto: int) -> torch.Tensor:
    ref_cache = _make_reference_cache(config)
    out = None
    for pos in range(upto + 1):
        token = tokens[pos].to(dtype=input_dtype)
        normed = rmsnorm(token, input_norm)
        out = layer.linear_attn(
            hidden_states=normed.unsqueeze(1),
            cache_params=ref_cache,
            attention_mask=None,
        )
        if isinstance(out, tuple):
            out = out[0]
    return out.reshape(1, base.HIDDEN)


def _run_runtime(root, layer_idx, tokens, input_dtype, device, projection_override=None):
    state = attention.state_for(root, device)
    state.reset()
    attention.activate(root, state)
    original_projection = attention._projection
    if projection_override is not None:
        attention._projection = projection_override
    out = None
    try:
        for token in tokens:
            token = token.to(dtype=input_dtype)
            out = attention.step_attention(root, layer_idx, token, device)
        return out - tokens[-1].to(dtype=input_dtype).float()
    finally:
        attention._projection = original_projection
        attention.deactivate(root)


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

    # One official reference cache is used only for the normal reference pass.
    # The selective BF16 runtime test uses fresh independent state.
    qwen, originals = _patch_official_conv()
    hook_values: list[torch.Tensor] = []
    hook = layer.linear_attn.in_proj_qkv.register_forward_hook(
        lambda module, inputs, output: hook_values.append(output.detach().clone())
    )

    original_projection = attention._projection
    all_ok = True

    print(
        f"op=linear-attention-qkv-dtype layer={args.layer} tokens={args.tokens} "
        f"device={args.device} tolerance={args.tolerance} materialized={loaded}/{total}"
    )

    try:
        ref_cache = _make_reference_cache(config)
        for pos, raw in enumerate(tokens):
            token = raw.to(dtype=input_dtype)
            normed = rmsnorm(token, input_norm)

            ref_out = layer.linear_attn(
                hidden_states=normed.unsqueeze(1),
                cache_params=ref_cache,
                attention_mask=None,
            )
            if isinstance(ref_out, tuple):
                ref_out = ref_out[0]
            ref_out = ref_out.reshape(1, base.HIDDEN)
            ref_q = hook_values[pos]
            if ref_q.ndim == 3:
                ref_q = ref_q[:, -1, :]
            ref_q = ref_q.reshape(1, LINEAR_CONV_DIM)

            normal_w = original_projection(
                root,
                base.layer_prefix(args.layer) + "linear_attn.in_proj_qkv",
                args.device,
            )
            normal = F.linear(normed.to(dtype=normal_w.dtype), normal_w)
            bf_w = normal_w.to(torch.bfloat16) if args.device == "cuda" else normal_w
            bf = F.linear(normed.to(dtype=bf_w.dtype), bf_w)

            print(f"\nTOKEN {pos}")
            all_ok &= report("qkv_vs_router_normal", ref_q, normal, args.tolerance)
            all_ok &= report("qkv_vs_router_bf16", ref_q, bf, args.tolerance)
            print(
                f"  reference_dtype={ref_q.dtype} cached_dtype={normal_w.dtype} "
                f"bf16_test_dtype={bf.dtype}"
            )

            # Only for the current position, replay the complete sequence with
            # fresh reference/runtime state. This avoids double-advancing caches.
            def selective_projection(root_path, prefix, dev):
                weight = original_projection(root_path, prefix, dev)
                if dev == "cuda" and prefix.endswith("linear_attn.in_proj_qkv"):
                    return weight.to(torch.bfloat16)
                return weight

            reference_current = _run_reference(
                layer,
                tokens,
                input_norm,
                input_dtype,
                config,
                pos,
            )
            runtime_current = _run_runtime(
                root,
                args.layer,
                tokens[: pos + 1],
                input_dtype,
                args.device,
                selective_projection,
            )
            all_ok &= report("linear_with_qkv_bf16", reference_current, runtime_current, args.tolerance)

        
    finally:
        attention._projection = original_projection
        attention.deactivate(root)
        qwen.causal_conv1d_fn, qwen.causal_conv1d_update = originals
        hook.remove()
        layer.to_empty(device="meta")
        del meta
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    print(f"\nRESULT status={'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
