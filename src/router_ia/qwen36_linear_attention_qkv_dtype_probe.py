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


def report(name: str, ref: torch.Tensor, got: torch.Tensor, tol: float) -> None:
    s = _stage_stats(ref, got)
    print(
        f"  {name:<34} {'PASS' if s[0] <= tol else 'FAIL'} "
        f"max_abs={s[0]:.6g} mean_abs={s[1]:.6g} "
        f"rel={s[2]:.6g} cosine={s[3]:.9f}"
    )


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

    ref_cache = _make_reference_cache(config)
    qwen, originals = _patch_official_conv()
    state = attention.state_for(root, args.device)
    state.reset()
    attention.activate(root, state)

    ref_qkv: list[torch.Tensor] = []
    hook = layer.linear_attn.in_proj_qkv.register_forward_hook(
        lambda module, inputs, output: ref_qkv.append(output.detach().clone())
    )

    original_projection = attention._projection
    force_bf16 = False

    def projection(root_path, prefix, dev):
        weight = original_projection(root_path, prefix, dev)
        if force_bf16 and dev == "cuda":
            return weight.to(torch.bfloat16)
        return weight

    attention._projection = projection

    print(
        f"op=linear-attention-qkv-dtype layer={args.layer} tokens={args.tokens} "
        f"device={args.device} tolerance={args.tolerance} materialized={loaded}/{total}"
    )

    try:
        for pos, raw in enumerate(tokens):
            token = raw.to(dtype=input_dtype)
            normed = rmsnorm(token, input_norm)

            # Reference: exact module computation.
            ref_out = layer.linear_attn(
                hidden_states=normed.unsqueeze(1),
                cache_params=ref_cache,
                attention_mask=None,
            )
            if isinstance(ref_out, tuple):
                ref_out = ref_out[0]
            ref_out = ref_out.reshape(1, base.HIDDEN)
            ref_q = ref_qkv[pos].reshape(1, base.LINEAR_CONV_DIM)

            # Router's normal projection path.
            normal_w = original_projection(root, base.layer_prefix(args.layer) + "linear_attn.in_proj_qkv", args.device)
            normal_w = normal_w.to(dtype=normal_w.dtype)
            normal = F.linear(normed.to(dtype=normal_w.dtype), normal_w)

            # Same cached weight explicitly recast to BF16, matching the module.
            bf_w = normal_w.to(torch.bfloat16) if args.device == "cuda" else normal_w
            bf = F.linear(normed.to(dtype=bf_w.dtype), bf_w)

            print(f"\nTOKEN {pos}")
            report("qkv_vs_router_normal", ref_q, normal, args.tolerance)
            report("qkv_vs_router_bf16", ref_q, bf, args.tolerance)
            print(
                f"  reference_dtype={ref_q.dtype} cached_dtype={normal_w.dtype} "
                f"bf16_test_dtype={bf.dtype}"
            )

            # Also prove the complete runtime linear-attention result with only
            # this one projection forced to BF16. All other projections remain
            # at their normal cached dtypes.
            def selective_projection(root_path, prefix, dev):
                weight = original_projection(root_path, prefix, dev)
                if dev == "cuda" and prefix.endswith("linear_attn.in_proj_qkv"):
                    return weight.to(torch.bfloat16)
                return weight

            attention._projection = selective_projection
            state.reset()
            # Rebuild reference cache to keep both paths at the same token position.
            ref_cache = _make_reference_cache(config)
            ref_out2 = None
            for prior in range(pos + 1):
                t = tokens[prior].to(dtype=input_dtype)
                n = rmsnorm(t, input_norm)
                ro = layer.linear_attn(hidden_states=n.unsqueeze(1), cache_params=ref_cache, attention_mask=None)
                if isinstance(ro, tuple):
                    ro = ro[0]
                if prior == pos:
                    ref_out2 = ro.reshape(1, base.HIDDEN)
                attention.step_attention(root, args.layer, t, args.device)
            got2 = attention.step_attention(root, args.layer, token, args.device) if False else None
            # The loop above already advanced runtime for all prior tokens only
            # through step_attention; for the current token it also advanced it.
            # Recover current runtime linear output from the final state by replaying
            # once in a fresh state for a clean comparison.
            state.reset()
            got_current = None
            for prior in range(pos + 1):
                t = tokens[prior].to(dtype=input_dtype)
                got_r = attention.step_attention(root, args.layer, t, args.device)
                if prior == pos:
                    got_current = got_r - t.float()
            report("linear_with_qkv_bf16", ref_out2, got_current, args.tolerance)

            attention._projection = projection

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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
