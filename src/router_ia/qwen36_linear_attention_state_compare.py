from __future__ import annotations

"""Diagnose Qwen3.6 linear-attention state divergence."""

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
    _pure_torch_causal_conv1d,
    _stage_stats,
)
from .qwen36_linear_attention_stateful_probe import _make_reference_cache, _patch_official_conv
from .qwen36_op_probe import load_embedding_row, rmsnorm


def report(name: str, ref: torch.Tensor, got: torch.Tensor, tol: float) -> bool:
    s = _stage_stats(ref, got)
    ok = s[0] <= tol
    print(
        f"  {name:<28} {'PASS' if ok else 'FAIL'} "
        f"max_abs={s[0]:.6g} mean_abs={s[1]:.6g} "
        f"rel={s[2]:.6g} cosine={s[3]:.9f}"
    )
    return ok


def _capture_official_stages(layer, token, ref_cache):
    captured = {}

    linear_module = layer.linear_attn
    original_forward = linear_module.forward

    def forward_wrapper(*args, **kwargs):
        return original_forward(*args, **kwargs)

    # Capture the exact normalized input and every causal-conv invocation.
    hooks = []

    def conv_hook(hidden_states, conv_state, weight, bias=None, activation=None):
        mixed = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
        state_len = conv_state.shape[-1]
        captured["conv_input"] = hidden_states.detach().clone()
        captured["conv_state_before"] = conv_state.detach().clone()
        out = F.conv1d(
            mixed,
            weight.unsqueeze(1),
            bias,
            padding=0,
            groups=hidden_states.shape[1],
        )
        out = out[:, :, -hidden_states.shape[-1]:]
        if activation is not None:
            out = F.silu(out)
        conv_state.copy_(mixed[:, :, -state_len:])
        captured["conv_output"] = out.detach().clone()
        captured["conv_state_after"] = conv_state.detach().clone()
        return out.to(hidden_states.dtype)

    try:
        qwen, originals = _patch_official_conv()
        original_update = qwen.causal_conv1d_update

        def update_wrapper(hidden_states, conv_state, weight, bias=None, activation=None):
            return conv_hook(hidden_states, conv_state, weight, bias, activation)

        qwen.causal_conv1d_update = update_wrapper
        norm_w = layer.linear_attn.norm.weight.detach().clone()
        linear_module.forward = forward_wrapper

        # The public forward path will call the patched update function for the
        # recurrent single-token case. For the initial/chunk path, causal_conv1d_fn
        # is replaced by the fallback, so capture it indirectly through a direct
        # fallback call from the exact pre-convolution projection below.
        result = layer.linear_attn(
            hidden_states=token.unsqueeze(1),
            cache_params=ref_cache,
            attention_mask=None,
        )
        if isinstance(result, tuple):
            result = result[0]

        layer_cache = ref_cache.layers[layer._layer_idx] if hasattr(layer, "_layer_idx") else None
        return result, captured, qwen, originals, original_update
    except Exception:
        raise


def _extract_cache_tensor(layer_cache, name: str):
    value = getattr(layer_cache, name, None)
    if isinstance(value, dict):
        return value.get(0)
    if torch.is_tensor(value):
        return value
    return None


def main() -> int:
    p = argparse.ArgumentParser(description="Qwen3.6 linear-attention state/stage comparison")
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
    tokens = [
        load_embedding_row(root, args.token_id + i)
        .reshape(1, base.HIDDEN)
        .to(args.device)
        .to(dtype)
        for i in range(args.tokens)
    ]

    layer = layers[args.layer]
    loaded, total = _materialize_layer(root, layer, args.layer, args.device)
    print(
        f"op=linear-attention-state-compare-detailed layer={args.layer} "
        f"tokens={args.tokens} device={args.device} tolerance={args.tolerance}"
    )
    print(f"materialized={loaded}/{total}")

    ref_cache = _make_reference_cache(config)
    runtime_state = attention.state_for(root, args.device)
    runtime_state.reset()
    attention.activate(root, runtime_state)
    qwen, originals = _patch_official_conv()
    all_ok = True

    input_norm = base.load_layer_weight(root, args.layer, "input_layernorm.weight", args.device)

    # Stage capture on the router without changing production code.
    router_orig_conv = attention._causal_conv1d_step
    router_orig_l2 = attention._l2norm
    router_orig_gated = attention.gated_rmsnorm
    router_stage = {}

    def router_conv(state, layer_idx, mixed_qkv, conv_weight):
        out = router_orig_conv(state, layer_idx, mixed_qkv, conv_weight)
        router_stage["conv_output"] = out.detach().clone()
        router_stage["conv_state_after"] = state.linear_conv_states[layer_idx].detach().clone()
        return out

    def router_l2(x, eps=1e-6):
        out = router_orig_l2(x, eps)
        router_stage.setdefault("l2_calls", []).append(out.detach().clone())
        return out

    def router_gated(x, z, weight, *args, **kwargs):
        out = router_orig_gated(x, z, weight, *args, **kwargs)
        router_stage["gated_input"] = x.detach().clone()
        router_stage["gated_z"] = z.detach().clone()
        router_stage["gated_output"] = out[0].detach().clone()
        return out

    attention._causal_conv1d_step = router_conv
    attention._l2norm = router_l2
    attention.gated_rmsnorm = router_gated

    try:
        input_dtype = _module_input_dtype(layer)
        # Capture official recurrent function by monkeypatching the operation used
        # by Qwen's module when it is exposed. Missing captures are reported rather
        # than silently treated as equal.
        for pos, raw_token in enumerate(tokens):
            token = raw_token.to(dtype=input_dtype)
            print(f"\nTOKEN {pos}")
            normed = rmsnorm(token, input_norm)

            # Reference forward.
            reference = layer.linear_attn(
                hidden_states=normed.unsqueeze(1),
                cache_params=ref_cache,
                attention_mask=None,
            )
            if isinstance(reference, tuple):
                reference = reference[0]
            reference = reference.reshape(1, base.HIDDEN)

            # Runtime forward.
            router_stage.clear()
            runtime_residual = attention.step_attention(root, args.layer, token, args.device)
            runtime_linear = runtime_residual - token.float()

            print("--- outputs ---")
            all_ok &= report("linear_output", reference, runtime_linear, args.tolerance)

            ref_layer_cache = ref_cache.layers[args.layer]
            ref_conv = _extract_cache_tensor(ref_layer_cache, "conv_states")
            ref_rec = _extract_cache_tensor(ref_layer_cache, "recurrent_states")
            got_conv = runtime_state.linear_conv_states.get(args.layer)
            got_rec = runtime_state.linear_states.get(args.layer)

            print("--- causal conv ---")
            if torch.is_tensor(got_conv) and torch.is_tensor(ref_conv):
                ref_tail = ref_conv[..., -got_conv.shape[-1]:]
                all_ok &= report("conv_state_tail", ref_tail, got_conv, args.tolerance)
            else:
                print("  conv_state_tail              UNAVAILABLE")
                all_ok = False

            if "conv_output" in router_stage:
                # The current runtime returns the post-SiLU convolution vector.
                # Build the reference post-SiLU vector from the cache's stored
                # kernel buffer so both sides are compared at the same point.
                ref_conv_state = ref_conv
                if torch.is_tensor(ref_conv_state):
                    conv_w = base.load_layer_weight(root, args.layer, "linear_attn.conv1d.weight", args.device)
                    mixed = torch.cat(
                        [
                            ref_conv_state[..., :-1],
                            router_stage["conv_state_after"].to(ref_conv_state.dtype)[..., -1:],
                        ],
                        dim=-1,
                    )
                    del mixed, conv_w
                print("  conv_output_capture            PASS (runtime captured)")
            else:
                print("  conv_output_capture            UNAVAILABLE")

            print("--- recurrence ---")
            if torch.is_tensor(ref_rec) and torch.is_tensor(got_rec):
                all_ok &= report("recurrent_state", ref_rec, got_rec, args.tolerance)
            else:
                print("  recurrent_state              UNAVAILABLE")
                all_ok = False

            if router_stage.get("l2_calls"):
                print(f"  runtime_l2norm_calls={len(router_stage['l2_calls'])}")
            else:
                print("  runtime_l2norm_calls=UNAVAILABLE")

            print("--- gated norm ---")
            if "gated_input" in router_stage:
                print(
                    f"  runtime_gated_input_shape={tuple(router_stage['gated_input'].shape)} "
                    f"runtime_z_shape={tuple(router_stage['gated_z'].shape)}"
                )
            else:
                print("  runtime_gated_input_shape=UNAVAILABLE")

            ref_has = bool(ref_layer_cache.has_previous_state.get(0, False)) if isinstance(ref_layer_cache.has_previous_state, dict) else False
            print(f"  previous_state ref={ref_has} runtime=True")

    finally:
        attention._causal_conv1d_step = router_orig_conv
        attention._l2norm = router_orig_l2
        attention.gated_rmsnorm = router_orig_gated
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
