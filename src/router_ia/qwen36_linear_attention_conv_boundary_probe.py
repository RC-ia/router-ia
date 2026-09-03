from __future__ import annotations

"""Isolate Linear Attention divergence between QKV projection and causal conv."""

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


def report(name: str, ref: torch.Tensor, got: torch.Tensor, tol: float) -> bool:
    s = _stage_stats(ref, got)
    ok = s[0] <= tol
    print(
        f"  {name:<34} {'PASS' if ok else 'FAIL'} "
        f"max_abs={s[0]:.6g} mean_abs={s[1]:.6g} "
        f"rel={s[2]:.6g} cosine={s[3]:.9f}"
    )
    return ok


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

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    raw_tokens = [
        load_embedding_row(root, args.token_id + i)
        .reshape(1, base.HIDDEN)
        .to(args.device)
        .to(dtype)
        for i in range(args.tokens)
    ]
    input_dtype = _module_input_dtype(layer)
    input_norm = base.load_layer_weight(root, args.layer, "input_layernorm.weight", args.device)

    ref_cache = _make_reference_cache(config)
    state = attention.state_for(root, args.device)
    state.reset()
    attention.activate(root, state)
    qwen, conv_originals = _patch_official_conv()

    ref_qkv: list[torch.Tensor] = []
    ref_conv_in: list[torch.Tensor] = []
    ref_conv_out: list[torch.Tensor] = []
    ref_conv_state_after: list[torch.Tensor] = []
    runtime_qkv: list[torch.Tensor] = []
    runtime_conv_in: list[torch.Tensor] = []
    runtime_conv_out: list[torch.Tensor] = []
    runtime_conv_state_after: list[torch.Tensor] = []

    # Capture the reference projection exactly at the module output.
    hook = None
    if hasattr(layer.linear_attn, "in_proj_qkv"):
        def qkv_hook(module, inputs, output):
            ref_qkv.append(output.detach().clone())
        hook = layer.linear_attn.in_proj_qkv.register_forward_hook(qkv_hook)
    else:
        raise SystemExit("Reference layer has no in_proj_qkv module")

    original_update = qwen.causal_conv1d_update
    original_fn = qwen.causal_conv1d_fn

    def capture_update(hidden_states, conv_state, weight, bias=None, activation=None):
        mixed = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
        state_len = conv_state.shape[-1]
        ref_conv_in.append(hidden_states.detach().clone())
        out = F.conv1d(
            mixed,
            weight.unsqueeze(1),
            bias,
            padding=0,
            groups=hidden_states.shape[1],
        )[:, :, -hidden_states.shape[-1]:]
        if activation is not None:
            out = torch.nn.functional.silu(out)
        conv_state.copy_(mixed[:, :, -state_len:])
        ref_conv_out.append(out.detach().clone())
        ref_conv_state_after.append(conv_state.detach().clone())
        return out.to(hidden_states.dtype)

    # Initial token uses causal_conv1d_fn rather than update. Capture it too.
    def capture_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
        ref_conv_in.append(hidden_states.detach().clone())
        padding = weight.shape[-1] - 1
        out = F.conv1d(
            hidden_states.to(weight.dtype),
            weight.unsqueeze(1),
            bias=bias,
            padding=padding,
            groups=hidden_states.shape[1],
        )[:, :, : hidden_states.shape[-1]]
        if activation is not None:
            out = torch.nn.functional.silu(out)
        ref_conv_out.append(out.detach().clone())
        return out.to(hidden_states.dtype)

    qwen.causal_conv1d_update = capture_update
    qwen.causal_conv1d_fn = capture_fn

    original_runtime_conv = attention._causal_conv1d_step

    def runtime_conv(state_obj, layer_idx, mixed_qkv, conv_weight):
        runtime_qkv.append(mixed_qkv.detach().clone())
        out = original_runtime_conv(state_obj, layer_idx, mixed_qkv, conv_weight)
        runtime_conv_in.append(mixed_qkv.detach().clone())
        runtime_conv_out.append(out.detach().clone())
        runtime_conv_state_after.append(state_obj.linear_conv_states[layer_idx].detach().clone())
        return out

    attention._causal_conv1d_step = runtime_conv

    all_ok = True
    print(
        f"op=linear-attention-conv-boundary layer={args.layer} tokens={args.tokens} "
        f"device={args.device} tolerance={args.tolerance} materialized={loaded}/{total}"
    )

    try:
        for pos, raw in enumerate(raw_tokens):
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

            got = attention.step_attention(root, args.layer, token, args.device)
            got = got - token.float()

            print(f"\nTOKEN {pos}")
            all_ok &= report("linear_output", reference, got, args.tolerance)

            # Reference projection hook emits [B,S,3D]. Runtime has [B,3D].
            if len(ref_qkv) <= pos or len(runtime_qkv) <= pos:
                print("  qkv_capture                     UNAVAILABLE")
                all_ok = False
                continue
            rq = ref_qkv[pos]
            if rq.ndim == 3:
                rq = rq[:, -1, :]
            gotq = runtime_qkv[pos]
            all_ok &= report("qkv_projection", rq, gotq, args.tolerance)

            # Each conv capture is one invocation per token. The initial function
            # and recurrent update are both normalized to the same boundary.
            if len(ref_conv_in) <= pos or len(runtime_conv_in) <= pos:
                print("  conv_input_capture              UNAVAILABLE")
                all_ok = False
                continue
            all_ok &= report("conv_input", ref_conv_in[pos], runtime_conv_in[pos].reshape_as(ref_conv_in[pos]), args.tolerance)
            all_ok &= report("conv_output", ref_conv_out[pos], runtime_conv_out[pos].reshape_as(ref_conv_out[pos]), args.tolerance)

            if pos > 0 and len(ref_conv_state_after) > pos and len(runtime_conv_state_after) >= pos + 1:
                # DynamicCache stores kernel-sized state. Compare exact state shape.
                rs = ref_conv_state_after[pos]
                gs = runtime_conv_state_after[pos]
                if tuple(rs.shape) == tuple(gs.shape):
                    all_ok &= report("conv_state_after", rs, gs, args.tolerance)
                else:
                    print(f"  conv_state_after                 SHAPE_FAIL reference={tuple(rs.shape)} runtime={tuple(gs.shape)}")
                    all_ok = False

            print(f"  reference_qkv_dtype={ref_qkv[pos].dtype} runtime_qkv_dtype={runtime_qkv[pos].dtype}")
            print(f"  reference_conv_dtype={ref_conv_out[pos].dtype} runtime_conv_dtype={runtime_conv_out[pos].dtype}")

    finally:
        attention._causal_conv1d_step = original_runtime_conv
        qwen.causal_conv1d_update = original_update
        qwen.causal_conv1d_fn = original_fn
        if hook is not None:
            hook.remove()
        attention.deactivate(root)
        layer.to_empty(device="meta")
        del meta
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    print(f"\nRESULT status={'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
