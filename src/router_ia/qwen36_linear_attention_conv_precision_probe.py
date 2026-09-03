from __future__ import annotations

"""Isolate Qwen3.6 Linear Attention causal-conv precision/state behavior."""

import argparse
import gc
from pathlib import Path

import torch
import torch.nn.functional as F

from . import qwen36_attention_cache as attention
from . import qwen36_40layer_loop as base
from .qwen36_dequant import dequantize_fp8_blockwise
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
from safetensors import safe_open
import json


def report(name: str, ref: torch.Tensor, got: torch.Tensor, tol: float) -> None:
    s = _stage_stats(ref, got)
    print(
        f"  {name:<36} {'PASS' if s[0] <= tol else 'FAIL'} "
        f"max_abs={s[0]:.6g} mean_abs={s[1]:.6g} rel={s[2]:.6g} cosine={s[3]:.9f}"
    )


def raw_weight(root: Path, name: str) -> tuple[torch.Tensor, torch.Tensor | None]:
    index = root / "model.safetensors.index.json"
    if index.is_file():
        payload = json.loads(index.read_text(encoding="utf-8"))
        shards = [root / payload["weight_map"][name]]
    else:
        shards = sorted(root.glob("*.safetensors"))
    scale_name = name.replace(".weight", ".weight_scale_inv")
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            if name in handle.keys():
                return handle.get_tensor(name), handle.get_tensor(scale_name) if scale_name in handle.keys() else None
    raise KeyError(name)


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
    tokens = [load_embedding_row(root, args.token_id + i).reshape(1, base.HIDDEN).to(args.device).to(dtype) for i in range(args.tokens)]

    qkv_name = base.layer_prefix(args.layer) + "linear_attn.in_proj_qkv.weight"
    raw_qkv, raw_qkv_scale = raw_weight(root, qkv_name)
    if raw_qkv.dtype == torch.float8_e4m3fn:
        if raw_qkv_scale is None:
            raise SystemExit("Missing QKV scale")
        qkv_weight = dequantize_fp8_blockwise(raw_qkv.to(args.device), raw_qkv_scale.to(args.device)).to(torch.bfloat16 if args.device == "cuda" else torch.float32)
    else:
        qkv_weight = raw_qkv.to(args.device)

    conv_ref_w = layer.linear_attn.conv1d.weight.detach().clone()
    conv_run_w = base.load_layer_weight(root, args.layer, "linear_attn.conv1d.weight", args.device)

    print(f"op=linear-attention-conv-precision layer={args.layer} tokens={args.tokens} device={args.device} tolerance={args.tolerance} materialized={loaded}/{total}")
    print(f"qkv_test_dtype={qkv_weight.dtype} reference_conv_weight_dtype={conv_ref_w.dtype} runtime_conv_weight_dtype={conv_run_w.dtype} reference_conv_weight_shape={tuple(conv_ref_w.shape)}")

    ref_cache = _make_reference_cache(config)
    qwen, originals = _patch_official_conv()
    state = attention.state_for(root, args.device)
    original_projection = attention._projection
    original_runtime_conv = attention._causal_conv1d_step
    attention.activate(root, state)

    ref_inputs: list[torch.Tensor] = []
    ref_outputs: list[torch.Tensor] = []
    ref_states: list[torch.Tensor] = []
    run_inputs: list[torch.Tensor] = []
    run_outputs: list[torch.Tensor] = []
    run_states: list[torch.Tensor] = []

    def projection(root_path, prefix, dev):
        if dev == "cuda" and prefix.endswith("linear_attn.in_proj_qkv"):
            return qkv_weight
        return original_projection(root_path, prefix, dev)

    def cap_update(hidden_states, conv_state, weight, bias=None, activation=None):
        ref_inputs.append(hidden_states.detach().clone())
        mixed = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
        state_len = conv_state.shape[-1]
        out = F.conv1d(mixed, weight.unsqueeze(1), bias, padding=0, groups=hidden_states.shape[1])[:, :, -hidden_states.shape[-1]:]
        if activation is not None:
            out = F.silu(out)
        conv_state.copy_(mixed[:, :, -state_len:])
        ref_outputs.append(out.detach().clone())
        ref_states.append(conv_state.detach().clone())
        return out.to(hidden_states.dtype)

    def cap_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
        ref_inputs.append(hidden_states.detach().clone())
        out = F.conv1d(hidden_states.to(weight.dtype), weight.unsqueeze(1), bias, padding=weight.shape[-1] - 1, groups=hidden_states.shape[1])[:, :, :hidden_states.shape[-1]]
        if activation is not None:
            out = F.silu(out)
        ref_outputs.append(out.detach().clone())
        return out.to(hidden_states.dtype)

    def run_conv(state_obj, layer_idx, mixed_qkv, conv_weight):
        run_inputs.append(mixed_qkv.detach().clone())
        out = original_runtime_conv(state_obj, layer_idx, mixed_qkv, conv_weight)
        run_outputs.append(out.detach().clone())
        run_states.append(state_obj.linear_conv_states[layer_idx].detach().clone())
        return out

    qwen.causal_conv1d_update = cap_update
    qwen.causal_conv1d_fn = cap_fn
    attention._projection = projection
    attention._causal_conv1d_step = run_conv

    ok = True
    try:
        for pos, token0 in enumerate(tokens):
            state.reset()
            ref_cache = _make_reference_cache(config)
            # Replay prefix for recurrent tokens.
            for i in range(pos + 1):
                token = tokens[i].to(dtype=input_dtype)
                normed = rmsnorm(token, input_norm)
                ref = layer.linear_attn(hidden_states=normed.unsqueeze(1), cache_params=ref_cache, attention_mask=None)
                if isinstance(ref, tuple):
                    ref = ref[0]
                attention.step_attention(root, args.layer, token, args.device)

            ri = ref_inputs[-1]
            rr = ref_outputs[-1]
            gi = run_inputs[-1]
            gr = run_outputs[-1]
            rs = ref_states[-1] if pos > 0 else None
            gs = run_states[-1]

            if ri.ndim == 3 and gi.ndim == 2:
                gi_cmp = gi.unsqueeze(-1)
            else:
                gi_cmp = gi
            if rr.ndim == 3 and gr.ndim == 2:
                gr_cmp = gr.unsqueeze(-1)
            else:
                gr_cmp = gr

            print(f"\nTOKEN {pos}")
            report("conv_input", ri, gi_cmp, args.tolerance)
            report("conv_output", rr, gr_cmp, args.tolerance)
            if rs is not None:
                report("conv_state_after", rs, gs, args.tolerance)
            print(f"  ref_conv_input_dtype={ri.dtype} runtime_conv_input_dtype={gi.dtype} ref_conv_output_dtype={rr.dtype} runtime_conv_output_dtype={gr.dtype}")

            # Compare the same QKV using reference conv weight vs runtime conv weight, isolated from all later stages.
            qkv = qkv_weight
            q = F.linear(rmsnorm(token0.to(dtype=input_dtype), input_norm).to(qkv.dtype), qkv).reshape(1, base.LINEAR_KEY_DIM * 2 + base.LINEAR_VALUE_DIM)
            q3 = q.reshape(1, q.shape[-1], 1)
            rw = conv_ref_w.to(args.device)
            tw = conv_run_w.to(args.device)
            ref_direct = F.silu(F.conv1d(q3.to(rw.dtype), rw, padding=0 if pos else 3, groups=q.shape[-1]))
            if pos == 0:
                ref_direct = ref_direct[:, :, :1]
            report("conv_direct_reference_weight", rr, ref_direct, args.tolerance)
            report("conv_weight_vs_reference", conv_ref_w, conv_run_w, args.tolerance)
    finally:
        attention._projection = original_projection
        attention._causal_conv1d_step = original_runtime_conv
        qwen.causal_conv1d_fn, qwen.causal_conv1d_update = originals
        attention.deactivate(root)
        layer.to_empty(device="meta")
        del meta
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
