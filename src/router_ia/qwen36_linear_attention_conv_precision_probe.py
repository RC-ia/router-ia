from __future__ import annotations

"""Isolate Qwen3.6 Linear Attention causal-conv precision/state behavior."""

import argparse
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

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


def report(name: str, ref: torch.Tensor, got: torch.Tensor, tol: float) -> bool:
    s = _stage_stats(ref, got)
    ok = s[0] <= tol
    print(
        f"  {name:<36} {'PASS' if ok else 'FAIL'} "
        f"max_abs={s[0]:.6g} mean_abs={s[1]:.6g} rel={s[2]:.6g} cosine={s[3]:.9f}"
    )
    return ok


def _token_channel_view(x: torch.Tensor) -> torch.Tensor:
    """Normalize conv captures to one-token [B,C].

    The reference functional causal-conv path can capture the whole sequence as
    [B,C,T], while the runtime step path captures one token as [B,C] or
    [B,C,1].  For a per-token boundary comparison, use the final time position.
    """
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        return x[..., -1]
    raise ValueError(f"Expected conv tensor [B,C] or [B,C,T], got {tuple(x.shape)}")


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
                scale = handle.get_tensor(scale_name) if scale_name in handle.keys() else None
                return handle.get_tensor(name), scale
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
            out = torch.nn.functional.silu(out)
        conv_state.copy_(mixed[:, :, -state_len:])
        ref_outputs.append(out.detach().clone())
        ref_states.append(conv_state.detach().clone())
        return out.to(hidden_states.dtype)

    def cap_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
        ref_inputs.append(hidden_states.detach().clone())
        out = F.conv1d(hidden_states.to(weight.dtype), weight.unsqueeze(1), bias, padding=weight.shape[-1] - 1, groups=hidden_states.shape[1])[:, :, :hidden_states.shape[-1]]
        if activation is not None:
            out = torch.nn.functional.silu(out)
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

    all_ok = True
    try:
        for pos in range(args.tokens):
            state.reset()
            ref_cache = _make_reference_cache(config)
            ref_inputs.clear(); ref_outputs.clear(); ref_states.clear()
            run_inputs.clear(); run_outputs.clear(); run_states.clear()

            for i in range(pos + 1):
                token = tokens[i].to(dtype=input_dtype)
                normed = rmsnorm(token, input_norm)
                ref = layer.linear_attn(hidden_states=normed.unsqueeze(1), cache_params=ref_cache, attention_mask=None)
                if isinstance(ref, tuple):
                    ref = ref[0]
                attention.step_attention(root, args.layer, token, args.device)

            ri = _token_channel_view(ref_inputs[-1])
            rr = _token_channel_view(ref_outputs[-1])
            gi = _token_channel_view(run_inputs[-1])
            gr = _token_channel_view(run_outputs[-1])
            rs = ref_states[-1] if pos > 0 and ref_states else None
            gs = run_states[-1] if run_states else None

            print(f"\nTOKEN {pos}")
            all_ok &= report("conv_input", ri, gi, args.tolerance)
            all_ok &= report("conv_output", rr, gr, args.tolerance)
            if rs is not None and gs is not None:
                all_ok &= report("conv_state_after", rs, gs, args.tolerance)

            print(f"  ref_conv_input_dtype={ref_inputs[-1].dtype} runtime_conv_input_dtype={run_inputs[-1].dtype} ref_conv_output_dtype={ref_outputs[-1].dtype} runtime_conv_output_dtype={run_outputs[-1].dtype}")

            print(f"  conv_ref_weight_dtype={conv_ref_w.dtype} conv_runtime_weight_dtype={conv_run_w.dtype}")
            all_ok &= report("conv_weight_vs_reference", conv_ref_w, conv_run_w, args.tolerance)

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

    print(f"\nRESULT status={'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
