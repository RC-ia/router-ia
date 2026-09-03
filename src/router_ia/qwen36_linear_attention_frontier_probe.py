from __future__ import annotations

"""Pinpoint divergence across Qwen3.6 linear-attention post-conv boundaries."""

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

TOLERANCE = 1e-3
HEAD_DIM = 128


def report(name: str, ref: torch.Tensor, got: torch.Tensor, tol: float) -> bool:
    s = _stage_stats(ref, got)
    ok = s[0] <= tol
    print(
        f"  {name:<42} {'PASS' if ok else 'FAIL'} "
        f"max_abs={s[0]:.6g} mean_abs={s[1]:.6g} "
        f"rel={s[2]:.6g} cosine={s[3]:.9f}"
    )
    return ok


def _raw_weight(root: Path, name: str) -> tuple[torch.Tensor, torch.Tensor | None]:
    index = root / "model.safetensors.index.json"
    shards = [root / json.loads(index.read_text(encoding="utf-8"))["weight_map"][name]] if index.is_file() else sorted(root.glob("*.safetensors"))
    scale_name = name.replace(".weight", ".weight_scale_inv")
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            if name in handle.keys():
                scale = handle.get_tensor(scale_name) if scale_name in handle.keys() else None
                return handle.get_tensor(name), scale
    raise KeyError(name)


def _qkv_bf16(root: Path, layer_idx: int, device: str) -> torch.Tensor | None:
    name = base.layer_prefix(layer_idx) + "linear_attn.in_proj_qkv.weight"
    raw, scale = _raw_weight(root, name)
    if raw.dtype == torch.float8_e4m3fn:
        if scale is None:
            raise RuntimeError(f"Missing scale tensor for {name}")
        out = dequantize_fp8_blockwise(raw.to(device), scale.to(device))
        return out.to(torch.bfloat16 if device == "cuda" else torch.float32)
    return raw.to(device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)


def _as_token_channels(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        return x[:, :, -1]
    raise ValueError(f"unexpected conv tensor shape: {tuple(x.shape)}")


def _split_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    x = _as_token_channels(x)
    return x.reshape(x.shape[0], heads, HEAD_DIM)


def _ref_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    if x.ndim == 4:
        if x.shape[1] == 1 and x.shape[2] == heads and x.shape[3] == HEAD_DIM:
            return x[:, 0]
        if x.shape[1] == heads and x.shape[2] == 1 and x.shape[3] == HEAD_DIM:
            return x[:, :, 0]
    return _split_heads(x, heads)


def _reference_conv_update(hidden_states, conv_state, weight, bias=None, activation=None):
    state_len = conv_state.shape[-1]
    mixed = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(mixed[:, :, -state_len:])
    out = F.conv1d(mixed, weight.unsqueeze(1), bias, padding=0, groups=hidden_states.shape[1])
    out = out[:, :, -hidden_states.shape[-1]:]
    if activation is not None:
        out = F.silu(out)
    return out.to(hidden_states.dtype)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--layer", type=int, default=1)
    p.add_argument("--tokens", type=int, default=4)
    p.add_argument("--token-id", type=int, default=0)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--tolerance", type=float, default=TOLERANCE)
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
    token_dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    tokens = [
        load_embedding_row(root, args.token_id + i).reshape(1, base.HIDDEN).to(args.device).to(token_dtype)
        for i in range(args.tokens)
    ]

    qkv_bf16 = _qkv_bf16(root, args.layer, args.device)
    if qkv_bf16 is None:
        raise RuntimeError("Unable to materialize QKV weight")

    ref_cache = _make_reference_cache(config)
    state = attention.state_for(root, args.device)
    state.reset()
    attention.activate(root, state)
    qwen, conv_originals = _patch_official_conv()
    original_projection = attention._projection
    original_runtime_conv = attention._causal_conv1d_step
    original_l2 = attention._l2norm
    original_recurrent = qwen.torch_recurrent_gated_delta_rule

    ref_conv_inputs: list[torch.Tensor] = []
    ref_conv_outputs: list[torch.Tensor] = []
    run_conv_inputs: list[torch.Tensor] = []
    run_conv_outputs: list[torch.Tensor] = []
    runtime_l2_inputs: list[torch.Tensor] = []
    runtime_l2_outputs: list[torch.Tensor] = []
    reference_calls: list[dict[str, torch.Tensor]] = []

    def projection(root_path, prefix, dev):
        if prefix.endswith("linear_attn.in_proj_qkv"):
            return qkv_bf16
        return original_projection(root_path, prefix, dev)

    def capture_ref_update(hidden_states, conv_state, weight, bias=None, activation=None):
        ref_conv_inputs.append(hidden_states.detach().clone())
        out = _reference_conv_update(hidden_states, conv_state, weight, bias=bias, activation=activation)
        ref_conv_outputs.append(out.detach().clone())
        return out

    def capture_ref_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
        ref_conv_inputs.append(hidden_states.detach().clone())
        mixed = hidden_states.to(weight.dtype)
        out = F.conv1d(mixed, weight.unsqueeze(1), bias, padding=weight.shape[-1] - 1, groups=hidden_states.shape[1])
        out = out[:, :, :hidden_states.shape[-1]]
        if activation is not None:
            out = F.silu(out)
        ref_conv_outputs.append(out.detach().clone())
        return out.to(hidden_states.dtype)

    def capture_runtime_conv(state_obj, layer_idx, mixed_qkv, conv_weight):
        run_conv_inputs.append(mixed_qkv.detach().clone())
        out = original_runtime_conv(state_obj, layer_idx, mixed_qkv, conv_weight)
        run_conv_outputs.append(out.detach().clone())
        return out

    def capture_l2(x, eps=1e-6):
        runtime_l2_inputs.append(x.detach().clone())
        out = original_l2(x, eps)
        runtime_l2_outputs.append(out.detach().clone())
        return out

    def arg_at(positional, keywords, index, name):
        if len(positional) > index:
            return positional[index]
        if name in keywords:
            return keywords[name]
        raise TypeError(f"missing recurrence argument: {name}")

    def capture_recurrent(*call_args, **call_kwargs):
        q = arg_at(call_args, call_kwargs, 0, "query")
        k = arg_at(call_args, call_kwargs, 1, "key")
        v = arg_at(call_args, call_kwargs, 2, "value")
        g = arg_at(call_args, call_kwargs, 3, "g")
        beta = arg_at(call_args, call_kwargs, 4, "beta")
        reference_calls.append({"q": q.detach().clone(), "k": k.detach().clone(), "v": v.detach().clone(), "g": g.detach().clone(), "beta": beta.detach().clone()})
        return original_recurrent(*call_args, **call_kwargs)

    qwen.causal_conv1d_update = capture_ref_update
    qwen.causal_conv1d_fn = capture_ref_fn
    attention._projection = projection
    attention._causal_conv1d_step = capture_runtime_conv
    attention._l2norm = capture_l2
    qwen.torch_recurrent_gated_delta_rule = capture_recurrent

    print(f"op=linear-attention-frontier layer={args.layer} tokens={args.tokens} device={args.device} tolerance={args.tolerance} materialized={loaded}/{total}")
    print(f"key_dim={base.LINEAR_KEY_DIM} value_dim={base.LINEAR_VALUE_DIM} k_heads={base.LINEAR_NUM_K_HEADS} v_heads={base.LINEAR_NUM_V_HEADS} head_dim={HEAD_DIM}")

    all_ok = True
    try:
        for pos, raw in enumerate(tokens):
            ref_before = len(ref_conv_inputs)
            run_before = len(run_conv_inputs)
            l2_before = len(runtime_l2_inputs)
            rec_before = len(reference_calls)

            token = raw.to(dtype=input_dtype)
            normed = rmsnorm(token, input_norm)
            reference = layer.linear_attn(hidden_states=normed.unsqueeze(1), cache_params=ref_cache, attention_mask=None)
            if isinstance(reference, tuple):
                reference = reference[0]

            runtime_residual = attention.step_attention(root, args.layer, token, args.device)
            runtime_linear = runtime_residual - token.float()

            ref_in = ref_conv_inputs[-1] if len(ref_conv_inputs) > ref_before else None
            run_in = run_conv_inputs[-1] if len(run_conv_inputs) > run_before else None
            ref_out = ref_conv_outputs[-1] if len(ref_conv_outputs) > ref_before else None
            run_out = run_conv_outputs[-1] if len(run_conv_outputs) > run_before else None
            new_l2 = runtime_l2_inputs[l2_before:]
            new_l2_out = runtime_l2_outputs[l2_before:]
            rec = reference_calls[-1] if len(reference_calls) > rec_before else None

            print(f"\nTOKEN {pos}")
            if ref_in is not None and run_in is not None:
                all_ok &= report("conv_input_ref_vs_runtime", _as_token_channels(ref_in), _as_token_channels(run_in), args.tolerance)
                print(f"  conv_input_ref_dtype={ref_in.dtype} conv_input_runtime_dtype={run_in.dtype}")
            else:
                print("  conv_input_boundary                  UNAVAILABLE")
                all_ok = False

            if ref_out is not None and run_out is not None:
                all_ok &= report("conv_output_ref_vs_runtime", _as_token_channels(ref_out), _as_token_channels(run_out), args.tolerance)
                print(f"  conv_output_ref_dtype={ref_out.dtype} conv_output_runtime_dtype={run_out.dtype}")
            else:
                print("  conv_output_boundary                 UNAVAILABLE")
                all_ok = False

            run_tok = _as_token_channels(run_out) if run_out is not None else None
            if run_tok is None:
                continue

            q_flat, k_flat, v_flat = torch.split(run_tok, [base.LINEAR_KEY_DIM, base.LINEAR_KEY_DIM, base.LINEAR_VALUE_DIM], dim=-1)
            q_split = q_flat.reshape(1, base.LINEAR_NUM_K_HEADS, HEAD_DIM).repeat_interleave(2, dim=1)
            k_split = k_flat.reshape(1, base.LINEAR_NUM_K_HEADS, HEAD_DIM).repeat_interleave(2, dim=1)
            v_split = v_flat.reshape(1, base.LINEAR_NUM_V_HEADS, HEAD_DIM)

            print("--- reshape / repeat boundary ---")
            print(f"  q_split_shape={tuple(q_split.shape)} k_split_shape={tuple(k_split.shape)} v_split_shape={tuple(v_split.shape)}")
            if len(new_l2) >= 2:
                all_ok &= report("q_split_vs_runtime_l2_input", q_split, new_l2[-2], args.tolerance)
                all_ok &= report("k_split_vs_runtime_l2_input", k_split, new_l2[-1], args.tolerance)

            if rec is not None:
                rq_ref = _ref_heads(rec["q"], base.LINEAR_NUM_V_HEADS)
                rk_ref = _ref_heads(rec["k"], base.LINEAR_NUM_V_HEADS)
                rv_ref = _ref_heads(rec["v"], base.LINEAR_NUM_V_HEADS)
                print("--- reference recurrence boundary ---")
                all_ok &= report("q_split_vs_reference_q", q_split, rq_ref, args.tolerance)
                all_ok &= report("k_split_vs_reference_k", k_split, rk_ref, args.tolerance)
                all_ok &= report("v_split_vs_reference_v", v_split, rv_ref, args.tolerance)
                print(f"  ref_q_dtype={rec['q'].dtype} ref_k_dtype={rec['k'].dtype} ref_v_dtype={rec['v'].dtype}")
            else:
                print("  reference_recurrence_boundary      SKIP (initial chunk path)")

            if len(new_l2_out) >= 2:
                q_norm = new_l2_out[-2]
                k_norm = new_l2_out[-1]
                q_expected = q_split.float() / torch.sqrt((q_split.float() * q_split.float()).sum(-1, keepdim=True) + 1e-6)
                k_expected = k_split.float() / torch.sqrt((k_split.float() * k_split.float()).sum(-1, keepdim=True) + 1e-6)
                print("--- L2 boundary ---")
                all_ok &= report("q_l2_vs_fp32_from_split", q_expected, q_norm, args.tolerance)
                all_ok &= report("k_l2_vs_fp32_from_split", k_expected, k_norm, args.tolerance)

            print(f"  l2_calls_this_token={len(new_l2)} recurrence_calls_this_token={len(reference_calls) - rec_before}")
            if pos == 0:
                print(f"  linear_output_vs_reference={'PASS' if _stage_stats(reference.reshape(1, base.HIDDEN), runtime_linear)[0] <= args.tolerance else 'FAIL'}")

    finally:
        qwen.causal_conv1d_update, qwen.causal_conv1d_fn = conv_originals[1], conv_originals[0]
        qwen.torch_recurrent_gated_delta_rule = original_recurrent
        attention._projection = original_projection
        attention._causal_conv1d_step = original_runtime_conv
        attention._l2norm = original_l2
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
