from __future__ import annotations

"""Pinpoint divergence between linear-attention conv output and recurrence inputs."""

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


TOLERANCE = 1e-3
HEAD_DIM = 128


def report(name: str, ref: torch.Tensor, got: torch.Tensor, tol: float) -> bool:
    s = _stage_stats(ref, got)
    ok = s[0] <= tol
    print(
        f"  {name:<40} {'PASS' if ok else 'FAIL'} "
        f"max_abs={s[0]:.6g} mean_abs={s[1]:.6g} "
        f"rel={s[2]:.6g} cosine={s[3]:.9f}"
    )
    return ok


def _as_heads(x: torch.Tensor, heads: int, name: str) -> torch.Tensor:
    """Normalize a token tensor to [B,H,D]."""
    if x.ndim == 2:
        if x.shape[-1] != heads * HEAD_DIM:
            raise ValueError(f"{name}: expected last dim {heads * HEAD_DIM}, got {tuple(x.shape)}")
        return x.reshape(x.shape[0], heads, HEAD_DIM)
    if x.ndim == 3:
        if x.shape[1] == heads and x.shape[2] == HEAD_DIM:
            return x
        if x.shape[-1] == heads * HEAD_DIM:
            return x.reshape(x.shape[0], -1, heads, HEAD_DIM)[:, -1]
    if x.ndim == 4:
        # [B,S,H,D] or [B,H,S,D]; select the final sequence token.
        if x.shape[-1] == HEAD_DIM and x.shape[-2] == heads:
            return x[:, -1]
        if x.shape[-1] == HEAD_DIM and x.shape[-3] == heads:
            return x[:, :, -1]
    raise ValueError(f"{name}: unsupported shape {tuple(x.shape)}")


def _ref_heads(x: torch.Tensor, heads: int, name: str) -> torch.Tensor:
    """Normalize official recurrence input to [B,H,D], selecting S=1."""
    if x.ndim == 4:
        if x.shape[1] == 1 and x.shape[2] == heads and x.shape[3] == HEAD_DIM:
            return x[:, 0]
        if x.shape[1] == heads and x.shape[2] == 1 and x.shape[3] == HEAD_DIM:
            return x[:, :, 0]
    return _as_heads(x, heads, name)


def _conv_token(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        return x[:, :, -1]
    raise ValueError(f"unexpected conv output shape: {tuple(x.shape)}")


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

    original_runtime_conv = attention._causal_conv1d_step
    original_l2 = attention._l2norm
    original_recurrent = qwen.torch_recurrent_gated_delta_rule

    ref_conv_outputs: list[torch.Tensor] = []
    run_conv_outputs: list[torch.Tensor] = []
    runtime_l2_inputs: list[torch.Tensor] = []
    runtime_l2_outputs: list[torch.Tensor] = []
    reference_calls: list[dict[str, torch.Tensor]] = []

    def capture_ref_update(hidden_states, conv_state, weight, bias=None, activation=None):
        out = conv_originals[1](hidden_states, conv_state, weight, bias=bias, activation=activation)
        ref_conv_outputs.append(out.detach().clone())
        return out

    def capture_ref_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
        out = conv_originals[0](hidden_states, weight, bias=bias, activation=activation, **kwargs)
        ref_conv_outputs.append(out.detach().clone())
        return out

    def capture_runtime_conv(state_obj, layer_idx, mixed_qkv, conv_weight):
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
        reference_calls.append({
            "q": q.detach().clone(),
            "k": k.detach().clone(),
            "v": v.detach().clone(),
            "g": g.detach().clone(),
            "beta": beta.detach().clone(),
        })
        return original_recurrent(*call_args, **call_kwargs)

    qwen.causal_conv1d_update = capture_ref_update
    qwen.causal_conv1d_fn = capture_ref_fn
    attention._causal_conv1d_step = capture_runtime_conv
    attention._l2norm = capture_l2
    qwen.torch_recurrent_gated_delta_rule = capture_recurrent

    print(
        f"op=linear-attention-frontier layer={args.layer} tokens={args.tokens} "
        f"device={args.device} tolerance={args.tolerance} materialized={loaded}/{total}"
    )
    print(
        f"key_dim={base.LINEAR_KEY_DIM} value_dim={base.LINEAR_VALUE_DIM} "
        f"k_heads={base.LINEAR_NUM_K_HEADS} v_heads={base.LINEAR_NUM_V_HEADS} head_dim={HEAD_DIM}"
    )

    all_ok = True
    try:
        for pos, raw in enumerate(raw_tokens):
            ref_before = len(ref_conv_outputs)
            run_before = len(run_conv_outputs)
            l2_before = len(runtime_l2_inputs)
            rec_before = len(reference_calls)

            token = raw.to(dtype=input_dtype)
            normed = rmsnorm(token, input_norm)
            reference = layer.linear_attn(
                hidden_states=normed.unsqueeze(1),
                cache_params=ref_cache,
                attention_mask=None,
            )
            if isinstance(reference, tuple):
                reference = reference[0]

            attention.step_attention(root, args.layer, token, args.device)

            ref_conv = ref_conv_outputs[-1] if len(ref_conv_outputs) > ref_before else None
            run_conv = run_conv_outputs[-1] if len(run_conv_outputs) > run_before else None
            new_l2_in = runtime_l2_inputs[l2_before:]
            new_l2_out = runtime_l2_outputs[l2_before:]
            rec = reference_calls[-1] if len(reference_calls) > rec_before else None

            print(f"\nTOKEN {pos}")
            if ref_conv is None or run_conv is None:
                print("  conv_boundary                       UNAVAILABLE")
                all_ok = False
                continue

            ref_conv_tok = _conv_token(ref_conv)
            run_conv_tok = _conv_token(run_conv)
            all_ok &= report("conv_output_ref_vs_runtime", ref_conv_tok, run_conv_tok, args.tolerance)
            print(f"  conv_ref_dtype={ref_conv.dtype} conv_runtime_dtype={run_conv.dtype} shape={tuple(run_conv.shape)}")

            q_flat, k_flat, v_flat = torch.split(
                run_conv_tok,
                [base.LINEAR_KEY_DIM, base.LINEAR_KEY_DIM, base.LINEAR_VALUE_DIM],
                dim=-1,
            )
            q_split = q_flat.reshape(1, base.LINEAR_NUM_K_HEADS, HEAD_DIM).repeat_interleave(2, dim=1)
            k_split = k_flat.reshape(1, base.LINEAR_NUM_K_HEADS, HEAD_DIM).repeat_interleave(2, dim=1)
            v_split = v_flat.reshape(1, base.LINEAR_NUM_V_HEADS, HEAD_DIM)

            print("--- reshape / repeat boundary ---")
            print(f"  q_split_shape={tuple(q_split.shape)} k_split_shape={tuple(k_split.shape)} v_split_shape={tuple(v_split.shape)}")

            if new_l2_in:
                # Runtime calls _l2norm(q), then _l2norm(k), in that order.
                rq = new_l2_in[-2]
                rk = new_l2_in[-1]
                all_ok &= report("q_split_vs_runtime_l2_input", q_split, rq, args.tolerance)
                all_ok &= report("k_split_vs_runtime_l2_input", k_split, rk, args.tolerance)
            else:
                print("  runtime_l2_inputs                  UNAVAILABLE")
                all_ok = False

            if rec is not None:
                rq_ref = _ref_heads(rec["q"], base.LINEAR_NUM_V_HEADS, "reference_q")
                rk_ref = _ref_heads(rec["k"], base.LINEAR_NUM_V_HEADS, "reference_k")
                rv_ref = _ref_heads(rec["v"], base.LINEAR_NUM_V_HEADS, "reference_v")
                print("--- reference recurrence boundary ---")
                all_ok &= report("q_split_vs_reference_q", q_split, rq_ref, args.tolerance)
                all_ok &= report("k_split_vs_reference_k", k_split, rk_ref, args.tolerance)
                all_ok &= report("v_split_vs_reference_v", v_split, rv_ref, args.tolerance)
                print(
                    f"  ref_q_dtype={rec['q'].dtype} ref_k_dtype={rec['k'].dtype} "
                    f"ref_v_dtype={rec['v'].dtype}"
                )
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
                if rec is not None:
                    rq_ref = _ref_heads(rec["q"], base.LINEAR_NUM_V_HEADS, "reference_q").float()
                    rk_ref = _ref_heads(rec["k"], base.LINEAR_NUM_V_HEADS, "reference_k").float()
                    # The official recurrence kernel can apply q/k L2 internally.
                    rq_l2 = rq_ref / torch.sqrt((rq_ref * rq_ref).sum(-1, keepdim=True) + 1e-6)
                    rk_l2 = rk_ref / torch.sqrt((rk_ref * rk_ref).sum(-1, keepdim=True) + 1e-6)
                    all_ok &= report("q_l2_vs_reference_fp32", rq_l2, q_norm, args.tolerance)
                    all_ok &= report("k_l2_vs_reference_fp32", rk_l2, k_norm, args.tolerance)

            print(f"  l2_calls_this_token={len(new_l2_in)} recurrence_calls_this_token={len(reference_calls) - rec_before}")

    finally:
        qwen.causal_conv1d_update, qwen.causal_conv1d_fn = conv_originals[1], conv_originals[0]
        qwen.torch_recurrent_gated_delta_rule = original_recurrent
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
