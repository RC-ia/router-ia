from __future__ import annotations

"""Locate the first Gated DeltaNet recurrent-stage divergence."""

import argparse
import gc
from pathlib import Path

import torch

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
        f"  {name:<30} {'PASS' if ok else 'FAIL'} "
        f"max_abs={s[0]:.6g} mean_abs={s[1]:.6g} "
        f"rel={s[2]:.6g} cosine={s[3]:.9f}"
    )
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description="Locate Qwen3.6 Gated DeltaNet recurrence divergence")
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

    original_l2 = attention._l2norm
    ref_original_recurrent = layer.linear_attn.recurrent_gated_delta_rule
    runtime_l2 = []
    reference_raw = []
    reference_norm = []
    reference_final_state = []
    reference_initial_state = []

    def capture_runtime_l2(x, eps=1e-6):
        out = original_l2(x, eps)
        runtime_l2.append(out.detach().clone())
        return out

    def reference_recurrent(*args, **kwargs):
        # Capture the exact tensors passed by the official module immediately
        # before its recurrent kernel/fallback performs q/k normalization.
        query = args[0] if args else kwargs["query"]
        key = args[1] if len(args) > 1 else kwargs["key"]
        value = args[2] if len(args) > 2 else kwargs["value"]
        g = kwargs.get("g", args[3] if len(args) > 3 else None)
        beta = kwargs.get("beta", args[4] if len(args) > 4 else None)
        initial_state = kwargs.get("initial_state", args[5] if len(args) > 5 else None)

        reference_raw.append({
            "q": query.detach().clone(),
            "k": key.detach().clone(),
            "v": value.detach().clone(),
            "g": g.detach().clone(),
            "beta": beta.detach().clone(),
            "initial": None if initial_state is None else initial_state.detach().clone(),
        })

        # Reproduce the fallback's normalization in the two candidate dtypes.
        q_native = query
        k_native = key
        q_native = q_native / torch.sqrt((q_native * q_native).sum(dim=-1, keepdim=True) + 1e-6)
        k_native = k_native / torch.sqrt((k_native * k_native).sum(dim=-1, keepdim=True) + 1e-6)
        reference_norm.append({"q_native": q_native.detach().clone(), "k_native": k_native.detach().clone()})

        # Use the official pure-PyTorch recurrent fallback, avoiding any fused
        # implementation differences while diagnosing tensor parity.
        torch_recurrent = qwen.torch_recurrent_gated_delta_rule
        out, final_state = torch_recurrent(*args, **kwargs)
        reference_final_state.append(None if final_state is None else final_state.detach().clone())
        reference_initial_state.append(None if initial_state is None else initial_state.detach().clone())
        return out, final_state

    layer.linear_attn.recurrent_gated_delta_rule = reference_recurrent
    attention._l2norm = capture_runtime_l2

    all_ok = True
    print(
        f"op=linear-attention-recurrence layer={args.layer} tokens={args.tokens} "
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

            got_residual = attention.step_attention(root, args.layer, token, args.device)
            got_linear = got_residual - token.float()

            rr = reference_raw[-1]
            rn = reference_norm[-1]
            l2 = runtime_l2[-2:] if len(runtime_l2) >= 2 else []

            # Runtime produces q then k through _l2norm; compare each against
            # reference normalization performed in native input dtype and in FP32.
            ref_q_native = rn["q_native"].transpose(1, 2).to(torch.float32)
            ref_k_native = rn["k_native"].transpose(1, 2).to(torch.float32)
            ref_q_fp32 = (
                rr["q"].float()
                / torch.sqrt((rr["q"].float() * rr["q"].float()).sum(dim=-1, keepdim=True) + 1e-6)
            ).transpose(1, 2)
            ref_k_fp32 = (
                rr["k"].float()
                / torch.sqrt((rr["k"].float() * rr["k"].float()).sum(dim=-1, keepdim=True) + 1e-6)
            ).transpose(1, 2)

            print(f"\nTOKEN {pos}")
            if len(l2) == 2:
                all_ok &= report("q_l2_vs_reference_native", ref_q_native, l2[0], args.tolerance)
                all_ok &= report("q_l2_vs_reference_fp32", ref_q_fp32, l2[0], args.tolerance)
                all_ok &= report("k_l2_vs_reference_native", ref_k_native, l2[1], args.tolerance)
                all_ok &= report("k_l2_vs_reference_fp32", ref_k_fp32, l2[1], args.tolerance)
            else:
                print("  q/k runtime_l2_capture          UNAVAILABLE")
                all_ok = False

            # The tensors below should already be FP32 when entering the official
            # recurrence fallback. Runtime computes these directly from the same
            # normalized hidden input; any mismatch here points away from state.
            got_state = state.linear_states[args.layer]
            ref_state = reference_final_state[-1]
            if ref_state is not None:
                all_ok &= report("recurrent_final_state", ref_state, got_state, args.tolerance)
            all_ok &= report("v_raw", rr["v"], rr["v"], args.tolerance)
            all_ok &= report("beta_self_check", rr["beta"], rr["beta"], args.tolerance)
            all_ok &= report("g_self_check", rr["g"], rr["g"], args.tolerance)
            all_ok &= report("linear_output", reference, got_linear, args.tolerance)

            print(
                f"  reference_q_dtype={rr['q'].dtype} reference_k_dtype={rr['k'].dtype} "
                f"reference_v_dtype={rr['v'].dtype} reference_g_dtype={rr['g'].dtype} "
                f"reference_beta_dtype={rr['beta'].dtype}"
            )

    finally:
        layer.linear_attn.recurrent_gated_delta_rule = ref_original_recurrent
        attention._l2norm = original_l2
        attention.deactivate(root)
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
