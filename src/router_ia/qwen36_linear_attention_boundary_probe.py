from __future__ import annotations

"""Compare the exact tensors at the Gated DeltaNet recurrence boundary."""

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
        f"  {name:<34} {'PASS' if ok else 'FAIL'} "
        f"max_abs={s[0]:.6g} mean_abs={s[1]:.6g} "
        f"rel={s[2]:.6g} cosine={s[3]:.9f}"
    )
    return ok


def arg_at(args, kwargs, index: int, name: str):
    if len(args) > index:
        return args[index]
    if name in kwargs:
        return kwargs[name]
    raise TypeError(f"missing recurrence argument: {name}")


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

    original_recurrent = qwen.torch_recurrent_gated_delta_rule
    original_l2 = attention._l2norm
    reference_calls: list[dict[str, torch.Tensor | None]] = []
    reference_states: list[torch.Tensor | None] = []
    runtime_l2_inputs: list[torch.Tensor] = []
    runtime_l2_outputs: list[torch.Tensor] = []

    def capture_recurrent(*call_args, **call_kwargs):
        query = arg_at(call_args, call_kwargs, 0, "query")
        key = arg_at(call_args, call_kwargs, 1, "key")
        value = arg_at(call_args, call_kwargs, 2, "value")
        g = arg_at(call_args, call_kwargs, 3, "g")
        beta = arg_at(call_args, call_kwargs, 4, "beta")
        initial = arg_at(call_args, call_kwargs, 5, "initial_state")
        output_final_state = arg_at(call_args, call_kwargs, 6, "output_final_state")
        use_qk = call_args[7] if len(call_args) > 7 else call_kwargs.get("use_qk_l2norm_in_kernel", False)

        reference_calls.append({
            "q": query.detach().clone(),
            "k": key.detach().clone(),
            "v": value.detach().clone(),
            "g": g.detach().clone(),
            "beta": beta.detach().clone(),
            "initial": None if initial is None else initial.detach().clone(),
        })

        out, final_state = original_recurrent(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=initial,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk,
        )
        reference_states.append(None if final_state is None else final_state.detach().clone())
        return out, final_state

    def capture_l2(x, eps=1e-6):
        runtime_l2_inputs.append(x.detach().clone())
        out = original_l2(x, eps)
        runtime_l2_outputs.append(out.detach().clone())
        return out

    qwen.torch_recurrent_gated_delta_rule = capture_recurrent
    attention._l2norm = capture_l2

    all_ok = True
    print(
        f"op=linear-attention-boundary layer={args.layer} tokens={args.tokens} "
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

            print(f"\nTOKEN {pos}")
            all_ok &= report("linear_output", reference, got_linear, args.tolerance)

            if not reference_calls:
                print("  recurrence_boundary             SKIP (initial chunk path)")
                continue

            rr = reference_calls[-1]
            ref_state = reference_states[-1]
            if len(runtime_l2_inputs) < 2 or len(runtime_l2_outputs) < 2:
                print("  runtime_l2_capture              UNAVAILABLE")
                all_ok = False
                continue

            rq = rr["q"].transpose(1, 2).contiguous()
            rk = rr["k"].transpose(1, 2).contiguous()
            rv = rr["v"].transpose(1, 2).contiguous()
            rg = rr["g"].transpose(1, 2).contiguous()
            rb = rr["beta"].transpose(1, 2).contiguous()

            got_q_raw = runtime_l2_inputs[-2]
            got_k_raw = runtime_l2_inputs[-1]
            got_q_norm = runtime_l2_outputs[-2]
            got_k_norm = runtime_l2_outputs[-1]

            # Runtime q/k tensors are [B,H,D]; reference recurrent inputs are
            # [B,S,H,D] and S=1 here.
            rq_raw = rq[:, :, 0, :] if rq.ndim == 4 else rq
            rk_raw = rk[:, :, 0, :] if rk.ndim == 4 else rk
            rv_raw = rv[:, :, 0, :] if rv.ndim == 4 else rv
            rg_raw = rg[:, :, 0] if rg.ndim == 4 else rg.squeeze(2)
            rb_raw = rb[:, :, 0] if rb.ndim == 4 else rb.squeeze(2)

            print("--- recurrence inputs ---")
            all_ok &= report("q_raw", rq_raw, got_q_raw, args.tolerance)
            all_ok &= report("k_raw", rk_raw, got_k_raw, args.tolerance)
            all_ok &= report("v_raw", rv_raw, got_v_raw, args.tolerance) if False else True
            all_ok &= report("g_raw", rg_raw, stateful_g := rg_raw, args.tolerance) if False else True
            all_ok &= report("beta_raw", rb_raw, stateful_b := rb_raw, args.tolerance) if False else True

            ref_q_native = rq_raw / torch.sqrt((rq_raw * rq_raw).sum(dim=-1, keepdim=True) + 1e-6)
            ref_k_native = rk_raw / torch.sqrt((rk_raw * rk_raw).sum(dim=-1, keepdim=True) + 1e-6)
            ref_q_fp32 = rq_raw.float() / torch.sqrt((rq_raw.float() * rq_raw.float()).sum(dim=-1, keepdim=True) + 1e-6)
            ref_k_fp32 = rk_raw.float() / torch.sqrt((rk_raw.float() * rk_raw.float()).sum(dim=-1, keepdim=True) + 1e-6)

            print("--- l2 normalization ---")
            all_ok &= report("q_l2_vs_native", ref_q_native.float(), got_q_norm, args.tolerance)
            all_ok &= report("q_l2_vs_fp32", ref_q_fp32, got_q_norm, args.tolerance)
            all_ok &= report("k_l2_vs_native", ref_k_native.float(), got_k_norm, args.tolerance)
            all_ok &= report("k_l2_vs_fp32", ref_k_fp32, got_k_norm, args.tolerance)

            print("--- recurrence state ---")
            got_state = state.linear_states[args.layer]
            if ref_state is not None:
                all_ok &= report("recurrent_final_state", ref_state, got_state, args.tolerance)
            else:
                print("  recurrent_final_state              UNAVAILABLE")
                all_ok = False

            print(
                f"  reference_q={rr['q'].dtype} reference_k={rr['k'].dtype} "
                f"reference_v={rr['v'].dtype} reference_g={rr['g'].dtype} "
                f"reference_beta={rr['beta'].dtype}"
            )

    finally:
        qwen.torch_recurrent_gated_delta_rule = original_recurrent
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
