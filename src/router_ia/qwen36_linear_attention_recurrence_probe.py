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
        f"  {name:<34} {'PASS' if ok else 'FAIL'} "
        f"max_abs={s[0]:.6g} mean_abs={s[1]:.6g} "
        f"rel={s[2]:.6g} cosine={s[3]:.9f}"
    )
    return ok


def _arg(call_args, call_kwargs, index: int, name: str, default=None):
    if len(call_args) > index:
        return call_args[index]
    return call_kwargs.get(name, default)


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
    if not 0 <= args.layer < base.DEFAULT_LAYERS:
        raise SystemExit(f"--layer must be in [0, {base.DEFAULT_LAYERS - 1}]")

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

    # The reference forward uses module-level dispatch symbols. Depending on
    # cache state, token 0 can use the chunk implementation while later tokens
    # use the recurrent implementation. Patch the actual module-level recurrent
    # symbol and do not assume it is an attribute of the layer instance.
    original_recurrent = qwen.torch_recurrent_gated_delta_rule
    original_chunk = qwen.torch_chunk_gated_delta_rule
    original_l2 = attention._l2norm

    runtime_l2: list[torch.Tensor] = []
    reference_calls: list[dict[str, torch.Tensor | None]] = []
    reference_states: list[torch.Tensor | None] = []

    def capture_runtime_l2(x, eps=1e-6):
        out = original_l2(x, eps)
        runtime_l2.append(out.detach().clone())
        return out

    def capture_recurrent(*call_args, **call_kwargs):
        query = _arg(call_args, call_kwargs, 0, "query")
        key = _arg(call_args, call_kwargs, 1, "key")
        value = _arg(call_args, call_kwargs, 2, "value")
        g = _arg(call_args, call_kwargs, 3, "g")
        beta = _arg(call_args, call_kwargs, 4, "beta")
        initial = _arg(call_args, call_kwargs, 5, "initial_state")
        output_final_state = _arg(call_args, call_kwargs, 6, "output_final_state", False)
        use_qk = _arg(call_args, call_kwargs, 7, "use_qk_l2norm_in_kernel", False)

        reference_calls.append(
            {
                "q": query.detach().clone(),
                "k": key.detach().clone(),
                "v": value.detach().clone(),
                "g": g.detach().clone(),
                "beta": beta.detach().clone(),
                "initial": None if initial is None else initial.detach().clone(),
            }
        )

        result = original_recurrent(
            query,
            key,
            value,
            g,
            beta,
            initial_state=initial,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk,
        )
        final_state = result[1] if isinstance(result, tuple) and len(result) == 2 else None
        reference_states.append(None if final_state is None else final_state.detach().clone())
        return result

    qwen.torch_recurrent_gated_delta_rule = capture_recurrent
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

            print(f"\nTOKEN {pos}")
            all_ok &= report("linear_output", reference, got_linear, args.tolerance)

            # Exactly one new recurrent call should appear for tokens after the
            # initial chunk path. Capture count is used instead of assuming a
            # particular layer attribute exists.
            if len(reference_calls) == 0:
                print("  recurrence_capture                SKIP (no recurrent dispatch)")
                if pos > 0:
                    print("  recurrence_dispatch               FAIL (expected recurrent dispatch)")
                    all_ok = False
                continue

            rr = reference_calls[-1]
            rs = reference_states[-1]
            l2 = runtime_l2[-2:] if len(runtime_l2) >= 2 else []

            qf = rr["q"].float()
            kf = rr["k"].float()
            q_native = rr["q"] / torch.sqrt((rr["q"] * rr["q"]).sum(dim=-1, keepdim=True) + 1e-6)
            k_native = rr["k"] / torch.sqrt((rr["k"] * rr["k"]).sum(dim=-1, keepdim=True) + 1e-6)
            q_fp32 = qf / torch.sqrt((qf * qf).sum(dim=-1, keepdim=True) + 1e-6)
            k_fp32 = kf / torch.sqrt((kf * kf).sum(dim=-1, keepdim=True) + 1e-6)

            print(f"  recurrence_calls_seen={len(reference_calls)}")
            if len(l2) == 2:
                all_ok &= report("q_l2_vs_reference_native", q_native.transpose(1, 2).float(), l2[0], args.tolerance)
                all_ok &= report("q_l2_vs_reference_fp32", q_fp32.transpose(1, 2), l2[0], args.tolerance)
                all_ok &= report("k_l2_vs_reference_native", k_native.transpose(1, 2).float(), l2[1], args.tolerance)
                all_ok &= report("k_l2_vs_reference_fp32", k_fp32.transpose(1, 2), l2[1], args.tolerance)
            else:
                print("  q/k_runtime_l2_capture            UNAVAILABLE")
                all_ok = False

            if rs is not None:
                all_ok &= report("recurrent_final_state", rs, state.linear_states[args.layer], args.tolerance)
            else:
                print("  recurrent_final_state              UNAVAILABLE")
                all_ok = False

            print(
                f"  reference_q_dtype={rr['q'].dtype} reference_k_dtype={rr['k'].dtype} "
                f"reference_v_dtype={rr['v'].dtype} reference_g_dtype={rr['g'].dtype} "
                f"reference_beta_dtype={rr['beta'].dtype}"
            )

    finally:
        qwen.torch_recurrent_gated_delta_rule = original_recurrent
        qwen.torch_chunk_gated_delta_rule = original_chunk
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
