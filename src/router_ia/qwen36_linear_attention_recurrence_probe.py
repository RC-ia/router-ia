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
        f"  {name:<32} {'PASS' if ok else 'FAIL'} "
        f"max_abs={s[0]:.6g} mean_abs={s[1]:.6g} "
        f"rel={s[2]:.6g} cosine={s[3]:.9f}"
    )
    return ok


def _call_args(args, kwargs, index, name):
    if len(args) > index:
        return args[index]
    value = kwargs.get(name)
    if value is None:
        raise TypeError(f"missing recurrence argument: {name}")
    return value


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

    original_l2 = attention._l2norm
    original_recurrent = getattr(layer.linear_attn, "recurrent_gated_delta_rule", None)
    original_chunk = getattr(layer.linear_attn, "chunk_gated_delta_rule", None)

    runtime_l2: list[torch.Tensor] = []
    reference_calls: list[dict[str, torch.Tensor | None]] = []
    reference_states: list[torch.Tensor | None] = []

    def capture_runtime_l2(x, eps=1e-6):
        out = original_l2(x, eps)
        runtime_l2.append(out.detach().clone())
        return out

    def capture_recurrent(*call_args, **call_kwargs):
        query = _call_args(call_args, call_kwargs, 0, "query")
        key = _call_args(call_args, call_kwargs, 1, "key")
        value = _call_args(call_args, call_kwargs, 2, "value")
        g = _call_args(call_args, call_kwargs, 3, "g")
        beta = _call_args(call_args, call_kwargs, 4, "beta")
        initial = _call_args(call_args, call_kwargs, 5, "initial_state")
        output_final_state = _call_args(call_args, call_kwargs, 6, "output_final_state")
        use_qk = call_args[7] if len(call_args) > 7 else call_kwargs.get("use_qk_l2norm_in_kernel", False)

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

        # Force the official pure-PyTorch implementation and preserve the same
        # normalization flag used by the model. This lets us compare q/k exactly
        # at the recurrence boundary.
        out, final_state = qwen.torch_recurrent_gated_delta_rule(
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

    # The instantiated Qwen module stores the chosen recurrent callable on the
    # instance, so patch the bound attribute directly. The first token uses the
    # chunk path; later tokens use this recurrent path.
    if original_recurrent is None:
        raise SystemExit("Reference layer has no recurrent_gated_delta_rule attribute")
    layer.linear_attn.recurrent_gated_delta_rule = capture_recurrent
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

            # First token intentionally goes through the chunk implementation and
            # therefore cannot populate reference_calls. We compare recurrence
            # stages only once the reference switches to cached single-token mode.
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
                print("  recurrence_capture              SKIP (initial chunk path)")
                continue

            rr = reference_calls[-1]
            ref_state = reference_states[-1]
            l2 = runtime_l2[-2:] if len(runtime_l2) >= 2 else []

            ref_q_native = rr["q"] / torch.sqrt((rr["q"] * rr["q"]).sum(dim=-1, keepdim=True) + 1e-6)
            ref_k_native = rr["k"] / torch.sqrt((rr["k"] * rr["k"]).sum(dim=-1, keepdim=True) + 1e-6)
            qf = rr["q"].float()
            kf = rr["k"].float()
            ref_q_fp32 = qf / torch.sqrt((qf * qf).sum(dim=-1, keepdim=True) + 1e-6)
            ref_k_fp32 = kf / torch.sqrt((kf * kf).sum(dim=-1, keepdim=True) + 1e-6)

            if len(l2) == 2:
                all_ok &= report("q_l2_vs_reference_native", ref_q_native.transpose(1, 2).float(), l2[0], args.tolerance)
                all_ok &= report("q_l2_vs_reference_fp32", ref_q_fp32.transpose(1, 2), l2[0], args.tolerance)
                all_ok &= report("k_l2_vs_reference_native", ref_k_native.transpose(1, 2).float(), l2[1], args.tolerance)
                all_ok &= report("k_l2_vs_reference_fp32", ref_k_fp32.transpose(1, 2), l2[1], args.tolerance)
            else:
                print("  q/k_runtime_l2_capture          UNAVAILABLE")
                all_ok = False

            got_state = state.linear_states[args.layer]
            if ref_state is not None:
                all_ok &= report("recurrent_final_state", ref_state, got_state, args.tolerance)
            else:
                print("  recurrent_final_state             UNAVAILABLE")
                all_ok = False

            print(
                f"  reference_q_dtype={rr['q'].dtype} reference_k_dtype={rr['k'].dtype} "
                f"reference_v_dtype={rr['v'].dtype} reference_g_dtype={rr['g'].dtype} "
                f"reference_beta_dtype={rr['beta'].dtype}"
            )
    finally:
        layer.linear_attn.recurrent_gated_delta_rule = original_recurrent
        if original_chunk is not None:
            layer.linear_attn.chunk_gated_delta_rule = original_chunk
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
