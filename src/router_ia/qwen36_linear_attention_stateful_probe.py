from __future__ import annotations

"""Stateful fidelity probe for Qwen3.6 Gated DeltaNet / linear attention."""

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
from .qwen36_op_probe import load_embedding_row, rmsnorm

DEFAULT_TOLERANCE = 1e-3


def report(name: str, reference: torch.Tensor, candidate: torch.Tensor, tolerance: float) -> bool:
    stats = _stage_stats(reference, candidate)
    status = "PASS" if stats[0] <= tolerance else "FAIL"
    print(
        f"  {name:<28} {status} "
        f"max_abs={stats[0]:.6g} mean_abs={stats[1]:.6g} "
        f"rel={stats[2]:.6g} cosine={stats[3]:.9f} "
        f"ref_norm={stats[4]:.6g} router_norm={stats[5]:.6g}"
    )
    return status == "PASS"


def _patch_official_conv():
    """Force the reference linear-attention conv through the PyTorch fallback."""
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as qwen

    originals = (qwen.causal_conv1d_fn, qwen.causal_conv1d_update)

    def conv_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
        return _pure_torch_causal_conv1d(hidden_states, weight, bias, activation, **kwargs)

    def conv_update(hidden_states, conv_state, weight, bias=None, activation=None):
        state_len = conv_state.shape[-1]
        mixed = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
        conv_state.copy_(mixed[:, :, -state_len:])
        out = F.conv1d(mixed, weight.unsqueeze(1), bias, padding=0, groups=hidden_states.shape[1])
        out = out[:, :, -hidden_states.shape[-1]:]
        if activation is not None:
            out = F.silu(out)
        return out.to(hidden_states.dtype)

    qwen.causal_conv1d_fn = conv_fn
    qwen.causal_conv1d_update = conv_update
    return qwen, originals


def _restore_official_conv(qwen, originals):
    qwen.causal_conv1d_fn, qwen.causal_conv1d_update = originals


def _make_reference_cache(config):
    from transformers import DynamicCache

    try:
        return DynamicCache(config=config)
    except TypeError:
        return DynamicCache()


def _cache_has_state(cache, layer_idx: int) -> bool:
    try:
        return bool(cache.has_previous_state(layer_idx))
    except TypeError:
        try:
            return bool(cache.has_previous_state(layer_idx, state_idx=0))
        except Exception:
            return False


def _cache_shapes(cache, layer_idx: int) -> tuple[str, str]:
    if layer_idx >= len(cache.layers):
        return "missing", "missing"
    layer_cache = cache.layers[layer_idx]
    conv = getattr(layer_cache, "conv_states", None)
    rec = getattr(layer_cache, "recurrent_states", None)
    return str(tuple(conv.shape) if torch.is_tensor(conv) else type(conv).__name__), str(
        tuple(rec.shape) if torch.is_tensor(rec) else type(rec).__name__
    )


def run_layer(
    root: Path,
    layer_idx: int,
    layer,
    hidden_tokens: list[torch.Tensor],
    device: str,
    tolerance: float,
    seed: int,
) -> bool:
    torch.manual_seed(seed)
    ref_cache = _make_reference_cache(_load_config(root))

    state = attention.state_for(root, device)
    state.reset()
    attention.activate(root, state)

    qwen, conv_originals = _patch_official_conv()
    all_pass = True
    print(f"\n===== LINEAR LAYER {layer_idx} =====")
    print(f"tokens={len(hidden_tokens)} device={device} tolerance={tolerance}")

    try:
        input_dtype = _module_input_dtype(layer)
        input_norm = None
        for position, token in enumerate(hidden_tokens):
            token = token.to(dtype=input_dtype)
            print(f"\n--- TOKEN {position} ---")

            # DecoderLayer applies input_layernorm before dispatching into
            # linear_attn. The runtime's step_attention does the same internally,
            # so the official reference must receive the normalized hidden state.
            if input_norm is None:
                input_norm = base.load_layer_weight(root, layer_idx, "input_layernorm.weight", device)
            normed = rmsnorm(token, input_norm)

            # Official Transformers reference. The first token goes through the
            # chunk path; later single-token calls reuse conv + recurrent state.
            reference = layer.linear_attn(
                hidden_states=normed.unsqueeze(1),
                cache_params=ref_cache,
                attention_mask=None,
            )
            if isinstance(reference, tuple):
                reference = reference[0]
            reference = reference.reshape(1, base.HIDDEN)

            # Runtime keeps the post-linear-attention residual, so remove the
            # input residual before comparing against the official linear output.
            runtime_residual = attention.step_attention(root, layer_idx, token, device)
            runtime_linear = runtime_residual - token.reshape(1, base.HIDDEN).float()

            all_pass &= report("linear_attention_output", reference, runtime_linear, tolerance)

            ref_has_state = _cache_has_state(ref_cache, layer_idx)
            runtime_has_state = (
                layer_idx in state.linear_states and layer_idx in state.linear_conv_states
            )
            print(
                f"  cache_state_presence         "
                f"{'PASS' if ref_has_state == runtime_has_state else 'FAIL'} "
                f"reference={ref_has_state} runtime={runtime_has_state}"
            )
            all_pass &= ref_has_state == runtime_has_state

            conv_shape, rec_shape = _cache_shapes(ref_cache, layer_idx)
            runtime_conv = state.linear_conv_states.get(layer_idx)
            runtime_rec = state.linear_states.get(layer_idx)
            if runtime_conv is not None and runtime_rec is not None:
                print(
                    f"  reference_cache_shapes       conv={conv_shape} recurrent={rec_shape} "
                    f"runtime_conv={tuple(runtime_conv.shape)} runtime_recurrent={tuple(runtime_rec.shape)}"
                )

            print(f"  transition={('initial' if position == 0 else 'recurrent')} ")

    finally:
        _restore_official_conv(qwen, conv_originals)
        attention.deactivate(root)

    print(f"\nLAYER {layer_idx} RESULT: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen3.6 stateful linear-attention fidelity probe")
    parser.add_argument("root", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--token-id", type=int, default=0)
    parser.add_argument("--all-linear", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args()

    if args.tokens <= 0:
        raise SystemExit("--tokens must be > 0")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")

    root = args.root.resolve()
    config = _load_config(root)
    meta_model = _build_meta_model(config)
    layers = _find_layers(meta_model)

    linear_layers = [
        idx for idx in range(base.DEFAULT_LAYERS)
        if base.attention_type(root, idx) == "linear_attention"
    ]
    if not linear_layers:
        raise SystemExit("No linear_attention layers found")

    if args.all_linear:
        target_layers = linear_layers
    else:
        if not 0 <= args.layer < base.DEFAULT_LAYERS:
            raise SystemExit(f"--layer must be in [0, {base.DEFAULT_LAYERS - 1}]")
        if args.layer not in linear_layers:
            raise SystemExit(f"Layer {args.layer} is not linear_attention")
        target_layers = [args.layer]

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    base_hidden = load_embedding_row(root, args.token_id).reshape(1, base.HIDDEN).to(args.device).to(dtype)
    hidden_tokens = [
        base_hidden if position == 0 else load_embedding_row(root, args.token_id + position).reshape(1, base.HIDDEN).to(args.device).to(dtype)
        for position in range(args.tokens)
    ]

    print("op=linear-attention-stateful-fidelity")
    print(f"layers={target_layers}")
    print(f"tokens={args.tokens}")
    print(f"token_id_start={args.token_id}")
    print(f"device={args.device}")
    print(f"tolerance={args.tolerance}")

    results: list[bool] = []
    try:
        for layer_idx in target_layers:
            layer = layers[layer_idx]
            loaded, total = _materialize_layer(root, layer, layer_idx, args.device)
            print(f"\nmaterialized layer {layer_idx}: {loaded}/{total}")
            results.append(
                run_layer(
                    root,
                    layer_idx,
                    layer,
                    hidden_tokens,
                    args.device,
                    args.tolerance,
                    args.seed,
                )
            )
            layer.to_empty(device="meta")
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()
    finally:
        del meta_model
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    ok = all(results)
    print("\n=== FINAL RESULT ===")
    print(f"linear_layers_tested={len(results)}")
    print(f"status={'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
