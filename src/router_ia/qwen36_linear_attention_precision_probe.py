from __future__ import annotations

"""Compare Linear Attention with cached projection weights vs bf16 projection weights."""

import argparse
from pathlib import Path

import torch

from . import qwen36_attention_cache as attention
from . import qwen36_40layer_loop as base
from .qwen36_linear_attention_state_compare import report
from .qwen36_linear_attention_stateful_probe import _make_reference_cache, _patch_official_conv
from .qwen36_layer_fidelity_probe import _build_meta_model, _find_layers, _load_config, _materialize_layer, _module_input_dtype
from .qwen36_op_probe import load_embedding_row, rmsnorm


def run(root: Path, layer_idx: int, tokens: int, token_id: int, device: str, tolerance: float, bf16: bool) -> bool:
    config = _load_config(root)
    meta = _build_meta_model(config)
    layers = _find_layers(meta)
    layer = layers[layer_idx]
    loaded, total = _materialize_layer(root, layer, layer_idx, device)
    print(f"op=linear-attention-precision layer={layer_idx} tokens={tokens} device={device} bf16_projections={bf16} tolerance={tolerance}")
    print(f"materialized={loaded}/{total}")

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    raw_tokens = [load_embedding_row(root, token_id + i).reshape(1, base.HIDDEN).to(device).to(dtype) for i in range(tokens)]
    input_dtype = _module_input_dtype(layer)
    input_norm = base.load_layer_weight(root, layer_idx, "input_layernorm.weight", device)
    ref_cache = _make_reference_cache(config)
    state = attention.state_for(root, device)
    state.reset()
    attention.activate(root, state)
    qwen, originals = _patch_official_conv()
    original_projection = attention._projection

    def projection(root_path, prefix, dev):
        w = original_projection(root_path, prefix, dev)
        if bf16 and dev == "cuda":
            return w.to(torch.bfloat16)
        return w

    attention._projection = projection
    ok = True
    try:
        for pos, raw in enumerate(raw_tokens):
            token = raw.to(dtype=input_dtype)
            normed = rmsnorm(token, input_norm)
            ref = layer.linear_attn(hidden_states=normed.unsqueeze(1), cache_params=ref_cache, attention_mask=None)
            if isinstance(ref, tuple):
                ref = ref[0]
            ref = ref.reshape(1, base.HIDDEN)

            got = attention.step_attention(root, layer_idx, token, device)
            got = got - token.float()
            ok &= report(f"token{pos}_linear", ref, got, tolerance)
            runtime_state = state.linear_states[layer_idx]
            print(f"  recurrent_state_norm={float(torch.linalg.vector_norm(runtime_state).item()):.8g}")
    finally:
        attention._projection = original_projection
        attention.deactivate(root)
        qwen.causal_conv1d_fn, qwen.causal_conv1d_update = originals
        layer.to_empty(device="meta")
        del meta
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"RESULT status={'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--layer", type=int, default=1)
    p.add_argument("--tokens", type=int, default=4)
    p.add_argument("--token-id", type=int, default=0)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--tolerance", type=float, default=1e-3)
    p.add_argument("--bf16", action="store_true", help="cast cached Linear Attention projection weights to bf16")
    args = p.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    if not 0 <= args.layer < base.DEFAULT_LAYERS:
        raise SystemExit(f"--layer must be in [0, {base.DEFAULT_LAYERS - 1}]")
    root = args.root.resolve()
    return 0 if run(root, args.layer, args.tokens, args.token_id, args.device, args.tolerance, args.bf16) else 1


if __name__ == "__main__":
    raise SystemExit(main())
