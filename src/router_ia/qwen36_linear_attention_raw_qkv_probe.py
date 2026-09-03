from __future__ import annotations

"""Test Linear Attention with the raw safetensors BF16 QKV weight."""

import argparse
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from . import qwen36_attention_cache as attention
from . import qwen36_40layer_loop as base
from .qwen36_linear_attention_stateful_probe import _make_reference_cache, _patch_official_conv
from .qwen36_layer_fidelity_probe import _build_meta_model, _find_layers, _load_config, _materialize_layer, _module_input_dtype, _stage_stats
from .qwen36_op_probe import load_embedding_row, rmsnorm


def report(name: str, ref: torch.Tensor, got: torch.Tensor, tol: float) -> None:
    s = _stage_stats(ref, got)
    print(f"  {name:<34} {'PASS' if s[0] <= tol else 'FAIL'} max_abs={s[0]:.6g} mean_abs={s[1]:.6g} rel={s[2]:.6g} cosine={s[3]:.9f}")


def load_raw_weight(root: Path, name: str) -> torch.Tensor:
    index = root / "model.safetensors.index.json"
    if index.is_file():
        payload = json.loads(index.read_text(encoding="utf-8"))
        shard_name = payload["weight_map"][name]
        shards = [root / shard_name]
    else:
        shards = sorted(root.glob("*.safetensors"))
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            if name in handle.keys():
                return handle.get_tensor(name)
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

    prefix = base.layer_prefix(args.layer) + "linear_attn.in_proj_qkv"
    raw_w = load_raw_weight(root, prefix + ".weight").to(args.device)
    print(f"op=linear-attention-raw-qkv layer={args.layer} tokens={args.tokens} device={args.device} tolerance={args.tolerance} materialized={loaded}/{total}")
    print(f"raw_weight_dtype={raw_w.dtype} raw_weight_shape={tuple(raw_w.shape)}")

    ref_cache = _make_reference_cache(config)
    qwen, originals = _patch_official_conv()
    state = attention.state_for(root, args.device)
    state.reset()
    attention.activate(root, state)
    original_projection = attention._projection

    def projection(root_path, proj_prefix, dev):
        if dev == "cuda" and proj_prefix == prefix:
            return raw_w
        return original_projection(root_path, proj_prefix, dev)

    attention._projection = projection
    hook = layer.linear_attn.in_proj_qkv.register_forward_hook(lambda module, inp, out: None)
    ok = True
    try:
        for pos, raw in enumerate(tokens):
            token = raw.to(dtype=input_dtype)
            normed = rmsnorm(token, input_norm)
            ref = layer.linear_attn(hidden_states=normed.unsqueeze(1), cache_params=ref_cache, attention_mask=None)
            if isinstance(ref, tuple): ref = ref[0]
            ref = ref.reshape(1, base.HIDDEN)

            ref_w = layer.linear_attn.in_proj_qkv.weight.detach()
            ref_qkv = F.linear(normed.to(ref_w.dtype), ref_w)
            raw_qkv = F.linear(normed.to(raw_w.dtype), raw_w)
            print(f"\nTOKEN {pos}")
            report("raw_weight_vs_module", ref_w, raw_w, args.tolerance)
            report("qkv_module_reference", ref_qkv, ref_qkv, args.tolerance)
            report("qkv_raw_weight", ref_qkv, raw_qkv, args.tolerance)

            got = attention.step_attention(root, args.layer, token, args.device)
            got = got - token.float()
            report("linear_with_raw_bf16_qkv", ref, got, args.tolerance)
    finally:
        attention._projection = original_projection
        attention.deactivate(root)
        qwen.causal_conv1d_fn, qwen.causal_conv1d_update = originals
        hook.remove()
        layer.to_empty(device="meta")
        del meta
        gc.collect()
        if args.device == "cuda": torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
