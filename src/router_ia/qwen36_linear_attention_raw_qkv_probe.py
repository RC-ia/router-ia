from __future__ import annotations

"""Test Linear Attention against directly dequantized FP8 QKV weights."""

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
from .qwen36_layer_fidelity_probe import _build_meta_model, _find_layers, _load_config, _materialize_layer, _module_input_dtype, _stage_stats
from .qwen36_op_probe import load_embedding_row, rmsnorm


def report(name: str, ref: torch.Tensor, got: torch.Tensor, tol: float) -> None:
    s = _stage_stats(ref, got)
    print(f"  {name:<34} {'PASS' if s[0] <= tol else 'FAIL'} max_abs={s[0]:.6g} mean_abs={s[1]:.6g} rel={s[2]:.6g} cosine={s[3]:.9f}")


def load_raw_weight(root: Path, name: str) -> tuple[torch.Tensor, torch.Tensor | None]:
    index = root / "model.safetensors.index.json"
    if index.is_file():
        payload = json.loads(index.read_text(encoding="utf-8"))
        shard_name = payload["weight_map"][name]
        shards = [root / shard_name]
    else:
        shards = sorted(root.glob("*.safetensors"))
    scale_name = name.replace(".weight", ".weight_scale_inv")
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            if name in handle.keys():
                return handle.get_tensor(name), handle.get_tensor(scale_name) if scale_name in handle.keys() else None
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
    raw_w_cpu, raw_scale_cpu = load_raw_weight(root, prefix + ".weight")
    raw_w = raw_w_cpu.to(args.device)
    if raw_w.dtype == torch.float8_e4m3fn:
        if raw_scale_cpu is None:
            raise SystemExit("FP8 QKV weight has no weight_scale_inv tensor")
        deq = dequantize_fp8_blockwise(raw_w, raw_scale_cpu.to(args.device))
        deq_test = deq.to(torch.bfloat16 if args.device == "cuda" else torch.float32)
    else:
        deq = raw_w.float()
        deq_test = deq.to(torch.bfloat16 if args.device == "cuda" else torch.float32)

    print(f"op=linear-attention-raw-qkv layer={args.layer} tokens={args.tokens} device={args.device} tolerance={args.tolerance} materialized={loaded}/{total}")
    print(f"raw_weight_dtype={raw_w.dtype} raw_weight_shape={tuple(raw_w.shape)} dequant_dtype={deq.dtype} test_dtype={deq_test.dtype}")

    ref_cache = _make_reference_cache(config)
    qwen, originals = _patch_official_conv()
    state = attention.state_for(root, args.device)
    state.reset()
    attention.activate(root, state)
    original_projection = attention._projection

    def projection(root_path, proj_prefix, dev):
        if dev == "cuda" and proj_prefix == prefix:
            return deq_test
        return original_projection(root_path, proj_prefix, dev)

    attention._projection = projection
    ok = True
    try:
        for pos, raw in enumerate(tokens):
            state.reset()
            ref_cache = _make_reference_cache(config)
            token = raw.to(dtype=input_dtype)
            normed = rmsnorm(token, input_norm)
            ref_out = layer.linear_attn(hidden_states=normed.unsqueeze(1), cache_params=ref_cache, attention_mask=None)
            if isinstance(ref_out, tuple):
                ref_out = ref_out[0]
            ref_out = ref_out.reshape(1, base.HIDDEN)

            ref_w = layer.linear_attn.in_proj_qkv.weight.detach()
            ref_qkv = F.linear(normed.to(ref_w.dtype), ref_w)
            deq_qkv = F.linear(normed.to(deq_test.dtype), deq_test)
            print(f"\nTOKEN {pos}")
            report("dequant_weight_vs_module", ref_w, deq_test, args.tolerance)
            report("qkv_dequant_vs_module", ref_qkv, deq_qkv, args.tolerance)
            got = attention.step_attention(root, args.layer, token, args.device) - token.float()
            report("linear_with_raw_dequant_qkv", ref_out, got, args.tolerance)
    finally:
        attention._projection = original_projection
        attention.deactivate(root)
        qwen.causal_conv1d_fn, qwen.causal_conv1d_update = originals
        layer.to_empty(device="meta")
        del meta
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
