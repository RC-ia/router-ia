from __future__ import annotations

"""Safely probe one Qwen3.6 Layer-0 operation at a time.

Default path is CPU and deliberately avoids the expert cache. Use --op to
select a single operation so failures cannot be hidden inside a full layer.
"""

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from safetensors import safe_open

HIDDEN = 2048
QKV_OUT = 8192
BLOCK = 128
CONV_KERNEL = 4
LAYER_PREFIX = "model.language_model.layers.0."
EMBEDDING_NAME = "model.language_model.embed_tokens.weight"
EPS = 1e-6


def load_tensor(root: Path, name: str, device: str = "cpu") -> torch.Tensor:
    index_path = root / "model.safetensors.index.json"
    shard_name = None
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        shard_name = payload.get("weight_map", {}).get(name)
    shards = [root / shard_name] if shard_name else sorted(root.glob("*.safetensors"))
    for shard in shards:
        if not shard.is_file():
            continue
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            if name in handle.keys():
                return handle.get_tensor(name).to(device=device)
    raise KeyError(f"Tensor not found: {name}")


def load_optional_tensor(root: Path, name: str, device: str = "cpu") -> torch.Tensor | None:
    index_path = root / "model.safetensors.index.json"
    shard_name = None
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        shard_name = payload.get("weight_map", {}).get(name)
    shards = [root / shard_name] if shard_name else sorted(root.glob("*.safetensors"))
    for shard in shards:
        if not shard.is_file():
            continue
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            if name in handle.keys():
                return handle.get_tensor(name).to(device=device)
    return None


def load_embedding_row(root: Path, token_id: int) -> torch.Tensor:
    emb = load_tensor(root, EMBEDDING_NAME, device="cpu")
    if emb.ndim != 2 or emb.shape[1] != HIDDEN:
        raise ValueError(f"Unexpected embedding shape: {tuple(emb.shape)}")
    if not 0 <= token_id < emb.shape[0]:
        raise ValueError(f"token_id {token_id} outside vocabulary")
    return emb[token_id].float()


def rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x = x.float()
    w = weight.float()
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS) * (1.0 + w)


def dequantize_fp8_blockwise(weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """Dequantize 2-D Qwen3.6 FP8 E4M3 weights with 128x128 inverse scales."""
    if weight.ndim != 2 or scale_inv.ndim != 2:
        raise ValueError(
            f"Expected 2-D weight/scale tensors, got {tuple(weight.shape)} and {tuple(scale_inv.shape)}"
        )
    out_features, in_features = map(int, weight.shape)
    expected = (
        (out_features + BLOCK - 1) // BLOCK,
        (in_features + BLOCK - 1) // BLOCK,
    )
    if tuple(scale_inv.shape) != expected:
        raise ValueError(
            f"Scale shape {tuple(scale_inv.shape)} does not match weight {tuple(weight.shape)}; "
            f"expected {expected}"
        )
    expanded = scale_inv.float().repeat_interleave(BLOCK, dim=0).repeat_interleave(BLOCK, dim=1)
    return weight.float() * expanded[:out_features, :in_features]


def dequantize_conv1d_weight(weight: torch.Tensor, scale_inv: torch.Tensor | None) -> torch.Tensor:
    """Dequantize a depthwise FP8 conv weight, accepting common scale layouts."""
    if weight.ndim != 3:
        raise ValueError(f"Expected conv weight [channels,1,kernel], got {tuple(weight.shape)}")
    if weight.shape[1] != 1 or weight.shape[2] != CONV_KERNEL:
        raise ValueError(f"Unexpected conv weight shape: {tuple(weight.shape)}")
    if weight.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        return weight.float()
    if scale_inv is None:
        raise KeyError("FP8 conv1d weight requires linear_attn.conv1d.weight_scale_inv")

    channels, _, kernel = map(int, weight.shape)
    flat = weight.reshape(channels, kernel)
    scale = scale_inv.float().reshape(-1)
    rows = (channels + BLOCK - 1) // BLOCK
    cols = (kernel + BLOCK - 1) // BLOCK
    if scale.numel() == rows:
        scale2 = scale[:, None]
    elif scale.numel() == rows * cols:
        scale2 = scale.reshape(rows, cols)
    else:
        raise ValueError(
            f"Unsupported conv scale shape {tuple(scale_inv.shape)} for weight {tuple(weight.shape)}"
        )
    expanded = scale2.repeat_interleave(BLOCK, dim=0).repeat_interleave(BLOCK, dim=1)
    return flat.float() * expanded[:channels, :kernel]


def stats(name: str, x: torch.Tensor) -> None:
    y = x.detach().float().cpu()
    print(f"{name}: shape={tuple(x.shape)} dtype={x.dtype} norm={torch.linalg.vector_norm(y).item():.8f} mean={y.mean().item():.8f} std={y.std().item():.8f}")
    print(f"{name}: min={y.min().item():.8f} max={y.max().item():.8f}")


def compute_qkv(root: Path, token_id: int, device: str) -> torch.Tensor:
    x = load_embedding_row(root, token_id).to(device)
    norm_weight = load_tensor(root, LAYER_PREFIX + "input_layernorm.weight", device=device)
    qkv_weight = load_tensor(root, LAYER_PREFIX + "linear_attn.in_proj_qkv.weight", device="cpu")
    qkv_scale = load_tensor(root, LAYER_PREFIX + "linear_attn.in_proj_qkv.weight_scale_inv", device="cpu")
    h = rmsnorm(x, norm_weight)
    qkv_fp32 = dequantize_fp8_blockwise(qkv_weight, qkv_scale).to(device)
    y = F.linear(h.float(), qkv_fp32.float())
    if device == "cuda":
        torch.cuda.synchronize()
    del x, norm_weight, qkv_weight, qkv_scale, h, qkv_fp32
    return y


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.6 isolated Layer-0 operation probe")
    parser.add_argument("root", type=Path)
    parser.add_argument("--token-id", type=int, default=0)
    parser.add_argument("--op", choices=("norm", "qkv", "conv"), default="qkv")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    root = args.root.resolve()

    if args.op == "norm":
        x = load_embedding_row(root, args.token_id).to(args.device)
        weight = load_tensor(root, LAYER_PREFIX + "input_layernorm.weight", device=args.device)
        stats("embedding", x)
        start = perf_counter()
        y = rmsnorm(x, weight)
        if args.device == "cuda":
            torch.cuda.synchronize()
        print(f"op=norm time={(perf_counter()-start)*1000:.3f} ms")
        stats("norm output", y)
        return

    if args.op == "qkv":
        x = load_embedding_row(root, args.token_id).to(args.device)
        norm_weight = load_tensor(root, LAYER_PREFIX + "input_layernorm.weight", device=args.device)
        qkv_weight = load_tensor(root, LAYER_PREFIX + "linear_attn.in_proj_qkv.weight", device="cpu")
        qkv_scale = load_tensor(root, LAYER_PREFIX + "linear_attn.in_proj_qkv.weight_scale_inv", device="cpu")
        h = rmsnorm(x, norm_weight)
        stats("norm input", h)
        print(f"qkv weight shape={tuple(qkv_weight.shape)} dtype={qkv_weight.dtype}")
        print(f"qkv scale shape={tuple(qkv_scale.shape)} dtype={qkv_scale.dtype}")
        start = perf_counter()
        qkv_fp32 = dequantize_fp8_blockwise(qkv_weight, qkv_scale).to(args.device)
        if args.device == "cuda":
            torch.cuda.synchronize()
        dequant_ms = (perf_counter() - start) * 1000.0
        start = perf_counter()
        y = F.linear(h.float(), qkv_fp32.float())
        if args.device == "cuda":
            torch.cuda.synchronize()
        compute_ms = (perf_counter() - start) * 1000.0
        print(f"op=qkv dequant time={dequant_ms:.3f} ms")
        print(f"op=qkv compute time={compute_ms:.3f} ms")
        stats("qkv output", y)
        del x, norm_weight, qkv_weight, qkv_scale, h, qkv_fp32, y
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
        return

    qkv = compute_qkv(root, args.token_id, args.device)
    conv_weight = load_tensor(root, LAYER_PREFIX + "linear_attn.conv1d.weight", device="cpu")
    conv_scale = load_optional_tensor(root, LAYER_PREFIX + "linear_attn.conv1d.weight_scale_inv", device="cpu")
    print(f"qkv input shape={tuple(qkv.shape)}")
    print(f"conv weight shape={tuple(conv_weight.shape)} dtype={conv_weight.dtype}")
    print(f"conv scale shape={tuple(conv_scale.shape) if conv_scale is not None else None}")

    start = perf_counter()
    conv_fp32 = dequantize_conv1d_weight(conv_weight, conv_scale).to(args.device)
    if args.device == "cuda":
        torch.cuda.synchronize()
    dequant_ms = (perf_counter() - start) * 1000.0

    qkv_current = qkv.reshape(1, -1)
    # Weight is [channel, 1, kernel]; for token 0 only the final causal tap
    # multiplies the current token because three history positions are zero.
    tap = conv_fp32[:, 0, -1].reshape(1, -1)
    start = perf_counter()
    y_linear = qkv_current * tap
    y = F.silu(y_linear)
    if args.device == "cuda":
        torch.cuda.synchronize()
    compute_ms = (perf_counter() - start) * 1000.0

    print("op=conv causal history=zeros")
    print(f"op=conv dequant time={dequant_ms:.3f} ms")
    print(f"op=conv compute time={compute_ms:.3f} ms")
    stats("conv pre-activation", y_linear.reshape(-1))
    stats("conv output", y.reshape(-1))

    del qkv, conv_weight, conv_scale, conv_fp32, qkv_current, tap, y_linear, y
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
