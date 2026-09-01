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


def stats(name: str, x: torch.Tensor) -> None:
    y = x.detach().float().cpu()
    print(f"{name}: shape={tuple(x.shape)} dtype={x.dtype} norm={torch.linalg.vector_norm(y).item():.8f} mean={y.mean().item():.8f} std={y.std().item():.8f}")
    print(f"{name}: min={y.min().item():.8f} max={y.max().item():.8f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.6 isolated Layer-0 operation probe")
    parser.add_argument("root", type=Path)
    parser.add_argument("--token-id", type=int, default=0)
    parser.add_argument("--op", choices=("norm", "qkv"), default="qkv")
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

    x = load_embedding_row(root, args.token_id).to(args.device)
    norm_weight = load_tensor(root, LAYER_PREFIX + "input_layernorm.weight", device=args.device)
    qkv_weight = load_tensor(root, LAYER_PREFIX + "linear_attn.in_proj_qkv.weight", device=args.device)
    h = rmsnorm(x, norm_weight)
    stats("norm input", h)
    print(f"qkv weight shape={tuple(qkv_weight.shape)} dtype={qkv_weight.dtype}")
    start = perf_counter()
    y = F.linear(h.float(), qkv_weight.float())
    if args.device == "cuda":
        torch.cuda.synchronize()
    print(f"op=qkv time={(perf_counter()-start)*1000:.3f} ms")
    stats("qkv output", y)
    del x, norm_weight, qkv_weight, h, y
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
