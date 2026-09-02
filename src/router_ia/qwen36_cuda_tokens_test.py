from __future__ import annotations

"""Benchmark GPU weight-cache reuse across several token passes.

This intentionally runs each token as an independent position-0 pass. The
purpose is measuring weight-cache reuse, not sequence/KV-cache correctness.
"""

import argparse
import gc
from pathlib import Path
from time import perf_counter

import torch

from . import qwen36_cuda_loop as runner
from .qwen36_op_probe import load_embedding_row

DEFAULT_TOKENS = "0,1,2"


def run_one(root: Path, token_id: int, start_layer: int, end_layer: int, top_k: int, cache: runner.GPUWeightCache) -> tuple[float, torch.Tensor]:
    device = "cuda"
    x = load_embedding_row(root, token_id).reshape(1, runner.HIDDEN).to(device).float()
    start = perf_counter()

    for layer in range(start_layer, end_layer + 1):
        kind = runner.attention_type(root, layer)
        residual = (
            runner.linear_attention_step(root, layer, x, device)
            if kind == "linear_attention"
            else runner.full_attention_step(root, layer, x, device)
        )
        x, *_ = runner.moe_step(root, layer, residual, top_k, device)
        del residual

    torch.cuda.synchronize()
    elapsed_ms = (perf_counter() - start) * 1000.0
    return elapsed_ms, x


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.6 CUDA multi-token cache test")
    parser.add_argument("root", type=Path)
    parser.add_argument("--tokens", default=DEFAULT_TOKENS, help="Comma-separated token IDs")
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--end-layer", type=int, default=runner.DEFAULT_LAYERS - 1)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--cache-mib", type=float, default=512.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    if args.start_layer < 0 or args.end_layer < args.start_layer or args.end_layer >= runner.DEFAULT_LAYERS:
        raise SystemExit(f"layer range must be inside 0..{runner.DEFAULT_LAYERS - 1}")
    if args.cache_mib < 0:
        raise SystemExit("cache-mib must be non-negative")

    try:
        tokens = [int(part.strip()) for part in args.tokens.split(",") if part.strip()]
    except ValueError as exc:
        raise SystemExit(f"invalid --tokens: {args.tokens}") from exc
    if not tokens:
        raise SystemExit("--tokens must contain at least one token id")

    root = args.root.resolve()
    cache = runner.GPUWeightCache(args.cache_mib)
    runner.set_gpu_cache(cache)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch CUDA: {torch.version.cuda}")
    print(f"tokens: {tokens}")
    print(f"layers: {args.start_layer}..{args.end_layer}")
    print(f"GPU cache budget: {args.cache_mib:.1f} MiB")
    print("NOTE: tokens are independent position-0 passes; this is a cache benchmark.")

    all_start = perf_counter()
    total_compute_ms = 0.0

    try:
        for index, token_id in enumerate(tokens, start=1):
            cache_before = cache.stats()
            elapsed_ms, output = run_one(
                root, token_id, args.start_layer, args.end_layer, args.top_k, cache
            )
            total_compute_ms += elapsed_ms
            used, entries, hits, misses = cache.stats()

            print(f"token {index}/{len(tokens)} id={token_id}:")
            print(f"  time: {elapsed_ms:.3f} ms")
            print(f"  output norm: {torch.linalg.vector_norm(output).item():.8f}")
            print(f"  output mean: {output.mean().item():.8f}")
            print(f"  cache delta hits: {hits - cache_before[2]}")
            print(f"  cache delta misses: {misses - cache_before[3]}")
            print(f"  cache size: {used / 1024**2:.1f} MiB / {args.cache_mib:.1f} MiB ({entries} tensors)")
            print(f"  cache total hits: {hits}")
            print(f"  cache total misses: {misses}")
            print(f"  cache evictions: {cache.evictions}")
            print(f"  VRAM allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MiB")
            print(f"  VRAM reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MiB")

            del output
            gc.collect()

        torch.cuda.synchronize()
        wall_ms = (perf_counter() - all_start) * 1000.0
        print(f"total compute time: {total_compute_ms:.3f} ms")
        print(f"total wall time: {wall_ms:.3f} ms")
        print(f"average per token: {total_compute_ms / len(tokens):.3f} ms")
        print(f"final cache size: {cache.used_bytes / 1024**2:.1f} MiB")
        print(f"final cache entries: {len(cache.items)}")
        print(f"final cache hits: {cache.hits}")
        print(f"final cache misses: {cache.misses}")
        print(f"final cache evictions: {cache.evictions}")
    finally:
        cache.clear()
        runner.set_gpu_cache(None)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
