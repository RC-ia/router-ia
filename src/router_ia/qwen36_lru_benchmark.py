from __future__ import annotations

"""Repeated single-token benchmark for the Qwen3.6 RAM LRU cache.

Runs the same token sequence for multiple generations while keeping the LRU
alive between tokens/generations. This intentionally does not implement a KV
cache: each token is an independent single-token pass, so the benchmark is
for measuring weight-cache reuse, not autoregressive quality/perplexity.
"""

import argparse
import gc
from pathlib import Path
from time import perf_counter

import torch

from . import qwen36_cached_loop as cached
from . import qwen36_40layer_loop as base


def run_token(root: Path, token_id: int, start_layer: int, end_layer: int, top_k: int, device: str) -> float:
    x = base.load_embedding_row(root, token_id).reshape(1, base.HIDDEN).to(device).float()
    start = perf_counter()

    for layer in range(start_layer, end_layer + 1):
        kind = base.attention_type(root, layer)
        if kind == "linear_attention":
            residual = base.linear_attention_step(root, layer, x, device)
        else:
            residual = base.full_attention_step(root, layer, x, device)
        x, expert_ids, weights, shared_gate, moe_input_norm = base.moe_step(
            root, layer, residual, top_k, device
        )
        del residual, expert_ids, weights, shared_gate, moe_input_norm
        gc.collect()

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = perf_counter() - start

    print(
        f"    token={token_id:>6} | "
        f"time={elapsed:>9.3f} s | "
        f"output_norm={torch.linalg.vector_norm(x).item():.8f}"
    )

    del x
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return elapsed


def print_cache_stats(label: str, root: Path) -> None:
    store = cached._stores.get(root.resolve())
    if store is None:
        print(f"  cache {label}: reader not initialized")
        return

    print(
        f"  cache {label}: "
        f"items={len(store.tensor_cache)} | "
        f"ram={store.cache_bytes / (1024 ** 2):.1f}/"
        f"{store.max_cache_bytes / (1024 ** 2):.1f} MiB | "
        f"hits={store.cache_hits} | misses={store.cache_misses} | "
        f"hit_rate={store.cache_hit_rate * 100.0:.2f}% | "
        f"evictions={store.cache_evictions}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeated Qwen3.6 LRU weight-cache benchmark")
    parser.add_argument("root", type=Path)
    parser.add_argument("--tokens", type=str, default="0,1,2", help="Comma-separated token IDs")
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--end-layer", type=int, default=base.DEFAULT_LAYERS - 1)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    if args.generations < 1:
        raise SystemExit("--generations must be >= 1")
    if not 0 <= args.start_layer <= args.end_layer < base.DEFAULT_LAYERS:
        raise SystemExit(f"layer range must be inside 0..{base.DEFAULT_LAYERS - 1}")

    try:
        tokens = [int(value.strip()) for value in args.tokens.split(",") if value.strip()]
    except ValueError as exc:
        raise SystemExit(f"invalid --tokens: {args.tokens!r}") from exc
    if not tokens:
        raise SystemExit("--tokens must contain at least one token id")

    root = args.root.resolve()
    total_start = perf_counter()
    all_times: list[float] = []

    print("op=lru-benchmark")
    print(f"tokens: {tokens}")
    print(f"generations: {args.generations}")
    print(f"layers: {args.start_layer}..{args.end_layer}")
    print(f"device: {args.device}")
    print("note: tokens are independent single-token passes; no KV cache")

    for generation in range(1, args.generations + 1):
        generation_start = perf_counter()
        print(f"\ngeneration {generation}/{args.generations}")
        print_cache_stats("before", root)

        for token_id in tokens:
            all_times.append(
                run_token(
                    root,
                    token_id,
                    args.start_layer,
                    args.end_layer,
                    args.top_k,
                    args.device,
                )
            )

        generation_time = perf_counter() - generation_start
        print(f"  generation time: {generation_time:.3f} s")
        print_cache_stats("after", root)

    total_time = perf_counter() - total_start
    print("\nsummary")
    print(f"  total time: {total_time:.3f} s")
    print(f"  tokens processed: {len(all_times)}")
    print(f"  avg token time: {sum(all_times) / len(all_times):.3f} s")
    if len(all_times) >= 2:
        print(f"  first token time: {all_times[0]:.3f} s")
        print(f"  last token time: {all_times[-1]:.3f} s")
        print(f"  change first->last: {(all_times[-1] - all_times[0]):+.3f} s")
    print_cache_stats("final", root)


if __name__ == "__main__":
    main()
