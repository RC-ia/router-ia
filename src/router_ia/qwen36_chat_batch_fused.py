from __future__ import annotations

"""Qwen3.6 chat runner with persistent per-layer expert GPU cache.

Routed experts are cached as complete (layer, expert) triplets. Each entry
contains gate/up/down FP16 matrices and is evicted atomically with LRU. The
cache uses the full VRAM stream budget, so it replaces projection-level
streaming for routed experts without allocating a second VRAM budget.
"""

from pathlib import Path

import torch

from . import qwen36_cached_loop as cached
from . import qwen36_chat_batch as chat
from .qwen36_expert_cache import RoutedExpertCache


_EXPERT_CACHES: dict[Path, RoutedExpertCache] = {}
_ORIGINAL_EXPERT_TRIPLET = chat._expert_projection_triplet
_ORIGINAL_CACHE_STATS = chat.cache_stats
_ORIGINAL_PRINT_CACHE = chat.print_cache


def _expert_cache(root: Path) -> RoutedExpertCache:
    key = root.resolve()
    cache = _EXPERT_CACHES.get(key)
    if cache is None:
        cache = RoutedExpertCache(cached.STREAM_BUDGET_BYTES)
        _EXPERT_CACHES[key] = cache
    return cache


def _cached_expert_projection_triplet(
    root: Path,
    layer_prefix: str,
    expert_id: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if device != "cuda":
        return _ORIGINAL_EXPERT_TRIPLET(root, layer_prefix, expert_id, device)

    layer_marker = ".layers."
    if layer_marker not in layer_prefix:
        return _ORIGINAL_EXPERT_TRIPLET(root, layer_prefix, expert_id, device)

    try:
        layer = int(layer_prefix.split(layer_marker, 1)[1].split(".", 1)[0])
    except (ValueError, IndexError):
        return _ORIGINAL_EXPERT_TRIPLET(root, layer_prefix, expert_id, device)

    return _expert_cache(root).get_or_load(
        cached._store(root), layer, expert_id, layer_prefix
    )


def _cache_stats_with_experts(root: Path) -> dict[str, int | float]:
    stats = dict(_ORIGINAL_CACHE_STATS(root))
    cache = _EXPERT_CACHES.get(root.resolve())
    if cache is None:
        return stats
    expert = cache.snapshot()
    stats.update(
        {
            "expert_cache_items": int(expert["items"]),
            "expert_cache_bytes": int(expert["bytes"]),
            "expert_cache_budget": int(expert["budget_bytes"]),
            "expert_cache_hits": int(expert["hits"]),
            "expert_cache_misses": int(expert["misses"]),
            "expert_cache_hit_rate": float(expert["hit_rate"]),
            "expert_cache_loads": int(expert["loads"]),
            "expert_cache_evictions": int(expert["evictions"]),
        }
    )
    return stats


def _print_cache_with_experts(root: Path, label: str) -> None:
    _ORIGINAL_PRINT_CACHE(root, label)
    cache = _EXPERT_CACHES.get(root.resolve())
    if cache is None:
        return
    expert = cache.snapshot()
    print(
        f"  expert_cache: entries={expert['items']} | "
        f"vram={expert['bytes'] / 1024**2:.1f}/{expert['budget_bytes'] / 1024**2:.1f} MiB | "
        f"hit_rate={expert['hit_rate']:.2f}% | "
        f"hits={expert['hits']} | misses={expert['misses']} | "
        f"loads={expert['loads']} | evictions={expert['evictions']}"
    )


chat._expert_projection_triplet = _cached_expert_projection_triplet
chat.cache_stats = _cache_stats_with_experts
chat.print_cache = _print_cache_with_experts


def main() -> None:
    print("expert_cache=complete-layer-expert")
    print("expert_cache_key=(layer,expert)")
    print("expert_cache_policy=atomic-lru")
    print("expert_cache_budget=full-stream-vram-budget")
    print("expert_cache_entry=gate+up+down-fp16")
    print("expert_cache_eviction=whole-expert")
    chat.main()


if __name__ == "__main__":
    main()
