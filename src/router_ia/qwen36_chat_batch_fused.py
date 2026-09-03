from __future__ import annotations

"""Qwen3.6 chat runner with persistent two-tier expert GPU cache."""

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
            "expert_cache_total_slots": int(expert["total_slots"]),
            "expert_cache_hits": int(expert["hits"]),
            "expert_cache_misses": int(expert["misses"]),
            "expert_cache_hit_rate": float(expert["hit_rate"]),
            "expert_cache_loads": int(expert["loads"]),
            "expert_cache_evictions": int(expert["evictions"]),
            "expert_cache_hot_items": int(expert["hot_items"]),
            "expert_cache_fp8_items": int(expert["warm_items"]),
            "expert_cache_hot_hits": int(expert["hot_hits"]),
            "expert_cache_fp8_hits": int(expert["fp8_hits"]),
            "expert_cache_fp16_to_fp8": int(expert["fp16_to_fp8"]),
            "expert_cache_fp8_drops": int(expert["q4_drops"]),
            "expert_cache_stream_prefetch_hits": int(expert["stream_prefetch_hits"]),
            "expert_cache_stream_prefetch_misses": int(expert["stream_prefetch_misses"]),
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
    print(
        f"    tiers: FP16={expert['hot_items']} | FP8={expert['warm_items']} | "
        f"FP8_hits={expert['fp8_hits']} | "
        f"compressions FP16>FP8={expert['fp16_to_fp8']} | "
        f"FP8_drops={expert['q4_drops']} | "
        f"stream_prefetch hits={expert['stream_prefetch_hits']} "
        f"misses={expert['stream_prefetch_misses']}"
    )


chat._expert_projection_triplet = _cached_expert_projection_triplet
chat.cache_stats = _cache_stats_with_experts
chat.print_cache = _print_cache_with_experts


def main() -> None:
    cache = _expert_cache(Path("."))
    print("expert_cache=complete-layer-expert")
    print("expert_cache_key=(layer,expert)")
    print("expert_cache_policy=per-layer-tiered-2fp16-4fp8")
    print("expert_cache_budget=full-stream-vram-budget")
    print("expert_cache_entry=FP16-hot|FP8-warm")
    print("expert_cache_eviction=compress-to-fp8-then-drop")
    print("expert_cache_fp8_promotion=transient-only")
    print("expert_cache_prefetch=raw-fp8-in-stream")
    print("expert_cache_compute=temporary-fp16")
    print(f"expert_cache_total_slots={cache.total_slots}")
    print(f"expert_cache_slots_per_layer={cache.slots_per_layer}")
    print(f"expert_cache_hot_slots_per_layer={cache.hot_slots}")
    print(f"expert_cache_fp8_slots_per_layer={cache.fp8_slots}")
    print(f"expert_cache_q4_slots_per_layer=0")
    chat.main()


if __name__ == "__main__":
    main()
