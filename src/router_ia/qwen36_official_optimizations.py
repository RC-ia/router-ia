from __future__ import annotations

"""Hot-path optimizations for the official Qwen3.6 stateful runner.

This module is intentionally a thin compatibility layer: it reuses the
existing ``RoutedExpertCache`` implementation without changing model math.
"""

import gc
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from . import qwen36_chat_batch as chat
from . import qwen36_cached_loop as cached
from .qwen36_expert_cache import RoutedExpertCache

EXPERT_LOAD_WORKERS = max(1, int(os.getenv("QWEN36_EXPERT_LOAD_WORKERS", "8")))
EXPERT_VRAM_GB = max(float(os.getenv("QWEN36_EXPERT_VRAM_GB", "1.0")), 0.0)
EXPERT_VRAM_BUDGET_BYTES = int(EXPERT_VRAM_GB * 1024**3)

_EXPERT_CACHES: dict[Path, RoutedExpertCache] = {}
_ORIGINAL_TRIPLET = chat._expert_projection_triplet
_ORIGINAL_WARM = chat._warm_expert_raw_cache
_ORIGINAL_RUN_FORWARD_TOKEN = chat.run_forward_token
_ORIGINAL_CACHE_STATS = chat.cache_stats
_ORIGINAL_PRINT_CACHE = chat.print_cache


def _configure_budget() -> None:
    """Reserve a dedicated expert window and give the rest to the stream."""
    if EXPERT_VRAM_BUDGET_BYTES <= 0:
        return
    if cached.VRAM_CACHE_BUDGET_BYTES <= cached.RESIDENT_VRAM_BUDGET_BYTES:
        return

    cached.STREAM_BUDGET_BYTES = max(
        cached.VRAM_CACHE_BUDGET_BYTES
        - cached.RESIDENT_VRAM_BUDGET_BYTES
        - EXPERT_VRAM_BUDGET_BYTES,
        0,
    )
    cached.STREAM_GB = cached.STREAM_BUDGET_BYTES / 1024**3


def _expert_cache(root: Path) -> RoutedExpertCache:
    key = root.resolve()
    cache = _EXPERT_CACHES.get(key)
    if cache is None:
        cache = RoutedExpertCache(EXPERT_VRAM_BUDGET_BYTES)
        _EXPERT_CACHES[key] = cache
    return cache


def _layer_from_prefix(layer_prefix: str) -> int | None:
    marker = ".layers."
    if marker not in layer_prefix:
        return None
    try:
        return int(layer_prefix.split(marker, 1)[1].split(".", 1)[0])
    except (ValueError, IndexError):
        return None


def _cached_triplet(
    root: Path,
    layer_prefix: str,
    expert_id: int,
    device: str,
):
    if device != "cuda":
        return _ORIGINAL_TRIPLET(root, layer_prefix, expert_id, device)
    layer = _layer_from_prefix(layer_prefix)
    if layer is None:
        return _ORIGINAL_TRIPLET(root, layer_prefix, expert_id, device)
    return _expert_cache(root).get_or_load(
        cached._store(root), layer, int(expert_id), layer_prefix
    )


def _warm_raw(root: Path, layer_prefix: str, expert_ids: list[int]) -> None:
    if not expert_ids:
        return
    if not torch.cuda.is_available():
        return

    store = cached._store(root)
    expert_cache = _expert_cache(root)
    workers = min(EXPERT_LOAD_WORKERS, len(expert_ids))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                expert_cache.prefetch_expert_raw,
                store,
                layer_prefix,
                int(expert_id),
            )
            for expert_id in dict.fromkeys(int(v) for v in expert_ids)
        ]
        for future in futures:
            future.result()


def _cache_stats(root: Path) -> dict[str, int | float]:
    stats = dict(_ORIGINAL_CACHE_STATS(root))
    if not stats:
        return stats

    expert = _EXPERT_CACHES.get(root.resolve())
    if expert is None:
        return stats

    snapshot = expert.snapshot()
    stats["vram_expert_bytes"] = int(snapshot["bytes"])
    stats["vram_expert_budget"] = int(expert.budget_bytes)
    stats["vram_expert_hit_rate"] = float(snapshot["hit_rate"])
    stats["vram_expert_evictions"] = int(snapshot["evictions"])
    stats["vram_expert_items"] = int(snapshot["warm_items"])
    stats["vram_expert_fp8_hits"] = int(snapshot["fp8_hits"])
    stats["vram_expert_q4_hits"] = int(snapshot["q4_hits"])

    # The dedicated expert pool is outside the generic cache object, so expose
    # the true combined VRAM footprint in the existing diagnostics.
    stats["vram_bytes"] = int(stats.get("vram_bytes", 0)) + int(snapshot["bytes"])
    stats["vram_expert_share"] = (
        int(snapshot["bytes"]) / max(int(stats["vram_bytes"]), 1) * 100.0
    )
    return stats


def _print_cache(root: Path, label: str) -> None:
    # Keep the original formatting but report the dedicated expert pool too.
    stats = _cache_stats(root)
    if not stats:
        print(f"  cache {label}: unavailable")
        return
    print(
        f"  cache {label}: "
        f"ram={stats['ram_bytes'] / 1024**2:.1f}/{cached.CACHE_BUDGET_BYTES / 1024**2:.1f} MiB | "
        f"vram={stats['vram_bytes'] / 1024**2:.1f}/{cached.VRAM_CACHE_BUDGET_BYTES / 1024**2:.1f} MiB | "
        f"resident={stats['vram_resident_bytes'] / 1024**2:.1f}/{stats['vram_resident_budget'] / 1024**2:.1f} MiB | "
        f"experts={stats['vram_expert_bytes'] / 1024**2:.1f}/{stats['vram_expert_budget'] / 1024**2:.1f} MiB | "
        f"stream={stats['vram_stream_bytes'] / 1024**2:.1f}/{stats['vram_stream_budget'] / 1024**2:.1f} MiB | "
        f"hit_rate={stats['hit_rate']:.2f}% | "
        f"ram_hit={stats['ram_hit_rate']:.2f}% | "
        f"vram_hit={stats['vram_hit_rate']:.2f}% | "
        f"expert_vram_hit={stats['vram_expert_hit_rate']:.2f}% | "
        f"stream_hit={stats['vram_stream_hit_rate']:.2f}% | "
        f"expert_evictions={stats['vram_expert_evictions']}"
    )


def _without_allocator_flush(fn):
    def wrapped(*args, **kwargs):
        old_collect = gc.collect
        old_empty_cache = torch.cuda.empty_cache
        gc.collect = lambda: 0
        torch.cuda.empty_cache = lambda: None
        try:
            return fn(*args, **kwargs)
        finally:
            gc.collect = old_collect
            torch.cuda.empty_cache = old_empty_cache

    wrapped.__name__ = getattr(fn, "__name__", "wrapped")
    wrapped.__doc__ = getattr(fn, "__doc__", None)
    return wrapped


# Apply once when imported by the package initializer.
_configure_budget()
chat._EXPERT_CACHES = _EXPERT_CACHES
chat._expert_projection_triplet = _cached_triplet
chat._warm_expert_raw_cache = _warm_raw
chat.cache_stats = _cache_stats
chat.print_cache = _print_cache
chat.run_forward_token = _without_allocator_flush(_ORIGINAL_RUN_FORWARD_TOKEN)
