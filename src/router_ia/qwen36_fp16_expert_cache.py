from __future__ import annotations

"""Optional GPU FP16 materialization cache for routed experts.

Disabled by default because the normal 1 GiB expert pool is intentionally
kept intact. Enable explicitly with ``QWEN36_EXPERT_FP16_GB`` when testing
FP16 materialization as an alternative cache layout.
"""

import os
from collections import OrderedDict
from pathlib import Path
from threading import Lock

import torch

from . import qwen36_expert_cache as expert_cache
from . import qwen36_official_optimizations as official
from . import qwen36_chat_batch as chat


def _configure_partition() -> tuple[int, int]:
    total = int(max(official.EXPERT_VRAM_BUDGET_BYTES, 0))
    requested = os.getenv("QWEN36_EXPERT_FP16_GB")
    if requested is None:
        # Keep the production/default compressed expert budget untouched.
        return 0, total
    try:
        fp16 = int(max(float(requested), 0.0) * 1024**3)
    except ValueError:
        fp16 = 0
    fp16 = min(fp16, total)
    compressed = max(total - fp16, 0)
    official.EXPERT_VRAM_BUDGET_BYTES = compressed
    return fp16, compressed


FP16_CACHE_BYTES, FP8_CACHE_BYTES = _configure_partition()

_CACHES: dict[Path, "FP16Cache"] = {}
_LOCK = Lock()
_ORIGINAL_GENERATE_RESPONSE = chat.generate_response
_ORIGINAL_CACHE_STATS = chat.cache_stats
_ORIGINAL_PRINT_CACHE = chat.print_cache
_ORIGINAL_GET_OR_LOAD_BATCH = expert_cache.RoutedExpertCache.get_or_load_batch


class FP16Cache:
    def __init__(self) -> None:
        self.items: OrderedDict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = OrderedDict()
        self.sizes: dict[tuple[int, int], int] = {}
        self.bytes_used = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.lock = Lock()

    @staticmethod
    def _size(entry) -> int:
        return sum(int(t.numel()) * int(t.element_size()) for t in entry)

    def get(self, key):
        with self.lock:
            entry = self.items.get(key)
            if entry is None:
                self.misses += 1
                return None
            self.items.move_to_end(key)
            self.hits += 1
            return entry

    def put(self, key, entry) -> None:
        size = self._size(entry)
        if size > FP16_CACHE_BYTES or FP16_CACHE_BYTES <= 0:
            return
        with self.lock:
            old = self.items.pop(key, None)
            if old is not None:
                self.bytes_used -= self.sizes.pop(key, 0)
            while self.bytes_used + size > FP16_CACHE_BYTES and self.items:
                victim, _ = self.items.popitem(last=False)
                self.bytes_used -= self.sizes.pop(victim, 0)
                self.evictions += 1
            self.items[key] = entry
            self.sizes[key] = size
            self.bytes_used += size

    def clear(self) -> None:
        with self.lock:
            self.items.clear()
            self.sizes.clear()
            self.bytes_used = 0

    def snapshot(self):
        with self.lock:
            total = self.hits + self.misses
            return {
                "items": len(self.items),
                "bytes": self.bytes_used,
                "budget": FP16_CACHE_BYTES,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / total * 100.0 if total else 0.0,
                "evictions": self.evictions,
            }


def _cache(root: Path) -> FP16Cache:
    key = root.resolve()
    with _LOCK:
        value = _CACHES.get(key)
        if value is None:
            value = FP16Cache()
            _CACHES[key] = value
        return value


def _root_for_expert_cache(cache) -> Path | None:
    for root, value in official._EXPERT_CACHES.items():
        if value is cache:
            return root
    return None


def _get_or_load_batch(self, store, layer: int, expert_ids: list[int], layer_prefix: str):
    if FP16_CACHE_BYTES <= 0:
        return _ORIGINAL_GET_OR_LOAD_BATCH(self, store, layer, expert_ids, layer_prefix)

    root = _root_for_expert_cache(self)
    if root is None:
        return _ORIGINAL_GET_OR_LOAD_BATCH(self, store, layer, expert_ids, layer_prefix)

    ids = [int(x) for x in expert_ids]
    output: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    missing: list[int] = []
    materialized = _cache(root)

    for expert_id in dict.fromkeys(ids):
        hit = materialized.get((int(layer), int(expert_id)))
        if hit is not None:
            output[int(expert_id)] = hit
        else:
            missing.append(int(expert_id))

    if missing:
        decoded = _ORIGINAL_GET_OR_LOAD_BATCH(self, store, layer, missing, layer_prefix)
        for expert_id, triplet in zip(missing, decoded):
            fp16 = tuple(t.to(device="cuda", dtype=torch.float16) for t in triplet)
            output[int(expert_id)] = fp16
            materialized.put((int(layer), int(expert_id)), fp16)

    return [output[int(expert_id)] for expert_id in ids]


expert_cache.RoutedExpertCache.get_or_load_batch = _get_or_load_batch


def reset(root: Path) -> None:
    _cache(root).clear()


def stats(root: Path):
    return _cache(root).snapshot()


def _generate_response(*args, **kwargs):
    root = args[0] if args else kwargs.get("root")
    if isinstance(root, Path) and torch.cuda.is_available():
        reset(root)
    return _ORIGINAL_GENERATE_RESPONSE(*args, **kwargs)


def _cache_stats(root: Path) -> dict[str, int | float]:
    result = dict(_ORIGINAL_CACHE_STATS(root))
    snap = stats(root)
    result.update({
        "expert_fp16_cache_items": int(snap["items"]),
        "expert_fp16_cache_bytes": int(snap["bytes"]),
        "expert_fp16_cache_budget": int(snap["budget"]),
        "expert_fp16_cache_hits": int(snap["hits"]),
        "expert_fp16_cache_misses": int(snap["misses"]),
        "expert_fp16_cache_hit_rate": float(snap["hit_rate"]),
        "expert_fp16_cache_evictions": int(snap["evictions"]),
        "expert_fp8_partition_budget": int(FP8_CACHE_BYTES),
    })
    return result


def _print_cache(root: Path, label: str) -> None:
    _ORIGINAL_PRINT_CACHE(root, label)
    snap = stats(root)
    print(
        f"  expert FP16 cache: {snap['bytes'] / 1024**2:.1f}/"
        f"{snap['budget'] / 1024**2:.1f} MiB | items={snap['items']} | "
        f"hits={snap['hits']} | misses={snap['misses']} | "
        f"hit_rate={snap['hit_rate']:.1f}% | evictions={snap['evictions']} | "
        f"FP8_budget={FP8_CACHE_BYTES / 1024**2:.1f} MiB"
    )


chat.generate_response = _generate_response
chat.cache_stats = _cache_stats
chat.print_cache = _print_cache

if FP16_CACHE_BYTES > 0:
    print(
        f"expert_fp16_cache=enabled|gpu-only|partition={FP16_CACHE_BYTES / 1024**3:.2f}GiB-FP16+"
        f"{FP8_CACHE_BYTES / 1024**3:.2f}GiB-FP8|same-expert-budget"
    )
else:
    print("expert_fp16_cache=disabled|default|expert-budget-unchanged")
