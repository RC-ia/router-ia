from __future__ import annotations

"""GPU FP16 materialization cache for repeatedly used routed experts.

The compressed FP8/Q4 tiers remain the backing store. This tiny cache only keeps
already-dequantized FP16 triplets in the dedicated expert VRAM budget, avoiding
repeated FP8->FP16 work for experts that are used again during the same
continuous generation epoch.
"""

from collections import OrderedDict
from pathlib import Path
from threading import Lock

import torch

from . import qwen36_expert_cache as expert_cache
from . import qwen36_official_optimizations as official

FP16_CACHE_BYTES = int(max(float(__import__("os").getenv("QWEN36_EXPERT_FP16_GB", "0.75")), 0.0) * 1024**3)

_CACHES: dict[Path, "FP16Cache"] = {}
_LOCK = Lock()


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
        cache = _CACHES.get(key)
        if cache is None:
            cache = FP16Cache()
            _CACHES[key] = cache
        return cache


def _root_for_expert_cache(cache) -> Path | None:
    for root, value in official._EXPERT_CACHES.items():
        if value is cache:
            return root
    return None


def _get_or_load_batch(self, store, layer: int, expert_ids: list[int], layer_prefix: str):
    root = _root_for_expert_cache(self)
    if root is None:
        # Preserve the original implementation when the cache is not owned by
        # the official runtime.
        return _ORIGINAL_GET_OR_LOAD_BATCH(self, store, layer, expert_ids, layer_prefix)

    ids = [int(x) for x in expert_ids]
    out: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    missing: list[int] = []
    cache = _cache(root)

    for expert_id in dict.fromkeys(ids):
        hit = cache.get((int(layer), int(expert_id)))
        if hit is not None:
            out[int(expert_id)] = hit
        else:
            missing.append(int(expert_id))

    if missing:
        decoded = _ORIGINAL_GET_OR_LOAD_BATCH(self, store, int(layer), missing, layer_prefix)
        for expert_id, triplet in zip(missing, decoded):
            # Store the already-dequantized FP16 tensors. Cloning is avoided so
            # the cache owns exactly the tensors returned by the decoder.
            fp16 = tuple(t.to(device="cuda", dtype=torch.float16) for t in triplet)
            out[int(expert_id)] = fp16
            cache.put((int(layer), int(expert_id)), fp16)

    return [out[int(expert_id)] for expert_id in ids]


_ORIGINAL_GET_OR_LOAD_BATCH = expert_cache.RoutedExpertCache.get_or_load_batch
expert_cache.RoutedExpertCache.get_or_load_batch = _get_or_load_batch


def reset(root: Path) -> None:
    _cache(root).clear()


def stats(root: Path):
    return _cache(root).snapshot()


def print_stats(root: Path) -> None:
    snap = stats(root)
    print(
        f"  expert FP16 cache: {snap['bytes'] / 1024**2:.1f}/"
        f"{snap['budget'] / 1024**2:.1f} MiB | items={snap['items']} | "
        f"hits={snap['hits']} | misses={snap['misses']} | "
        f"hit_rate={snap['hit_rate']:.1f}% | evictions={snap['evictions']}"
    )


print("expert_fp16_cache=enabled|gpu-only|budget=0.75GiB|backing=FP8/Q4")
