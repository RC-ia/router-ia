from __future__ import annotations

"""Qwen3.6 loop with persistent shard handles, bounded LRU tensor cache, and FP8 dequantization.

This wrapper keeps the reference math in qwen36_40layer_loop.py unchanged while
replacing only the tensor-loading backend and FP8 dequantizer.

The reader now keeps the Safetensors shards open for the whole run and also
keeps recently used CPU tensors in a bounded LRU cache. This is intended to
smooth out storage variability by reusing hot tensors without allowing RAM use
to grow without bound.

Environment variables:
    QWEN36_CACHE_GB: RAM budget for raw tensor cache, default 2.5 GiB.
    QWEN36_CACHE_LOG_INTERVAL: print cache progress every N tensor loads,
        default 0 (disabled).
"""

import atexit
import json
import os
from collections import OrderedDict
from contextlib import ExitStack
from pathlib import Path
from threading import Lock

from safetensors import safe_open

from . import qwen36_40layer_loop as base
from .qwen36_dequant import dequantize_fp8_blockwise


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else 0


CACHE_GB = _env_float("QWEN36_CACHE_GB", 2.5)
CACHE_BUDGET_BYTES = int(CACHE_GB * 1024 * 1024 * 1024)
CACHE_LOG_INTERVAL = _env_int("QWEN36_CACHE_LOG_INTERVAL", 0)


class _TensorLRU:
    """Bounded LRU cache for CPU tensors with explicit byte accounting."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self.items: OrderedDict[str, object] = OrderedDict()
        self.item_bytes: dict[str, int] = {}
        self.bytes_used = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.skipped_oversize = 0
        self.loads = 0
        self.lock = Lock()

    @staticmethod
    def _tensor_bytes(tensor) -> int:
        return int(tensor.numel()) * int(tensor.element_size())

    def get(self, name: str):
        with self.lock:
            tensor = self.items.get(name)
            if tensor is None:
                self.misses += 1
                return None
            self.hits += 1
            self.items.move_to_end(name)
            return tensor

    def put(self, name: str, tensor) -> None:
        size = self._tensor_bytes(tensor)
        with self.lock:
            self.loads += 1

            previous = self.items.pop(name, None)
            if previous is not None:
                self.bytes_used -= self.item_bytes.pop(name, 0)

            if size > self.max_bytes:
                self.skipped_oversize += 1
                return

            while self.items and self.bytes_used + size > self.max_bytes:
                old_name, _ = self.items.popitem(last=False)
                old_size = self.item_bytes.pop(old_name, 0)
                self.bytes_used -= old_size
                self.evictions += 1

            self.items[name] = tensor
            self.item_bytes[name] = size
            self.bytes_used += size

    def clear(self) -> None:
        with self.lock:
            self.items.clear()
            self.item_bytes.clear()
            self.bytes_used = 0

    def snapshot(self) -> dict[str, int | float]:
        with self.lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total * 100.0) if total else 0.0
            return {
                "items": len(self.items),
                "bytes": self.bytes_used,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "skipped_oversize": self.skipped_oversize,
                "loads": self.loads,
                "hit_rate": hit_rate,
            }


class _ShardStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.stack = ExitStack()
        self.weight_map: dict[str, str] = {}
        self.handles: dict[Path, object] = {}
        self.handle_opens = 0
        self.handle_hits = 0
        self.cache = _TensorLRU(CACHE_BUDGET_BYTES)
        self._last_log_loads = 0

        index_path = self.root / "model.safetensors.index.json"
        if index_path.is_file():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            self.weight_map = dict(payload.get("weight_map", {}))

    def _handle(self, shard: Path):
        handle = self.handles.get(shard)
        if handle is not None:
            self.handle_hits += 1
            return handle

        handle = self.stack.enter_context(
            safe_open(str(shard), framework="pt", device="cpu")
        )
        self.handles[shard] = handle
        self.handle_opens += 1
        return handle

    def _maybe_log_progress(self, name: str) -> None:
        if CACHE_LOG_INTERVAL <= 0:
            return

        stats = self.cache.snapshot()
        if stats["loads"] - self._last_log_loads < CACHE_LOG_INTERVAL:
            return

        self._last_log_loads = int(stats["loads"])
        used_mib = stats["bytes"] / (1024 * 1024)
        budget_mib = self.cache.max_bytes / (1024 * 1024)
        print(
            "cache progress: "
            f"loads={stats['loads']} | hits={stats['hits']} | "
            f"misses={stats['misses']} | hit_rate={stats['hit_rate']:.1f}% | "
            f"evictions={stats['evictions']} | "
            f"ram={used_mib:.1f}/{budget_mib:.1f} MiB | "
            f"last={name}"
        )

    def load(self, name: str, device: str):
        cached = self.cache.get(name)
        if cached is not None:
            self._maybe_log_progress(name)
            if device == "cpu":
                return cached
            return cached.to(device=device)

        shard_name = self.weight_map.get(name)
        if shard_name:
            shards = [self.root / shard_name]
        else:
            shards = sorted(self.root.glob("*.safetensors"))

        for shard in shards:
            if not shard.is_file():
                continue
            handle = self._handle(shard)
            if name in handle.keys():
                tensor = handle.get_tensor(name)
                self.cache.put(name, tensor)
                self._maybe_log_progress(name)
                if device == "cpu":
                    return tensor
                return tensor.to(device=device)

        raise KeyError(f"Tensor not found: {name}")

    def close(self) -> None:
        self.cache.clear()
        self.stack.close()
        self.handles.clear()


_stores: dict[Path, _ShardStore] = {}


def _store(root: Path) -> _ShardStore:
    key = root.resolve()
    store = _stores.get(key)
    if store is None:
        store = _ShardStore(key)
        _stores[key] = store
    return store


def _cached_load_tensor(root: Path, name: str, device: str = "cpu"):
    return _store(root).load(name, device)


# qwen36_40layer_loop resolves both names through its module globals. Patching
# them here preserves the exact reference computation and changes only the
# I/O/dequantization implementation.
base.load_tensor = _cached_load_tensor
base.dequantize_fp8_blockwise = dequantize_fp8_blockwise


@atexit.register
def _close_stores() -> None:
    for store in _stores.values():
        store.close()


def _format_mib(value: int) -> str:
    return f"{value / (1024 * 1024):.1f} MiB"


def main() -> None:
    print(
        "LRU config: "
        f"budget={CACHE_GB:.2f} GiB | "
        f"log_interval={CACHE_LOG_INTERVAL or 'off'}"
    )

    base.main()

    for root, store in _stores.items():
        stats = store.cache.snapshot()
        total = stats["hits"] + stats["misses"]
        print(
            "cached reader: "
            f"root={root} | "
            f"shards opened={store.handle_opens} | "
            f"cached handle hits={store.handle_hits}"
        )
        print(
            "LRU summary: "
            f"items={stats['items']} | "
            f"ram={_format_mib(int(stats['bytes']))}/{_format_mib(store.cache.max_bytes)} | "
            f"hits={stats['hits']} | misses={stats['misses']} | "
            f"hit_rate={stats['hit_rate']:.2f}% | "
            f"evictions={stats['evictions']} | "
            f"oversize_skips={stats['skipped_oversize']} | "
            f"loads={stats['loads']} | lookups={total}"
        )

        if stats["items"]:
            print("LRU status: hot tensor cache retained until process exit")


if __name__ == "__main__":
    main()
