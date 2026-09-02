from __future__ import annotations

"""Qwen3.6 loop with persistent shards and soft priority-aware RAM cache.

The cache keeps raw CPU tensors and uses the full RAM budget as one shared pool.
Eviction is global, but expert tensors receive a strong preservation bonus so
frequently reused MoE weights are less likely to be evicted than ordinary
attention/norm tensors.

Environment variables:
    QWEN36_CACHE_GB:
        Total RAM budget for raw tensor cache, default 3.0 GiB.
    QWEN36_EXPERT_BONUS:
        Preservation bonus for expert tensors in the eviction score,
        default 4.0.
    QWEN36_CACHE_LOG_INTERVAL:
        Print cache progress every N cache inserts, default 0 (disabled).
"""

import atexit
import json
import math
import os
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


CACHE_GB = _env_float("QWEN36_CACHE_GB", 3.0)
CACHE_BUDGET_BYTES = int(CACHE_GB * 1024 * 1024 * 1024)
EXPERT_BONUS = _env_float("QWEN36_EXPERT_BONUS", 4.0)
CACHE_LOG_INTERVAL = _env_int("QWEN36_CACHE_LOG_INTERVAL", 0)


def _is_expert_tensor(name: str) -> bool:
    marker = ".mlp.experts."
    if marker not in name:
        return False

    _, tail = name.split(marker, 1)
    parts = tail.split(".")
    if len(parts) < 2:
        return False

    try:
        int(parts[0])
    except ValueError:
        return False

    return parts[1] in {
        "gate_proj",
        "up_proj",
        "down_proj",
    }


class _PriorityTensorCache:
    """Single shared RAM cache with soft priority for repeatedly used experts."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes

        self.items: dict[str, object] = {}
        self.item_bytes: dict[str, int] = {}
        self.item_hits: dict[str, int] = {}
        self.item_last_access: dict[str, int] = {}
        self.item_expert: dict[str, bool] = {}
        self.clock = 0

        self.bytes_used = 0
        self.expert_bytes = 0
        self.general_bytes = 0

        self.hits = 0
        self.misses = 0
        self.expert_hits = 0
        self.expert_misses = 0
        self.general_hits = 0
        self.general_misses = 0

        self.evictions = 0
        self.expert_evictions = 0
        self.general_evictions = 0
        self.skipped_oversize = 0
        self.loads = 0

        self.lock = Lock()

    @staticmethod
    def _tensor_bytes(tensor) -> int:
        return int(tensor.numel()) * int(tensor.element_size())

    def _score(self, name: str) -> float:
        hits = self.item_hits.get(name, 0)
        last = self.item_last_access.get(name, 0)
        age = max(self.clock - last, 0)
        expert = self.item_expert.get(name, False)

        frequency = 3.0 * math.log1p(hits)
        recency = 8.0 / math.sqrt(1.0 + age)
        expert_bonus = EXPERT_BONUS if expert else 0.0
        return frequency + recency + expert_bonus

    def _select_victim(self) -> str | None:
        if not self.items:
            return None
        return min(self.items, key=self._score)

    def _remove(self, name: str) -> tuple[int, bool]:
        size = self.item_bytes.pop(name, 0)
        expert = self.item_expert.pop(name, False)

        self.items.pop(name, None)
        self.item_hits.pop(name, None)
        self.item_last_access.pop(name, None)

        self.bytes_used -= size
        if expert:
            self.expert_bytes -= size
        else:
            self.general_bytes -= size

        return size, expert

    def get(self, name: str):
        with self.lock:
            self.clock += 1
            tensor = self.items.get(name)

            if tensor is None:
                self.misses += 1
                if _is_expert_tensor(name):
                    self.expert_misses += 1
                else:
                    self.general_misses += 1
                return None

            self.hits += 1
            self.item_hits[name] = self.item_hits.get(name, 0) + 1
            self.item_last_access[name] = self.clock

            if self.item_expert.get(name, False):
                self.expert_hits += 1
            else:
                self.general_hits += 1

            return tensor

    def put(self, name: str, tensor) -> None:
        size = self._tensor_bytes(tensor)
        expert = _is_expert_tensor(name)

        with self.lock:
            self.clock += 1
            self.loads += 1

            if name in self.items:
                self._remove(name)

            if size > self.max_bytes:
                self.skipped_oversize += 1
                return

            while self.bytes_used + size > self.max_bytes:
                victim = self._select_victim()
                if victim is None:
                    break

                _, victim_expert = self._remove(victim)
                self.evictions += 1

                if victim_expert:
                    self.expert_evictions += 1
                else:
                    self.general_evictions += 1

            if self.bytes_used + size > self.max_bytes:
                self.skipped_oversize += 1
                return

            self.items[name] = tensor
            self.item_bytes[name] = size
            self.item_hits[name] = 0
            self.item_last_access[name] = self.clock
            self.item_expert[name] = expert

            self.bytes_used += size
            if expert:
                self.expert_bytes += size
            else:
                self.general_bytes += size

    def clear(self) -> None:
        with self.lock:
            self.items.clear()
            self.item_bytes.clear()
            self.item_hits.clear()
            self.item_last_access.clear()
            self.item_expert.clear()

            self.bytes_used = 0
            self.expert_bytes = 0
            self.general_bytes = 0

    def snapshot(self) -> dict[str, int | float]:
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total * 100.0 if total else 0.0

            expert_total = self.expert_hits + self.expert_misses
            expert_hit_rate = (
                self.expert_hits / expert_total * 100.0 if expert_total else 0.0
            )

            general_total = self.general_hits + self.general_misses
            general_hit_rate = (
                self.general_hits / general_total * 100.0 if general_total else 0.0
            )

            expert_items = sum(
                1 for value in self.item_expert.values() if value
            )
            expert_share = (
                self.expert_bytes / self.bytes_used * 100.0
                if self.bytes_used
                else 0.0
            )

            return {
                "items": len(self.items),
                "expert_items": expert_items,
                "general_items": len(self.items) - expert_items,
                "bytes": self.bytes_used,
                "expert_bytes": self.expert_bytes,
                "general_bytes": self.general_bytes,
                "expert_share": expert_share,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "expert_hits": self.expert_hits,
                "expert_misses": self.expert_misses,
                "expert_hit_rate": expert_hit_rate,
                "general_hits": self.general_hits,
                "general_misses": self.general_misses,
                "general_hit_rate": general_hit_rate,
                "evictions": self.evictions,
                "expert_evictions": self.expert_evictions,
                "general_evictions": self.general_evictions,
                "skipped_oversize": self.skipped_oversize,
                "loads": self.loads,
            }


class _ShardStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.stack = ExitStack()
        self.weight_map: dict[str, str] = {}
        self.handles: dict[Path, object] = {}
        self.handle_opens = 0
        self.handle_hits = 0
        self.cache = _PriorityTensorCache(CACHE_BUDGET_BYTES)
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
        expert_mib = stats["expert_bytes"] / (1024 * 1024)

        print(
            "cache progress: "
            f"loads={stats['loads']} | hits={stats['hits']} | "
            f"misses={stats['misses']} | hit_rate={stats['hit_rate']:.1f}% | "
            f"expert_hit={stats['expert_hit_rate']:.1f}% | "
            f"general_hit={stats['general_hit_rate']:.1f}% | "
            f"evictions={stats['evictions']} "
            f"(expert={stats['expert_evictions']}, general={stats['general_evictions']}) | "
            f"ram={used_mib:.1f}/{budget_mib:.1f} MiB | "
            f"experts_ram={expert_mib:.1f} MiB "
            f"({stats['expert_share']:.1f}%) | "
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
        f"mode=global-soft-priority | "
        f"expert_bonus={EXPERT_BONUS:.2f} | "
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
            f"items={stats['items']} "
            f"(expert={stats['expert_items']}, general={stats['general_items']}) | "
            f"ram={_format_mib(int(stats['bytes']))}/"
            f"{_format_mib(store.cache.max_bytes)} | "
            f"expert_ram={_format_mib(int(stats['expert_bytes']))} "
            f"({stats['expert_share']:.1f}%) | "
            f"hits={stats['hits']} | misses={stats['misses']} | "
            f"hit_rate={stats['hit_rate']:.2f}% | "
            f"expert_hit_rate={stats['expert_hit_rate']:.2f}% | "
            f"general_hit_rate={stats['general_hit_rate']:.2f}% | "
            f"evictions={stats['evictions']} "
            f"(expert={stats['expert_evictions']}, general={stats['general_evictions']}) | "
            f"oversize_skips={stats['skipped_oversize']} | "
            f"loads={stats['loads']} | lookups={total}"
        )


if __name__ == "__main__":
    main()
