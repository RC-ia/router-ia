from __future__ import annotations

"""Qwen3.6 loop with persistent shards and priority-aware RAM tensor cache.

The cache keeps raw CPU tensors and uses two residency pools:
- expert tensors: reserved budget, scored by frequency + recency + expert bonus;
- general tensors: normal recency/frequency scoring.

This avoids evicting frequently reused MoE expert weights merely because many
attention/norm tensors were touched afterward.

Environment variables:
    QWEN36_CACHE_GB: total RAM budget for raw tensor cache, default 2.5 GiB.
    QWEN36_EXPERT_CACHE_RATIO: fraction reserved for expert tensors, default 0.80.
    QWEN36_CACHE_LOG_INTERVAL: print cache progress every N tensor loads,
        default 0 (disabled).
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


CACHE_GB = _env_float("QWEN36_CACHE_GB", 2.5)
CACHE_BUDGET_BYTES = int(CACHE_GB * 1024 * 1024 * 1024)
EXPERT_CACHE_RATIO = min(_env_float("QWEN36_EXPERT_CACHE_RATIO", 0.80), 0.95)
EXPERT_CACHE_BUDGET_BYTES = int(CACHE_BUDGET_BYTES * EXPERT_CACHE_RATIO)
GENERAL_CACHE_BUDGET_BYTES = CACHE_BUDGET_BYTES - EXPERT_CACHE_BUDGET_BYTES
CACHE_LOG_INTERVAL = _env_int("QWEN36_CACHE_LOG_INTERVAL", 0)


def _is_expert_tensor(name: str) -> bool:
    marker = ".mlp.experts."
    if marker not in name:
        return False
    prefix, tail = name.split(marker, 1)
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
    } or parts[1] in {
        "gate_proj.weight_scale_inv",
        "up_proj.weight_scale_inv",
        "down_proj.weight_scale_inv",
    }


class _PriorityTensorCache:
    """Bounded cache that protects hot experts from ordinary tensor churn."""

    def __init__(self, max_bytes: int, expert_budget_bytes: int) -> None:
        self.max_bytes = max_bytes
        self.expert_budget_bytes = expert_budget_bytes
        self.general_budget_bytes = max_bytes - expert_budget_bytes

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

        # Frequency has a strong influence; recency prevents ancient hot items
        # from becoming permanently immortal. Experts receive a modest bonus.
        frequency = math.log1p(hits)
        recency = 1.0 / (1.0 + age)
        bonus = 1.75 if self.item_expert.get(name, False) else 0.0
        return (frequency * 2.0) + (recency * 10.0) + bonus

    def _usage_for(self, expert: bool) -> int:
        return self.expert_bytes if expert else self.general_bytes

    def _budget_for(self, expert: bool) -> int:
        return self.expert_budget_bytes if expert else self.general_budget_bytes

    def _select_victim(self, expert: bool) -> str | None:
        candidates = [
            name
            for name in self.items
            if self.item_expert.get(name, False) == expert
        ]
        if not candidates:
            return None
        return min(candidates, key=self._score)

    def _remove(self, name: str) -> None:
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

    def get(self, name: str):
        with self.lock:
            self.clock += 1
            tensor = self.items.get(name)
            expert = self.item_expert.get(name, False)
            if tensor is None:
                self.misses += 1
                if expert or _is_expert_tensor(name):
                    self.expert_misses += 1
                else:
                    self.general_misses += 1
                return None

            self.hits += 1
            self.item_hits[name] += 1
            self.item_last_access[name] = self.clock
            if expert:
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

            previous = self.items.get(name)
            if previous is not None:
                self._remove(name)

            if size > self.max_bytes:
                self.skipped_oversize += 1
                return

            pool_budget = self._budget_for(expert)
            while self._usage_for(expert) + size > pool_budget:
                victim = self._select_victim(expert)
                if victim is None:
                    # A newly inserted general tensor may still use free expert
                    # capacity, and vice versa. Borrow only unused capacity.
                    other = not expert
                    if self._usage_for(expert) + size > pool_budget:
                        free_other = self._budget_for(other) - self._usage_for(other)
                        free_total = self.max_bytes - self.bytes_used
                        if free_other + free_total < size:
                            victim = self._select_victim(other)
                    if victim is None:
                        break
                victim_expert = self.item_expert.get(victim, False)
                self._remove(victim)
                self.evictions += 1
                if victim_expert:
                    self.expert_evictions += 1
                else:
                    self.general_evictions += 1

            while self.bytes_used + size > self.max_bytes:
                victim = self._select_victim(expert)
                if victim is None:
                    victim = self._select_victim(False)
                if victim is None:
                    victim = self._select_victim(True)
                if victim is None:
                    break
                victim_expert = self.item_expert.get(victim, False)
                self._remove(victim)
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
            hit_rate = (self.hits / total * 100.0) if total else 0.0
            expert_total = self.expert_hits + self.expert_misses
            general_total = self.general_hits + self.general_misses
            expert_hit_rate = (
                self.expert_hits / expert_total * 100.0 if expert_total else 0.0
            )
            general_hit_rate = (
                self.general_hits / general_total * 100.0 if general_total else 0.0
            )
            expert_items = sum(1 for value in self.item_expert.values() if value)
            general_items = len(self.items) - expert_items
            return {
                "items": len(self.items),
                "expert_items": expert_items,
                "general_items": general_items,
                "bytes": self.bytes_used,
                "expert_bytes": self.expert_bytes,
                "general_bytes": self.general_bytes,
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
        self.cache = _PriorityTensorCache(CACHE_BUDGET_BYTES, EXPERT_CACHE_BUDGET_BYTES)
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
        expert_budget_mib = self.cache.expert_budget_bytes / (1024 * 1024) if hasattr(self.cache, 'expert_budget_bytes') else EXPERT_CACHE_BUDGET_BYTES / (1024 * 1024)
        print(
            "cache progress: "
            f"loads={stats['loads']} | hits={stats['hits']} | misses={stats['misses']} | "
            f"hit_rate={stats['hit_rate']:.1f}% | "
            f"expert_hit={stats['expert_hit_rate']:.1f}% | "
            f"evictions={stats['evictions']} (expert={stats['expert_evictions']}, general={stats['general_evictions']}) | "
            f"ram={used_mib:.1f}/{budget_mib:.1f} MiB | "
            f"experts_ram={expert_mib:.1f}/{expert_budget_mib:.1f} MiB | "
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


# qwen36_40layer_loop resolves these names through its module globals.
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
        f"expert_ratio={EXPERT_CACHE_RATIO:.0%} | "
        f"expert_budget={_format_mib(EXPERT_CACHE_BUDGET_BYTES)} | "
        f"general_budget={_format_mib(GENERAL_CACHE_BUDGET_BYTES)} | "
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
            f"items={stats['items']} (expert={stats['expert_items']}, general={stats['general_items']}) | "
            f"ram={_format_mib(int(stats['bytes']))}/{_format_mib(store.cache.max_bytes)} | "
            f"expert_ram={_format_mib(int(stats['expert_bytes']))}/{_format_mib(EXPERT_CACHE_BUDGET_BYTES)} | "
            f"general_ram={_format_mib(int(stats['general_bytes']))}/{_format_mib(GENERAL_CACHE_BUDGET_BYTES)}"
        )
        print(
            "LRU hits: "
            f"total={stats['hits']} | misses={stats['misses']} | hit_rate={stats['hit_rate']:.2f}% | "
            f"expert={stats['expert_hits']}/{stats['expert_hits'] + stats['expert_misses']} ({stats['expert_hit_rate']:.2f}%) | "
            f"general={stats['general_hits']}/{stats['general_hits'] + stats['general_misses']} ({stats['general_hit_rate']:.2f}%)"
        )
        print(
            "LRU evictions: "
            f"total={stats['evictions']} | expert={stats['expert_evictions']} | "
            f"general={stats['general_evictions']} | oversize_skips={stats['skipped_oversize']} | "
            f"loads={stats['loads']} | lookups={total}"
        )


if __name__ == "__main__":
    main()
