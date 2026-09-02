from __future__ import annotations

"""Qwen3.6 loop with hierarchical RAM/VRAM tensor caches.

Cache hierarchy for CUDA runs:

    VRAM cache -> RAM cache -> SSD shards

The VRAM cache keeps raw weight tensors resident on the GPU when possible.
The RAM cache remains the larger fallback tier. Both caches use a soft
frequency/recency priority score, with a preservation bonus for MoE experts.

Environment variables:
    QWEN36_CACHE_GB:
        Total RAM cache budget in GiB, default 3.0.
    QWEN36_VRAM_GB:
        Optional per-process CUDA allocator limit in GiB, default 0 (disabled).
    QWEN36_VRAM_CACHE_GB:
        VRAM tensor-cache budget in GiB, default 3.0.
    QWEN36_EXPERT_BONUS:
        Preservation bonus for expert tensors, default 4.0.
    QWEN36_CACHE_LOG_INTERVAL:
        Print cache progress every N RAM/VRAM cache inserts, default 0.
"""

import atexit
import json
import math
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from threading import Lock

import torch
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
VRAM_GB = _env_float("QWEN36_VRAM_GB", 0.0)
VRAM_CACHE_GB = _env_float("QWEN36_VRAM_CACHE_GB", 3.0)
VRAM_CACHE_BUDGET_BYTES = int(VRAM_CACHE_GB * 1024 * 1024 * 1024)
EXPERT_BONUS = _env_float("QWEN36_EXPERT_BONUS", 4.0)
CACHE_LOG_INTERVAL = _env_int("QWEN36_CACHE_LOG_INTERVAL", 0)


def _requested_device() -> str:
    for index, arg in enumerate(sys.argv):
        if arg == "--device" and index + 1 < len(sys.argv):
            return sys.argv[index + 1].lower()
        if arg.startswith("--device="):
            return arg.split("=", 1)[1].lower()
    return "cpu"


def _configure_vram_limit(device: str) -> None:
    """Optionally cap this process's CUDA allocator before CUDA allocations."""
    if device != "cuda" or VRAM_GB <= 0:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable but QWEN36_VRAM_GB was requested")

    props = torch.cuda.get_device_properties(0)
    total_gib = props.total_memory / (1024 ** 3)
    fraction = VRAM_GB / total_gib

    if fraction >= 1.0:
        print(
            f"VRAM limit: requested {VRAM_GB:.2f} GiB >= detected "
            f"{total_gib:.2f} GiB; leaving allocator uncapped"
        )
        return

    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    print(
        f"VRAM limit: {VRAM_GB:.2f} GiB / {total_gib:.2f} GiB "
        f"({fraction * 100.0:.1f}% of allocator limit)"
    )


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

    return parts[1] in {"gate_proj", "up_proj", "down_proj"}


class _PriorityTensorCache:
    """Generic soft-priority tensor cache for CPU or CUDA tensors."""

    def __init__(self, max_bytes: int, name: str) -> None:
        self.max_bytes = max_bytes
        self.name = name

        self.items: dict[str, torch.Tensor] = {}
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
    def _tensor_bytes(tensor: torch.Tensor) -> int:
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

    def get(self, name: str) -> torch.Tensor | None:
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

    def put(self, name: str, tensor: torch.Tensor) -> None:
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
            expert_total = self.expert_hits + self.expert_misses
            general_total = self.general_hits + self.general_misses
            expert_items = sum(1 for value in self.item_expert.values() if value)
            return {
                "items": len(self.items),
                "expert_items": expert_items,
                "general_items": len(self.items) - expert_items,
                "bytes": self.bytes_used,
                "expert_bytes": self.expert_bytes,
                "general_bytes": self.general_bytes,
                "expert_share": self.expert_bytes / self.bytes_used * 100.0 if self.bytes_used else 0.0,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / total * 100.0 if total else 0.0,
                "expert_hits": self.expert_hits,
                "expert_misses": self.expert_misses,
                "expert_hit_rate": self.expert_hits / expert_total * 100.0 if expert_total else 0.0,
                "general_hits": self.general_hits,
                "general_misses": self.general_misses,
                "general_hit_rate": self.general_hits / general_total * 100.0 if general_total else 0.0,
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
        self.ram_cache = _PriorityTensorCache(CACHE_BUDGET_BYTES, "ram")
        self.vram_cache = _PriorityTensorCache(VRAM_CACHE_BUDGET_BYTES, "vram")
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

        ram = self.ram_cache.snapshot()
        vram = self.vram_cache.snapshot()
        total_loads = int(ram["loads"] + vram["loads"])
        if total_loads - self._last_log_loads < CACHE_LOG_INTERVAL:
            return
        self._last_log_loads = total_loads

        print(
            "cache progress: "
            f"ram={ram['bytes'] / (1024 ** 2):.1f} MiB | "
            f"vram={vram['bytes'] / (1024 ** 2):.1f} MiB | "
            f"ram_hit={ram['hit_rate']:.1f}% | "
            f"vram_hit={vram['hit_rate']:.1f}% | "
            f"vram_expert={vram['expert_share']:.1f}% | "
            f"evict_ram={ram['evictions']} | evict_vram={vram['evictions']} | "
            f"last={name}"
        )

    def _load_ssd(self, name: str) -> torch.Tensor:
        shard_name = self.weight_map.get(name)
        shards = [self.root / shard_name] if shard_name else sorted(self.root.glob("*.safetensors"))

        for shard in shards:
            if not shard.is_file():
                continue
            handle = self._handle(shard)
            if name in handle.keys():
                return handle.get_tensor(name)

        raise KeyError(f"Tensor not found: {name}")

    def load(self, name: str, device: str):
        if device == "cuda":
            gpu_cached = self.vram_cache.get(name)
            if gpu_cached is not None:
                self._maybe_log_progress(name)
                return gpu_cached

        cpu_cached = self.ram_cache.get(name)
        if cpu_cached is not None:
            if device == "cpu":
                self._maybe_log_progress(name)
                return cpu_cached

            gpu_tensor = cpu_cached.to(device=device)
            self.vram_cache.put(name, gpu_tensor)
            self._maybe_log_progress(name)
            return gpu_tensor

        tensor = self._load_ssd(name)
        self.ram_cache.put(name, tensor)

        if device == "cpu":
            self._maybe_log_progress(name)
            return tensor

        gpu_tensor = tensor.to(device=device)
        self.vram_cache.put(name, gpu_tensor)
        self._maybe_log_progress(name)
        return gpu_tensor

    def close(self) -> None:
        self.vram_cache.clear()
        self.ram_cache.clear()
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
    device = _requested_device()
    _configure_vram_limit(device)

    print(
        "Cache config: "
        f"ram={CACHE_GB:.2f} GiB | "
        f"vram_cache={VRAM_CACHE_GB:.2f} GiB | "
        f"mode=hierarchical-vram-ram-ssd | "
        f"expert_bonus={EXPERT_BONUS:.2f} | "
        f"vram_limit={VRAM_GB:.2f} GiB | "
        f"log_interval={CACHE_LOG_INTERVAL or 'off'}"
    )

    base.main()

    for root, store in _stores.items():
        ram = store.ram_cache.snapshot()
        vram = store.vram_cache.snapshot()
        total = int(ram["hits"] + ram["misses"])

        print(
            "cached reader: "
            f"root={root} | "
            f"shards opened={store.handle_opens} | "
            f"cached handle hits={store.handle_hits}"
        )
        print(
            "RAM cache: "
            f"items={ram['items']} "
            f"(expert={ram['expert_items']}, general={ram['general_items']}) | "
            f"ram={_format_mib(int(ram['bytes']))}/{_format_mib(store.ram_cache.max_bytes)} | "
            f"expert_ram={_format_mib(int(ram['expert_bytes']))} "
            f"({ram['expert_share']:.1f}%) | "
            f"hits={ram['hits']} | misses={ram['misses']} | "
            f"hit_rate={ram['hit_rate']:.2f}% | "
            f"expert_hit_rate={ram['expert_hit_rate']:.2f}% | "
            f"evictions={ram['evictions']} "
            f"(expert={ram['expert_evictions']}, general={ram['general_evictions']}) | "
            f"loads={ram['loads']} | lookups={total}"
        )
        print(
            "VRAM cache: "
            f"items={vram['items']} "
            f"(expert={vram['expert_items']}, general={vram['general_items']}) | "
            f"vram={_format_mib(int(vram['bytes']))}/{_format_mib(store.vram_cache.max_bytes)} | "
            f"expert_vram={_format_mib(int(vram['expert_bytes']))} "
            f"({vram['expert_share']:.1f}%) | "
            f"hits={vram['hits']} | misses={vram['misses']} | "
            f"hit_rate={vram['hit_rate']:.2f}% | "
            f"expert_hit_rate={vram['expert_hit_rate']:.2f}% | "
            f"evictions={vram['evictions']} "
            f"(expert={vram['expert_evictions']}, general={vram['general_evictions']}) | "
            f"loads={vram['loads']}"
        )


if __name__ == "__main__":
    main()
