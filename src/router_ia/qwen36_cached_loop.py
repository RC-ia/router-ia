from __future__ import annotations

"""Qwen3.6 loop with hierarchical RAM/VRAM tensor caches.

CUDA cache layout:

    resident VRAM + hot-expert VRAM + streaming window -> RAM -> SSD shards

The resident pool never evicts. The hot-expert pool evicts only experts.
The streaming window is transient and is intended to feed the current layer.

Environment variables:
    QWEN36_CACHE_GB: RAM cache budget, default 3.0 GiB.
    QWEN36_VRAM_GB: optional CUDA allocator limit, default 0 (disabled).
    QWEN36_VRAM_CACHE_GB: persistent VRAM cache budget, default 3.0 GiB.
    QWEN36_RESIDENT_VRAM_RATIO: resident share, default 0.60.
    QWEN36_VRAM_STREAM_GB: streaming window, default 0.60 GiB.
    QWEN36_EXPERT_BONUS: hot-expert priority bonus, default 4.0.
    QWEN36_CACHE_LOG_INTERVAL: progress interval, default 0.
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


CACHE_GB = _env_float("QWEN36_CACHE_GB", 3.0)
CACHE_BUDGET_BYTES = int(CACHE_GB * 1024**3)
VRAM_GB = _env_float("QWEN36_VRAM_GB", 0.0)
VRAM_CACHE_GB = _env_float("QWEN36_VRAM_CACHE_GB", 3.0)
VRAM_CACHE_BUDGET_BYTES = int(VRAM_CACHE_GB * 1024**3)
RESIDENT_VRAM_RATIO = min(_env_float("QWEN36_RESIDENT_VRAM_RATIO", 0.60), 0.95)
# Give the streaming path more VRAM and deliberately keep only a small
# persistent expert pool. This favors RAM -> GPU -> compute over expert churn.
STREAM_GB = min(
    _env_float("QWEN36_VRAM_STREAM_GB", 0.60),
    max(VRAM_CACHE_GB - 0.10, 0.01),
)
STREAM_BUDGET_BYTES = int(STREAM_GB * 1024**3)
RESIDENT_VRAM_BUDGET_BYTES = int(VRAM_CACHE_BUDGET_BYTES * RESIDENT_VRAM_RATIO)
EXPERT_VRAM_BUDGET_BYTES = max(
    VRAM_CACHE_BUDGET_BYTES - RESIDENT_VRAM_BUDGET_BYTES - STREAM_BUDGET_BYTES,
    0,
)
EXPERT_BONUS = _env_float("QWEN36_EXPERT_BONUS", 4.0)
CACHE_LOG_INTERVAL = int(_env_float("QWEN36_CACHE_LOG_INTERVAL", 0.0))


def _requested_device() -> str:
    for index, arg in enumerate(sys.argv):
        if arg == "--device" and index + 1 < len(sys.argv):
            return sys.argv[index + 1].lower()
        if arg.startswith("--device="):
            return arg.split("=", 1)[1].lower()
    return "cpu"


def _configure_vram_limit(device: str) -> None:
    if device != "cuda" or VRAM_GB <= 0:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable but QWEN36_VRAM_GB was requested")

    props = torch.cuda.get_device_properties(0)
    total_gib = props.total_memory / 1024**3
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
    def __init__(self, max_bytes: int, name: str, evict: bool = True) -> None:
        self.max_bytes = max_bytes
        self.name = name
        self.evict = evict
        self.items: dict[str, torch.Tensor] = {}
        self.item_bytes: dict[str, int] = {}
        self.item_hits: dict[str, int] = {}
        self.item_last_access: dict[str, int] = {}
        self.item_expert: dict[str, bool] = {}
        self.clock = 0
        self.bytes_used = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expert_hits = 0
        self.expert_misses = 0
        self.expert_evictions = 0
        self.loads = 0
        self.skipped_oversize = 0
        self.lock = Lock()

    @staticmethod
    def _tensor_bytes(tensor: torch.Tensor) -> int:
        return int(tensor.numel()) * int(tensor.element_size())

    def _score(self, name: str) -> float:
        hits = self.item_hits.get(name, 0)
        age = max(self.clock - self.item_last_access.get(name, 0), 0)
        return 3.0 * math.log1p(hits) + 8.0 / math.sqrt(1.0 + age) + (
            EXPERT_BONUS if self.item_expert.get(name, False) else 0.0
        )

    def _remove(self, name: str) -> tuple[int, bool]:
        size = self.item_bytes.pop(name, 0)
        expert = self.item_expert.pop(name, False)
        self.items.pop(name, None)
        self.item_hits.pop(name, None)
        self.item_last_access.pop(name, None)
        self.bytes_used -= size
        return size, expert

    def get(self, name: str) -> torch.Tensor | None:
        with self.lock:
            self.clock += 1
            tensor = self.items.get(name)
            if tensor is None:
                self.misses += 1
                if _is_expert_tensor(name):
                    self.expert_misses += 1
                return None
            self.hits += 1
            self.item_hits[name] = self.item_hits.get(name, 0) + 1
            self.item_last_access[name] = self.clock
            if self.item_expert.get(name, False):
                self.expert_hits += 1
            return tensor

    def put(self, name: str, tensor: torch.Tensor) -> bool:
        size = self._tensor_bytes(tensor)
        expert = _is_expert_tensor(name)
        with self.lock:
            self.clock += 1
            self.loads += 1
            if name in self.items:
                self._remove(name)
            if size > self.max_bytes:
                self.skipped_oversize += 1
                return False
            if self.evict:
                while self.bytes_used + size > self.max_bytes and self.items:
                    victim = min(self.items, key=self._score)
                    self._remove(victim)
                    self.evictions += 1
                    if _is_expert_tensor(victim):
                        self.expert_evictions += 1
            elif self.bytes_used + size > self.max_bytes:
                self.skipped_oversize += 1
                return False
            if self.bytes_used + size > self.max_bytes:
                self.skipped_oversize += 1
                return False
            self.items[name] = tensor
            self.item_bytes[name] = size
            self.item_hits[name] = 0
            self.item_last_access[name] = self.clock
            self.item_expert[name] = expert
            self.bytes_used += size
            return True

    def clear(self) -> None:
        with self.lock:
            self.items.clear()
            self.item_bytes.clear()
            self.item_hits.clear()
            self.item_last_access.clear()
            self.item_expert.clear()
            self.bytes_used = 0

    def snapshot(self) -> dict[str, int | float]:
        with self.lock:
            total = self.hits + self.misses
            expert_total = self.expert_hits + self.expert_misses
            expert_items = sum(self.item_expert.values())
            expert_bytes = sum(
                self.item_bytes[name]
                for name, value in self.item_expert.items()
                if value
            )
            general_bytes = self.bytes_used - expert_bytes
            return {
                "items": len(self.items),
                "expert_items": expert_items,
                "general_items": len(self.items) - expert_items,
                "bytes": self.bytes_used,
                "expert_bytes": expert_bytes,
                "general_bytes": general_bytes,
                "expert_share": expert_bytes / self.bytes_used * 100.0 if self.bytes_used else 0.0,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / total * 100.0 if total else 0.0,
                "expert_hits": self.expert_hits,
                "expert_misses": self.expert_misses,
                "expert_hit_rate": self.expert_hits / expert_total * 100.0 if expert_total else 0.0,
                "evictions": self.evictions,
                "expert_evictions": self.expert_evictions,
                "general_evictions": self.evictions - self.expert_evictions,
                "skipped_oversize": self.skipped_oversize,
                "loads": self.loads,
            }


class _DualVRAMCache:
    """Resident pool + small expert cache + larger rotating stream pool."""

    def __init__(self, total_bytes: int) -> None:
        self.resident = _PriorityTensorCache(
            RESIDENT_VRAM_BUDGET_BYTES, "vram-resident", evict=False
        )
        self.experts = _PriorityTensorCache(
            EXPERT_VRAM_BUDGET_BYTES, "vram-experts", evict=True
        )
        self.stream = _PriorityTensorCache(
            STREAM_BUDGET_BYTES, "vram-stream", evict=True
        )
        self.max_bytes = total_bytes

    def _persistent_pool(self, name: str) -> _PriorityTensorCache:
        return self.experts if _is_expert_tensor(name) else self.resident

    def get(self, name: str) -> torch.Tensor | None:
        return self._persistent_pool(name).get(name)

    def put(self, name: str, tensor: torch.Tensor) -> bool:
        return self._persistent_pool(name).put(name, tensor)

    def get_stream(self, name: str) -> torch.Tensor | None:
        return self.stream.get(name)

    def put_stream(self, name: str, tensor: torch.Tensor) -> bool:
        return self.stream.put(name, tensor)

    def clear_stream(self) -> None:
        self.stream.clear()

    def clear(self) -> None:
        self.resident.clear()
        self.experts.clear()
        self.stream.clear()

    def snapshot(self) -> dict[str, int | float]:
        resident = self.resident.snapshot()
        experts = self.experts.snapshot()
        stream = self.stream.snapshot()
        hits = int(resident["hits"] + experts["hits"] + stream["hits"])
        misses = int(resident["misses"] + experts["misses"] + stream["misses"])
        bytes_used = int(resident["bytes"] + experts["bytes"] + stream["bytes"])
        return {
            "items": int(resident["items"] + experts["items"] + stream["items"]),
            "expert_items": int(experts["expert_items"]),
            "general_items": int(resident["items"]),
            "stream_items": int(stream["items"]),
            "bytes": bytes_used,
            "expert_bytes": int(experts["bytes"]),
            "general_bytes": int(resident["bytes"]),
            "stream_bytes": int(stream["bytes"]),
            "expert_share": int(experts["bytes"]) / bytes_used * 100.0 if bytes_used else 0.0,
            "hits": hits,
            "misses": misses,
            "hit_rate": hits / (hits + misses) * 100.0 if hits + misses else 0.0,
            "expert_hits": int(experts["hits"]),
            "expert_misses": int(experts["misses"]),
            "expert_hit_rate": experts["hit_rate"],
            "general_hits": int(resident["hits"]),
            "general_misses": int(resident["misses"]),
            "general_hit_rate": resident["hit_rate"],
            "stream_hits": int(stream["hits"]),
            "stream_misses": int(stream["misses"]),
            "stream_hit_rate": stream["hit_rate"],
            "evictions": int(experts["evictions"]),
            "expert_evictions": int(experts["evictions"]),
            "general_evictions": 0,
            "stream_evictions": int(stream["evictions"]),
            "skipped_oversize": int(
                resident["skipped_oversize"]
                + experts["skipped_oversize"]
                + stream["skipped_oversize"]
            ),
            "loads": int(resident["loads"] + experts["loads"] + stream["loads"]),
            "resident_items": int(resident["items"]),
            "resident_bytes": int(resident["bytes"]),
            "resident_budget_bytes": RESIDENT_VRAM_BUDGET_BYTES,
            "resident_skipped": int(resident["skipped_oversize"]),
            "expert_budget_bytes": EXPERT_VRAM_BUDGET_BYTES,
            "expert_pool_items": int(experts["expert_items"]),
            "expert_pool_hit_rate": experts["hit_rate"],
            "stream_budget_bytes": STREAM_BUDGET_BYTES,
        }


class _ShardStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.stack = ExitStack()
        self.weight_map: dict[str, str] = {}
        self.handles: dict[Path, object] = {}
        self.handle_opens = 0
        self.handle_hits = 0
        self.ram_cache = _PriorityTensorCache(CACHE_BUDGET_BYTES, "ram", evict=True)
        self.vram_cache = _DualVRAMCache(VRAM_CACHE_BUDGET_BYTES)
        self.target_device = "cpu"
        self._last_log_loads = 0
        self._stream_layer: int | None = None

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
            f"ram={ram['bytes'] / 1024**2:.1f} MiB | "
            f"vram={vram['bytes'] / 1024**2:.1f} MiB "
            f"(resident={vram['resident_bytes'] / 1024**2:.1f}, "
            f"experts={vram['expert_bytes'] / 1024**2:.1f}, "
            f"stream={vram['stream_bytes'] / 1024**2:.1f}) | "
            f"ram_hit={ram['hit_rate']:.1f}% | vram_hit={vram['hit_rate']:.1f}% | "
            f"expert_vram_hit={vram['expert_pool_hit_rate']:.1f}% | "
            f"stream_hit={vram['stream_hit_rate']:.1f}% | "
            f"expert_evict={vram['expert_evictions']} | "
            f"stream_evict={vram['stream_evictions']} | last={name}"
        )

    def _load_ssd(self, name: str) -> torch.Tensor:
        shard_name = self.weight_map.get(name)
        shards = [self.root / shard_name] if shard_name else sorted(
            self.root.glob("*.safetensors")
        )
        for shard in shards:
            if not shard.is_file():
                continue
            handle = self._handle(shard)
            if name in handle.keys():
                return handle.get_tensor(name)
        raise KeyError(f"Tensor not found: {name}")

    def load(self, name: str, device: str):
        if device == "cuda":
            self.target_device = "cuda"
            cached_tensor = self.vram_cache.get(name)
            if cached_tensor is not None:
                self._maybe_log_progress(name)
                return cached_tensor

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

    def load_projection(self, prefix: str, device: str) -> torch.Tensor:
        self.target_device = device
        cache_key = prefix + ".__projection__"
        if device == "cuda":
            cached_projection = self.vram_cache.get(cache_key)
            if cached_projection is not None:
                self._maybe_log_progress(cache_key)
                return cached_projection

        weight = self.load(prefix + ".weight", device="cpu")
        if weight.dtype == torch.float8_e4m3fn and device == "cuda":
            scale = self.load(prefix + ".weight_scale_inv", device="cpu")
            gpu_weight = weight.to(device="cuda")
            gpu_scale = scale.to(device="cuda")
            del weight, scale
            out = dequantize_fp8_blockwise(gpu_weight, gpu_scale).to(dtype=torch.float16)
            del gpu_weight, gpu_scale
            self.vram_cache.put(cache_key, out)
            self._maybe_log_progress(cache_key)
            return out

        if weight.dtype == torch.float8_e4m3fn:
            scale = self.load(prefix + ".weight_scale_inv", device="cpu")
            out = dequantize_fp8_blockwise(weight, scale)
            del scale, weight
        else:
            out = weight.float()
            del weight

        if device == "cuda":
            out = out.to(device="cuda", dtype=torch.float16)
            self.vram_cache.put(cache_key, out)
        self._maybe_log_progress(cache_key)
        return out

    def stream_projection(self, prefix: str) -> torch.Tensor:
        """Stream one expert projection from RAM into the rotating staging pool."""
        if self.target_device != "cuda":
            raise RuntimeError("stream_projection requires CUDA")

        layer_prefix_marker = ".layers."
        layer_id: int | None = None
        if layer_prefix_marker in prefix:
            try:
                layer_id = int(prefix.split(layer_prefix_marker, 1)[1].split(".", 1)[0])
            except (ValueError, IndexError):
                layer_id = None

        if layer_id is not None and layer_id != self._stream_layer:
            self.vram_cache.clear_stream()
            self._stream_layer = layer_id

        stream_key = prefix + ".__stream__"
        cached_stream = self.vram_cache.get_stream(stream_key)
        if cached_stream is not None:
            return cached_stream

        weight = self.load(prefix + ".weight", device="cpu")
        if weight.dtype == torch.float8_e4m3fn:
            scale = self.load(prefix + ".weight_scale_inv", device="cpu")
            gpu_weight = weight.to(device="cuda")
            gpu_scale = scale.to(device="cuda")
            del weight, scale
            gpu_out = dequantize_fp8_blockwise(gpu_weight, gpu_scale).to(dtype=torch.float16)
            del gpu_weight, gpu_scale
        else:
            gpu_out = weight.to(device="cuda", dtype=torch.float16)
            del weight

        self.vram_cache.put_stream(stream_key, gpu_out)
        self._maybe_log_progress(stream_key)
        return gpu_out

    def clear_stream(self) -> None:
        self.vram_cache.clear_stream()
        self._stream_layer = None

    def runtime_tensor(
        self,
        name: str,
        device: str,
        dtype: torch.dtype | None = torch.float32,
    ) -> torch.Tensor:
        self.target_device = device
        cache_key = name + ".__runtime__"
        if device == "cuda":
            cached_tensor = self.vram_cache.get(cache_key)
            if cached_tensor is not None:
                return cached_tensor
        tensor = self.load(name, device="cpu")
        if dtype is not None:
            tensor = tensor.float() if dtype == torch.float32 else tensor.to(dtype=dtype)
        out = tensor.to(device)
        if device == "cuda":
            self.vram_cache.put(cache_key, out)
        return out

    def close(self) -> None:
        self.vram_cache.clear()
        self.ram_cache.clear()
        self.stack.close()
        self.handles.clear()


_stores: dict[Path, _ShardStore] = {}
_tensor_origins: dict[int, tuple[Path, str]] = {}


def _store(root: Path) -> _ShardStore:
    key = root.resolve()
    store = _stores.get(key)
    if store is None:
        store = _ShardStore(key)
        _stores[key] = store
    return store


def _cached_load_tensor(root: Path, name: str, device: str = "cpu"):
    store = _store(root)
    tensor = store.load(name, device)
    _tensor_origins[id(tensor)] = (root.resolve(), name)
    return tensor


def _dequantize_for_store(
    store: _ShardStore,
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    cache_key: str,
) -> torch.Tensor:
    if store.target_device != "cuda":
        return dequantize_fp8_blockwise(weight, scale_inv)

    cached_tensor = store.vram_cache.get(cache_key)
    if cached_tensor is not None:
        return cached_tensor

    gpu_weight = weight.to(device="cuda")
    gpu_scale = scale_inv.to(device="cuda")
    gpu_out = dequantize_fp8_blockwise(gpu_weight, gpu_scale).to(dtype=torch.float16)
    del gpu_weight, gpu_scale
    store.vram_cache.put(cache_key, gpu_out)
    store._maybe_log_progress(cache_key)
    return gpu_out


def _cached_dequantize(weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    origin = _tensor_origins.get(id(weight))
    if origin is None:
        return dequantize_fp8_blockwise(weight, scale_inv)
    root, name = origin
    store = _store(root)
    return _dequantize_for_store(store, weight, scale_inv, name + ".__dequant__")


def _cached_load_projection(root: Path, prefix: str, device: str) -> torch.Tensor:
    store = _store(root)
    if device == "cuda" and _is_expert_tensor(prefix + ".weight"):
        return store.stream_projection(prefix)
    return store.load_projection(prefix, device)


def cached_runtime_tensor(
    root: Path,
    name: str,
    device: str,
    dtype: torch.dtype | None = torch.float32,
) -> torch.Tensor:
    return _store(root).runtime_tensor(name, device, dtype)


base.load_tensor = _cached_load_tensor
base.load_projection = _cached_load_projection
base.dequantize_fp8_blockwise = _cached_dequantize


@atexit.register
def _close_stores() -> None:
    for store in _stores.values():
        store.close()


def _format_mib(value: int) -> str:
    return f"{value / 1024**2:.1f} MiB"


def main() -> None:
    device = _requested_device()
    _configure_vram_limit(device)

    print(
        "Cache config: "
        f"ram={CACHE_GB:.2f} GiB | "
        f"vram={VRAM_CACHE_GB:.2f} GiB | "
        f"resident={RESIDENT_VRAM_BUDGET_BYTES / 1024**3:.2f} GiB | "
        f"hot_experts={EXPERT_VRAM_BUDGET_BYTES / 1024**3:.2f} GiB | "
        f"stream={STREAM_GB:.2f} GiB | "
        f"mode=resident-hot-stream | fp16_vram=on | "
        f"expert_bonus={EXPERT_BONUS:.2f}"
    )

    base.main()

    for root, store in _stores.items():
        ram = store.ram_cache.snapshot()
        vram = store.vram_cache.snapshot()
        print(
            "cached reader: "
            f"root={root} | shards opened={store.handle_opens} | cached handle hits={store.handle_hits}"
        )
        print(
            "RAM cache: "
            f"items={ram['items']} | ram={_format_mib(int(ram['bytes']))}/{_format_mib(store.ram_cache.max_bytes)} | "
            f"hits={ram['hits']} | misses={ram['misses']} | hit_rate={ram['hit_rate']:.2f}% | evictions={ram['evictions']}"
        )
        print(
            "VRAM cache: "
            f"items={vram['items']} | total={_format_mib(int(vram['bytes']))}/{_format_mib(store.vram_cache.max_bytes)} | "
            f"resident={_format_mib(int(vram['resident_bytes']))}/{_format_mib(RESIDENT_VRAM_BUDGET_BYTES)} | "
            f"experts={_format_mib(int(vram['expert_bytes']))}/{_format_mib(EXPERT_VRAM_BUDGET_BYTES)} | "
            f"stream={_format_mib(int(vram['stream_bytes']))}/{_format_mib(STREAM_BUDGET_BYTES)} | "
            f"hit_rate={vram['hit_rate']:.2f}% | expert_hit_rate={vram['expert_pool_hit_rate']:.2f}% | "
            f"stream_hit_rate={vram['stream_hit_rate']:.2f}% | expert_evictions={vram['expert_evictions']} | "
            f"stream_evictions={vram['stream_evictions']}"
        )


if __name__ == "__main__":
    main()
