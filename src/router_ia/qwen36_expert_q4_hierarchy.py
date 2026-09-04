from __future__ import annotations

"""Q4 hierarchy for routed experts: VRAM -> RAM -> temporary SSD backing.

The existing router policies remain intact. This module changes only the
physical representation of routed experts:

    source checkpoint FP8 -> Q4 RAM/VRAM -> temporary Q4 SSD backing

Q4 is the only persistent expert representation after first materialization.
The VRAM tier owns packed Q4 buffers; CUDA dequantization happens only for the
short-lived tensors needed by the MoE GEMM.
"""

import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from . import qwen36_adaptive_experts as adaptive
from . import qwen36_expert_batch_plan_v3 as planner
from . import qwen36_expert_cache as expert_cache
from . import qwen36_official_optimizations as official
from . import qwen36_cached_loop as cached

MODEL_LAYERS = expert_cache.MODEL_LAYERS
EXPERTS_PER_LAYER = expert_cache.EXPERTS_PER_LAYER

Q4_VRAM_GB = max(float(os.getenv("QWEN36_Q4_VRAM_GB", "1.5")), 0.0)
Q4_VRAM_BUDGET_BYTES = int(Q4_VRAM_GB * 1024**3)
Q4_RAM_GB = max(float(os.getenv("QWEN36_Q4_RAM_GB", "2.0")), 0.0)
Q4_RAM_BUDGET_BYTES = int(Q4_RAM_GB * 1024**3)
Q4_RAM_SLOTS_PER_LAYER = max(int(os.getenv("QWEN36_Q4_RAM_SLOTS_PER_LAYER", "24")), 1)
Q4_SSD_DIRNAME = os.getenv("QWEN36_Q4_SSD_DIRNAME", ".router_q4_cache")

_Q4Matrix = expert_cache.Q4Matrix
_ColdEntry = expert_cache.ColdEntry

_ORIGINAL_EXPERT_CACHE = planner._expert_cache
_ORIGINAL_PREFETCH = getattr(expert_cache.RoutedExpertCache, "prefetch_expert_raw")
_ORIGINAL_SNAPSHOT = expert_cache.RoutedExpertCache.snapshot
_ORIGINAL_CLEAR = expert_cache.RoutedExpertCache.clear


def _entry_size(entry: _ColdEntry) -> int:
    return expert_cache.RoutedExpertCache._q4_size(entry)


def _ssd_dir(root: Path) -> Path:
    path = root / Q4_SSD_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ssd_path(root: Path, layer: int, expert_id: int) -> Path:
    return _ssd_dir(root) / f"L{int(layer):02d}_E{int(expert_id):03d}.q4.pt"


def _save_ssd(root: Path, layer: int, expert_id: int, entry: _ColdEntry) -> int:
    target = _ssd_path(root, layer, expert_id)
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(entry, tmp)
    tmp.replace(target)
    return int(target.stat().st_size)


def _load_ssd(root: Path, layer: int, expert_id: int) -> _ColdEntry | None:
    target = _ssd_path(root, layer, expert_id)
    if not target.is_file():
        return None
    entry = torch.load(target, map_location="cpu", weights_only=False)
    return tuple(
        (packed.contiguous(), scale.contiguous(), tuple(shape))
        for packed, scale, shape in entry
    )  # type: ignore[return-value]


def _ensure(cache: Any) -> Any:
    if getattr(cache, "_q4_hierarchy_ready", False):
        return cache

    cache._q4_hierarchy_ready = True
    cache._q4_vram_entries = OrderedDict()  # (layer, expert) -> ColdEntry on CUDA
    cache._q4_vram_bytes = 0
    cache._q4_vram_budget = Q4_VRAM_BUDGET_BYTES
    cache._q4_ram_budget = Q4_RAM_BUDGET_BYTES
    cache._q4_ram_bytes_peak = 0
    cache._q4_ssd_writes = 0
    cache._q4_ssd_reads = 0
    cache._q4_ssd_bytes_written = 0
    cache._q4_ssd_bytes_read = 0
    cache._q4_vram_hits = 0
    cache._q4_ram_hits = 0
    cache._q4_ssd_hits = 0
    cache._q4_source_quantizations = 0
    cache._q4_vram_evictions = 0
    cache._q4_ram_evictions_to_ssd = 0

    # The original Q4 RAM bank is deliberately enlarged. Its existing
    # per-layer accounting is retained so the adaptive/LRU logic keeps working.
    cache.q4_slots = max(int(getattr(cache, "q4_slots", 0)), Q4_RAM_SLOTS_PER_LAYER)
    cache.budget_bytes = Q4_VRAM_BUDGET_BYTES
    return cache


def _ram_insert(cache: Any, root: Path, layer: int, expert_id: int, entry: _ColdEntry) -> None:
    layer = int(layer)
    expert_id = int(expert_id)
    bank = cache.q4_entries.setdefault(layer, OrderedDict())
    old = bank.pop(expert_id, None)
    if old is not None:
        cache._erase(layer, expert_id, "q4")
    bank[expert_id] = entry
    cache._record(layer, expert_id, "q4", _entry_size(entry))
    bank.move_to_end(expert_id)

    # Enforce both the enlarged per-layer window and a real global RAM budget.
    while len(bank) > cache.q4_slots:
        victim_id, victim = bank.popitem(last=False)
        cache._erase(layer, victim_id, "q4")
        written = _save_ssd(root, layer, victim_id, victim)
        cache._q4_ssd_writes += 1
        cache._q4_ssd_bytes_written += written
        cache._q4_ram_evictions_to_ssd += 1
        cache.q4_drops += 1
        cache.q4_ram_evictions += 1

    while cache.q4_bytes_used > cache._q4_ram_budget:
        victim_layer = None
        victim_id = None
        for scan_layer in range(cache.layers):
            candidate = cache.q4_entries.setdefault(scan_layer, OrderedDict())
            if candidate:
                victim_layer = scan_layer
                victim_id = next(iter(candidate))
                break
        if victim_layer is None or victim_id is None:
            break
        candidate = cache.q4_entries[victim_layer]
        victim = candidate.pop(victim_id)
        cache._erase(victim_layer, victim_id, "q4")
        written = _save_ssd(root, victim_layer, victim_id, victim)
        cache._q4_ssd_writes += 1
        cache._q4_ssd_bytes_written += written
        cache._q4_ram_evictions_to_ssd += 1
        cache.q4_drops += 1
        cache.q4_ram_evictions += 1

    cache._q4_ram_bytes_peak = max(cache._q4_ram_bytes_peak, int(cache.q4_bytes_used))


def _to_cuda_entry(entry: _ColdEntry) -> _ColdEntry:
    return tuple(
        (
            packed.to(device="cuda", non_blocking=True),
            scale.to(device="cuda", non_blocking=True),
            shape,
        )
        for packed, scale, shape in entry
    )  # type: ignore[return-value]


def _vram_remove(cache: Any, key: tuple[int, int], *, demote_to_ram: bool, root: Path) -> _ColdEntry | None:
    entry = cache._q4_vram_entries.pop(key, None)
    if entry is None:
        return None
    cache._q4_vram_bytes -= _entry_size(entry)
    cache._q4_vram_evictions += 1
    layer, expert_id = key
    if demote_to_ram:
        cpu_entry = tuple(
            (packed.detach().cpu(), scale.detach().cpu(), shape)
            for packed, scale, shape in entry
        )  # type: ignore[return-value]
        _ram_insert(cache, root, layer, expert_id, cpu_entry)
        return cpu_entry
    return entry


def _vram_insert(cache: Any, root: Path, layer: int, expert_id: int, entry: _ColdEntry) -> _ColdEntry:
    key = (int(layer), int(expert_id))
    existing = cache._q4_vram_entries.pop(key, None)
    if existing is not None:
        cache._q4_vram_bytes -= _entry_size(existing)
    cuda_entry = _to_cuda_entry(entry)
    cache._q4_vram_entries[key] = cuda_entry
    cache._q4_vram_entries.move_to_end(key)
    cache._q4_vram_bytes += _entry_size(cuda_entry)

    while cache._q4_vram_bytes > cache._q4_vram_budget and cache._q4_vram_entries:
        victim_key = next(iter(cache._q4_vram_entries))
        if victim_key == key and len(cache._q4_vram_entries) > 1:
            cache._q4_vram_entries.move_to_end(victim_key)
            victim_key = next(iter(cache._q4_vram_entries))
        if victim_key == key and len(cache._q4_vram_entries) == 1:
            break
        _vram_remove(cache, victim_key, demote_to_ram=True, root=root)

    return cuda_entry


def _source_to_q4(store: Any, root: Path, layer: int, expert_id: int, layer_prefix: str) -> _ColdEntry:
    expert_prefix = f"{layer_prefix}mlp.experts.{int(expert_id)}"
    matrices = []
    for name in ("gate_proj", "up_proj", "down_proj"):
        weight = store._load_ssd(expert_prefix + "." + name + ".weight")
        scale = store._load_ssd(expert_prefix + "." + name + ".weight_scale_inv")
        if weight.dtype == torch.float8_e4m3fn:
            weight_fp16 = expert_cache.dequant.dequantize_fp8_blockwise(weight, scale).to(torch.float16)
        else:
            weight_fp16 = weight.to(torch.float16)
        matrices.append(expert_cache._q4_quantize_matrix(weight_fp16.cpu()))
        del weight, scale, weight_fp16
    entry: _ColdEntry = tuple(matrices)  # type: ignore[assignment]
    cache = _ensure(_ORIGINAL_EXPERT_CACHE(root))
    cache._q4_source_quantizations += 1
    return entry


def _get_or_load_one(cache: Any, store: Any, root: Path, layer: int, expert_id: int, layer_prefix: str):
    layer = int(layer)
    expert_id = int(expert_id)
    key = (layer, expert_id)
    vram = cache._q4_vram_entries.get(key)
    if vram is not None:
        cache._q4_vram_entries.move_to_end(key)
        cache.hits += 1
        cache.fp8_hits += 1
        cache._q4_vram_hits += 1
        return vram, "vram"

    ram = cache.q4_entries.setdefault(layer, OrderedDict()).get(expert_id)
    if ram is not None:
        cache.q4_entries[layer].move_to_end(expert_id)
        cache.hits += 1
        cache.q4_hits += 1
        cache._q4_ram_hits += 1
        cuda_entry = _vram_insert(cache, root, layer, expert_id, ram)
        return cuda_entry, "ram"

    ssd_entry = _load_ssd(root, layer, expert_id)
    if ssd_entry is not None:
        cache.hits += 1
        cache.q4_hits += 1
        cache._q4_ssd_hits += 1
        cache._q4_ssd_reads += 1
        try:
            cache._q4_ssd_bytes_read += _ssd_path(root, layer, expert_id).stat().st_size
        except OSError:
            pass
        _ram_insert(cache, root, layer, expert_id, ssd_entry)
        cuda_entry = _vram_insert(cache, root, layer, expert_id, ssd_entry)
        return cuda_entry, "ssd"

    cache.misses += 1
    q4 = _source_to_q4(store, root, layer, expert_id, layer_prefix)
    _ram_insert(cache, root, layer, expert_id, q4)
    cuda_entry = _vram_insert(cache, root, layer, expert_id, q4)
    cache.loads += 1
    cache.fp8_to_q4 += 1
    return cuda_entry, "source"


def _hierarchy_get_or_load_batch(self: Any, store: Any, layer: int, expert_ids: list[int], layer_prefix: str):
    cache = _ensure(self)
    root = store.root
    result: list[_ColdEntry] = []
    tiers: list[str] = []
    for expert_id in expert_ids:
        entry, tier = _get_or_load_one(cache, store, root, int(layer), int(expert_id), layer_prefix)
        result.append(entry)
        tiers.append(tier)
    # Record a separate stat for SSD/source misses without changing existing
    # adaptive tier names used by the rest of the router.
    return result


def _hierarchy_get_or_load(self: Any, store: Any, layer: int, expert_id: int, layer_prefix: str):
    return _hierarchy_get_or_load_batch(self, store, layer, [expert_id], layer_prefix)[0]


def _hierarchy_prefetch(self: Any, store: Any, layer_prefix: str, expert_id: int) -> None:
    cache = _ensure(self)
    root = store.root
    layer_marker = ".layers."
    if layer_marker not in layer_prefix:
        return _ORIGINAL_PREFETCH(self, store, layer_prefix, expert_id)
    layer = int(layer_prefix.split(layer_marker, 1)[1].split(".", 1)[0])
    _get_or_load_one(cache, store, root, layer, int(expert_id), layer_prefix)


def _plan_layer_q4(root: Path, layer: int, layer_prefix: str, expert_ids: list[int]):
    ids = [int(x) for x in expert_ids]
    unique_ids = list(dict.fromkeys(ids))
    cache = _ensure(_ORIGINAL_EXPERT_CACHE(root))
    policy = adaptive._policy(root)
    store = cached._store(root)

    entries: dict[int, _ColdEntry] = {}
    tiers: dict[int, str] = {}
    for expert_id in unique_ids:
        entry, tier = _get_or_load_one(cache, store, root, int(layer), expert_id, layer_prefix)
        entries[expert_id] = entry
        tiers[expert_id] = tier
        policy.record(layer, expert_id, "fp8" if tier == "vram" else "q4")

    # Dequantization remains batched and GPU-side via the already-installed
    # qwen36_gpu_q4 implementation.
    packed = [(expert_id, "q4", entries[expert_id]) for expert_id in unique_ids]
    output = planner._decode_q4(packed)
    planner._stat("q4_hits", sum(tier in {"vram", "ram", "ssd"} for tier in tiers.values()))
    planner._stat("fp8_hits", sum(tier == "vram" for tier in tiers.values()))
    planner._stat("miss_experts", sum(tier == "source" for tier in tiers.values()))
    planner._stat("plans")
    planner._stat("unique_experts", len(unique_ids))
    return [output[expert_id] for expert_id in ids]


def _snapshot(self: Any) -> dict[str, int | float]:
    snap = dict(_ORIGINAL_SNAPSHOT(self))
    cache = _ensure(self)
    snap.update(
        {
            "bytes": int(cache._q4_vram_bytes),
            "budget_bytes": int(cache._q4_vram_budget),
            "warm_items": len(cache._q4_vram_entries),
            "cold_items": sum(len(bank) for bank in cache.q4_entries.values()),
            "q4_ram_bytes": int(cache.q4_bytes_used),
            "q4_vram_bytes": int(cache._q4_vram_bytes),
            "q4_vram_budget": int(cache._q4_vram_budget),
            "q4_ram_budget": int(cache._q4_ram_budget),
            "fp8_hits": int(cache._q4_vram_hits),
            "q4_hits": int(cache._q4_ram_hits + cache._q4_ssd_hits),
            "fp8_to_q4": int(cache._q4_source_quantizations),
            "q4_ssd_hits": int(cache._q4_ssd_hits),
            "q4_ssd_writes": int(cache._q4_ssd_writes),
            "q4_ssd_reads": int(cache._q4_ssd_reads),
            "q4_ssd_bytes_written": int(cache._q4_ssd_bytes_written),
            "q4_ssd_bytes_read": int(cache._q4_ssd_bytes_read),
            "q4_vram_evictions": int(cache._q4_vram_evictions),
            "q4_ram_evictions_to_ssd": int(cache._q4_ram_evictions_to_ssd),
            "shared_items": sum(len(bank) for bank in cache.q4_entries.values()),
            "protected_items": len(cache._q4_vram_entries),
            "warm_slots_per_layer": max(len(cache._q4_vram_entries) // max(cache.layers, 1), 0),
            "cold_slots_per_layer": int(cache.q4_slots),
        }
    )
    return snap


def _clear(self: Any) -> None:
    cache = _ensure(self)
    for entry in list(cache._q4_vram_entries.values()):
        del entry
    cache._q4_vram_entries.clear()
    cache._q4_vram_bytes = 0
    _ORIGINAL_CLEAR(self)


def _patched_expert_cache(root: Path):
    return _ensure(_ORIGINAL_EXPERT_CACHE(root))


# Install hierarchy methods without disturbing the existing policy modules.
expert_cache.RoutedExpertCache.get_or_load_batch = _hierarchy_get_or_load_batch
expert_cache.RoutedExpertCache.get_or_load = _hierarchy_get_or_load
expert_cache.RoutedExpertCache.prefetch_expert_raw = _hierarchy_prefetch
expert_cache.RoutedExpertCache.snapshot = _snapshot
expert_cache.RoutedExpertCache.clear = _clear
planner._expert_cache = _patched_expert_cache
official._expert_cache = _patched_expert_cache
planner._plan_layer = _plan_layer_q4

print(
    "expert_q4_hierarchy=enabled|representation=Q4|"
    f"vram={Q4_VRAM_GB:.2f}GiB|ram={Q4_RAM_GB:.2f}GiB|"
    f"ram_slots_per_layer={Q4_RAM_SLOTS_PER_LAYER}|ssd=temporary"
)
