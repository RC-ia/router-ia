from __future__ import annotations

"""Stable Q4 expert hierarchy used by the canonical runner.

The existing routing, adaptive heat, batch planner and scheduler stay intact.
Only the storage format of routed experts changes to packed Q4:

    checkpoint FP8 -> Q4 -> VRAM <-> RAM <-> temporary SSD
"""

import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from . import qwen36_adaptive_experts as adaptive
from . import qwen36_expert_batch_plan_v2 as planner_v2
from . import qwen36_expert_batch_plan_v3 as planner_v3
from . import qwen36_expert_cache as expert_cache
from . import qwen36_official_optimizations as official
from . import qwen36_cached_loop as cached
from . import qwen36_chat_batch as chat

Q4_VRAM_GB = max(float(os.getenv("QWEN36_Q4_VRAM_GB", "1.50")), 0.0)
Q4_RAM_GB = max(float(os.getenv("QWEN36_Q4_RAM_GB", "1.50")), 0.0)
Q4_RAM_SLOTS_PER_LAYER = max(int(os.getenv("QWEN36_Q4_RAM_SLOTS_PER_LAYER", "24")), 1)
Q4_SSD_DIRNAME = os.getenv("QWEN36_Q4_SSD_DIRNAME", ".router_q4_cache")
Q4_VRAM_BUDGET_BYTES = int(Q4_VRAM_GB * 1024**3)
Q4_RAM_BUDGET_BYTES = int(Q4_RAM_GB * 1024**3)
GENERIC_RESIDENT_GB = max(float(os.getenv("QWEN36_Q4_RESIDENT_GB", "0.75")), 0.0)
GENERIC_RESIDENT_BYTES = int(GENERIC_RESIDENT_GB * 1024**3)

_ORIGINAL_EXPERT_CACHE = official._expert_cache
_ORIGINAL_PREFETCH = expert_cache.RoutedExpertCache.prefetch_expert_raw
_ORIGINAL_SNAPSHOT = expert_cache.RoutedExpertCache.snapshot
_ORIGINAL_CLEAR = expert_cache.RoutedExpertCache.clear
_ORIGINAL_PRINT_CACHE = official._print_cache


def _configure_vram() -> None:
    total = int(cached.VRAM_CACHE_BUDGET_BYTES)
    resident = min(GENERIC_RESIDENT_BYTES, total)
    q4_vram = min(Q4_VRAM_BUDGET_BYTES, max(total - resident, 0))
    stream = max(total - resident - q4_vram, 0)
    cached.RESIDENT_VRAM_BUDGET_BYTES = resident
    cached.RESIDENT_VRAM_GB = resident / 1024**3
    cached.STREAM_BUDGET_BYTES = stream
    cached.STREAM_GB = stream / 1024**3
    globals()["Q4_VRAM_BUDGET_BYTES"] = q4_vram


_configure_vram()


def _ensure(cache: Any) -> Any:
    if getattr(cache, "_q4_hierarchy_ready", False):
        return cache
    cache._q4_hierarchy_ready = True
    cache._q4_vram_entries = OrderedDict()
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
    cache.q4_slots = Q4_RAM_SLOTS_PER_LAYER
    cache.budget_bytes = Q4_VRAM_BUDGET_BYTES
    return cache


def _size(entry: expert_cache.ColdEntry) -> int:
    return expert_cache.RoutedExpertCache._q4_size(entry)


def _ssd_path(root: Path, layer: int, expert_id: int) -> Path:
    directory = root / Q4_SSD_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"L{int(layer):02d}_E{int(expert_id):03d}.q4.pt"


def _save_ssd(root: Path, layer: int, expert_id: int, entry: expert_cache.ColdEntry) -> int:
    target = _ssd_path(root, layer, expert_id)
    tmp = target.with_name(target.name + ".tmp")
    torch.save(entry, tmp)
    tmp.replace(target)
    return int(target.stat().st_size)


def _load_ssd(root: Path, layer: int, expert_id: int) -> expert_cache.ColdEntry | None:
    target = _ssd_path(root, layer, expert_id)
    if not target.is_file():
        return None
    value = torch.load(target, map_location="cpu", weights_only=False)
    return tuple(
        (packed.contiguous(), scale.contiguous(), tuple(shape))
        for packed, scale, shape in value
    )  # type: ignore[return-value]


def _ram_insert(cache: Any, root: Path, layer: int, expert_id: int, entry: expert_cache.ColdEntry) -> None:
    layer, expert_id = int(layer), int(expert_id)
    bank = cache.q4_entries.setdefault(layer, OrderedDict())
    old = bank.pop(expert_id, None)
    if old is not None:
        cache._erase(layer, expert_id, "q4")
    bank[expert_id] = entry
    cache._record(layer, expert_id, "q4", _size(entry))

    while len(bank) > cache.q4_slots or cache.q4_bytes_used > cache._q4_ram_budget:
        victim_layer = layer
        if len(bank) <= cache.q4_slots:
            victim_layer = next(
                (candidate for candidate in range(cache.layers)
                 if cache.q4_entries.setdefault(candidate, OrderedDict())),
                layer,
            )
        victim_bank = cache.q4_entries[victim_layer]
        if not victim_bank:
            break
        victim_id, victim = victim_bank.popitem(last=False)
        cache._erase(victim_layer, victim_id, "q4")
        written = _save_ssd(root, victim_layer, victim_id, victim)
        cache._q4_ssd_writes += 1
        cache._q4_ssd_bytes_written += written
        cache._q4_ram_evictions_to_ssd += 1
        cache.q4_drops += 1
        cache.q4_ram_evictions += 1

    cache._q4_ram_bytes_peak = max(cache._q4_ram_bytes_peak, int(cache.q4_bytes_used))


def _to_cuda(entry: expert_cache.ColdEntry) -> expert_cache.ColdEntry:
    return tuple(
        (packed.to("cuda", non_blocking=True), scale.to("cuda", non_blocking=True), shape)
        for packed, scale, shape in entry
    )  # type: ignore[return-value]


def _vram_remove(cache: Any, root: Path, key: tuple[int, int]) -> None:
    entry = cache._q4_vram_entries.pop(key, None)
    if entry is None:
        return
    cache._q4_vram_bytes -= _size(entry)
    cache._q4_vram_evictions += 1
    layer, expert_id = key
    cpu_entry = tuple(
        (packed.detach().cpu(), scale.detach().cpu(), shape)
        for packed, scale, shape in entry
    )  # type: ignore[return-value]
    _ram_insert(cache, root, layer, expert_id, cpu_entry)


def _vram_insert(cache: Any, root: Path, layer: int, expert_id: int, entry: expert_cache.ColdEntry) -> expert_cache.ColdEntry:
    key = (int(layer), int(expert_id))
    old = cache._q4_vram_entries.pop(key, None)
    if old is not None:
        cache._q4_vram_bytes -= _size(old)
    cuda_entry = _to_cuda(entry)
    cache._q4_vram_entries[key] = cuda_entry
    cache._q4_vram_entries.move_to_end(key)
    cache._q4_vram_bytes += _size(cuda_entry)
    while cache._q4_vram_bytes > cache._q4_vram_budget and len(cache._q4_vram_entries) > 1:
        victim_key = next(iter(cache._q4_vram_entries))
        if victim_key == key:
            cache._q4_vram_entries.move_to_end(victim_key)
            victim_key = next(iter(cache._q4_vram_entries))
        _vram_remove(cache, root, victim_key)
    return cuda_entry


def _source_to_q4(store: Any, root: Path, layer: int, expert_id: int, layer_prefix: str) -> expert_cache.ColdEntry:
    prefix = f"{layer_prefix}mlp.experts.{int(expert_id)}"
    matrices = []
    for name in ("gate_proj", "up_proj", "down_proj"):
        weight = store._load_ssd(prefix + "." + name + ".weight")
        scale = store._load_ssd(prefix + "." + name + ".weight_scale_inv")
        if weight.dtype == torch.float8_e4m3fn:
            fp16 = expert_cache.dequant.dequantize_fp8_blockwise(weight, scale).to(torch.float16)
        else:
            fp16 = weight.to(torch.float16)
        matrices.append(expert_cache._q4_quantize_matrix(fp16.cpu()))
        del weight, scale, fp16
    return tuple(matrices)  # type: ignore[return-value]


def _get_or_load(cache: Any, store: Any, root: Path, layer: int, expert_id: int, layer_prefix: str):
    layer, expert_id = int(layer), int(expert_id)
    key = (layer, expert_id)
    with cache.lock:
        entry = cache._q4_vram_entries.get(key)
        if entry is not None:
            cache._q4_vram_entries.move_to_end(key)
            cache.hits += 1
            cache._q4_vram_hits += 1
            return entry, "vram"
        entry = cache.q4_entries.setdefault(layer, OrderedDict()).get(expert_id)
        if entry is not None:
            cache.q4_entries[layer].move_to_end(expert_id)
            cache.hits += 1
            cache._q4_ram_hits += 1
            return _vram_insert(cache, root, layer, expert_id, entry), "ram"
        entry = _load_ssd(root, layer, expert_id)
        if entry is not None:
            cache.hits += 1
            cache._q4_ssd_hits += 1
            cache._q4_ssd_reads += 1
            try:
                cache._q4_ssd_bytes_read += _ssd_path(root, layer, expert_id).stat().st_size
            except OSError:
                pass
            _ram_insert(cache, root, layer, expert_id, entry)
            return _vram_insert(cache, root, layer, expert_id, entry), "ssd"
        cache.misses += 1

    q4 = _source_to_q4(store, root, layer, expert_id, layer_prefix)
    with cache.lock:
        existing = cache._q4_vram_entries.get(key)
        if existing is not None:
            return existing, "vram"
        _ram_insert(cache, root, layer, expert_id, q4)
        cuda_entry = _vram_insert(cache, root, layer, expert_id, q4)
        cache.loads += 1
        cache.fp8_to_q4 += 1
        return cuda_entry, "source"


def _hierarchy_get_or_load_batch(self: Any, store: Any, layer: int, expert_ids: list[int], layer_prefix: str):
    cache = _ensure(self)
    root = store.root
    return [_get_or_load(cache, store, root, layer, int(expert_id), layer_prefix)[0] for expert_id in expert_ids]


def _hierarchy_get_or_load(self: Any, store: Any, layer: int, expert_id: int, layer_prefix: str):
    return _hierarchy_get_or_load_batch(self, store, layer, [expert_id], layer_prefix)[0]


def _hierarchy_prefetch(self: Any, store: Any, layer_prefix: str, expert_id: int) -> None:
    marker = ".layers."
    if marker not in layer_prefix:
        return _ORIGINAL_PREFETCH(self, store, layer_prefix, expert_id)
    layer = int(layer_prefix.split(marker, 1)[1].split(".", 1)[0])
    _get_or_load(_ensure(self), store, store.root, layer, int(expert_id), layer_prefix)


def _plan_layer_q4(root: Path, layer: int, layer_prefix: str, expert_ids: list[int]):
    ids = [int(x) for x in expert_ids]
    unique = list(dict.fromkeys(ids))
    cache = _ensure(_ORIGINAL_EXPERT_CACHE(root))
    store = cached._store(root)
    policy = adaptive._policy(root)
    entries = {}
    tiers = {}
    for expert_id in unique:
        entry, tier = _get_or_load(cache, store, root, int(layer), expert_id, layer_prefix)
        entries[expert_id] = entry
        tiers[expert_id] = tier
        policy.record(layer, expert_id, "q4")
    decoded = planner_v2._decode_q4([(expert_id, "q4", entries[expert_id]) for expert_id in unique])
    planner_v2._stat("q4_hits", sum(t in {"vram", "ram", "ssd"} for t in tiers.values()))
    planner_v2._stat("fp8_hits", 0)
    planner_v2._stat("miss_experts", sum(t == "source" for t in tiers.values()))
    planner_v2._stat("plans")
    planner_v2._stat("unique_experts", len(unique))
    return [decoded[expert_id] for expert_id in ids]


def _snapshot(self: Any) -> dict[str, int | float]:
    base = dict(_ORIGINAL_SNAPSHOT(self))
    cache = _ensure(self)
    ram_items = sum(len(bank) for bank in cache.q4_entries.values())
    base.update({
        "bytes": int(cache._q4_vram_bytes),
        "budget_bytes": int(cache._q4_vram_budget),
        "warm_items": len(cache._q4_vram_entries),
        "cold_items": ram_items,
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
        "q4_vram_evictions": int(cache._q4_vram_evictions),
        "q4_ram_evictions_to_ssd": int(cache._q4_ram_evictions_to_ssd),
    })
    return base


def _print_cache(root: Path, label: str) -> None:
    _ORIGINAL_PRINT_CACHE(root, label)
    cache = official._EXPERT_CACHES.get(root.resolve())
    if cache is None:
        return
    cache = _ensure(cache)
    print(
        "  q4 hierarchy: "
        f"vram={cache._q4_vram_bytes / 1024**2:.1f}/{cache._q4_vram_budget / 1024**2:.1f} MiB | "
        f"ram={cache.q4_bytes_used / 1024**2:.1f}/{cache._q4_ram_budget / 1024**2:.1f} MiB | "
        f"vram_items={len(cache._q4_vram_entries)} | "
        f"ram_items={sum(len(bank) for bank in cache.q4_entries.values())} | "
        f"ssd_hits={cache._q4_ssd_hits} | ssd_writes={cache._q4_ssd_writes} | "
        f"ssd_reads={cache._q4_ssd_reads} | vram_evictions={cache._q4_vram_evictions}"
    )


def _clear(self: Any) -> None:
    cache = _ensure(self)
    with cache.lock:
        cache._q4_vram_entries.clear()
        cache._q4_vram_bytes = 0
    _ORIGINAL_CLEAR(self)


def _cache_factory(root: Path):
    return _ensure(_ORIGINAL_EXPERT_CACHE(root))


expert_cache.RoutedExpertCache.get_or_load_batch = _hierarchy_get_or_load_batch
expert_cache.RoutedExpertCache.get_or_load = _hierarchy_get_or_load
expert_cache.RoutedExpertCache.prefetch_expert_raw = _hierarchy_prefetch
expert_cache.RoutedExpertCache.snapshot = _snapshot
expert_cache.RoutedExpertCache.clear = _clear
planner_v3._expert_cache = _cache_factory
planner_v3._plan_layer = _plan_layer_q4
official._expert_cache = _cache_factory
chat.print_cache = _print_cache

print(
    "expert_q4_hierarchy=enabled|representation=Q4|"
    f"vram={Q4_VRAM_BUDGET_BYTES / 1024**3:.2f}GiB|"
    f"ram={Q4_RAM_GB:.2f}GiB|ram_slots_per_layer={Q4_RAM_SLOTS_PER_LAYER}|"
    f"generic_resident={cached.RESIDENT_VRAM_GB:.2f}GiB|"
    f"generic_stream={cached.STREAM_GB:.2f}GiB|ssd=temporary|stable=planner-v2-decode"
)
