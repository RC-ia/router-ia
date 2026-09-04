from __future__ import annotations

"""Batch Q4 hierarchy loader.

The Q4 planner used to resolve each routed expert independently. When an expert
was cold, that meant a separate Python/torch.load path for each expert. This
module replaces the layer hot path with one tier-resolution pass and concurrent
SSD reads for the missing Q4 experts.

Only storage/loading behavior is changed. Q4 representation, VRAM/RAM budgets,
CUDA dequantization and compute math remain unchanged.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from . import qwen36_adaptive_experts as adaptive
from . import qwen36_cached_loop as cached
from . import qwen36_expert_batch_plan_v2 as planner_v2
from . import qwen36_expert_batch_plan_v3 as planner_v3
from . import qwen36_expert_q4_hierarchy_fixed as hierarchy
from . import qwen36_official_optimizations as official

SSD_WORKERS = max(1, int(os.getenv("QWEN36_Q4_SSD_LOAD_WORKERS", "4")))

_ORIGINAL_PLAN = hierarchy._plan_layer_q4
_ORIGINAL_EXPERT_CACHE = official._expert_cache


def _ensure(cache: Any) -> Any:
    return hierarchy._ensure(cache)


def _ssd_load_one(args):
    root, layer, expert_id = args
    entry = hierarchy._load_ssd(root, layer, expert_id)
    return int(expert_id), entry


def _record_ssd_hit(cache: Any, root: Path, layer: int, expert_id: int) -> None:
    with cache.lock:
        cache.hits += 1
        cache._q4_ssd_hits += 1
        cache._q4_ssd_reads += 1
        try:
            cache._q4_ssd_bytes_read += hierarchy._ssd_path(root, layer, expert_id).stat().st_size
        except OSError:
            pass


def _plan_layer_batch(root: Path, layer: int, layer_prefix: str, expert_ids: list[int]):
    ids = [int(x) for x in expert_ids]
    unique = list(dict.fromkeys(ids))
    cache = _ensure(_ORIGINAL_EXPERT_CACHE(root))
    store = cached._store(root)
    policy = adaptive._policy(root)

    entries: dict[int, Any] = {}
    tiers: dict[int, str] = {}
    ram_pending: list[tuple[int, Any]] = []
    ssd_pending: list[int] = []

    # One metadata pass for the whole layer. No disk IO under the cache lock.
    with cache.lock:
        bank = cache.q4_entries.setdefault(int(layer), {})
        for expert_id in unique:
            key = (int(layer), int(expert_id))
            entry = cache._q4_vram_entries.get(key)
            if entry is not None:
                cache._q4_vram_entries.move_to_end(key)
                cache.hits += 1
                cache._q4_vram_hits += 1
                entries[expert_id] = entry
                tiers[expert_id] = "vram"
                continue

            ram_entry = bank.get(expert_id)
            if ram_entry is not None:
                bank.move_to_end(expert_id)
                cache.hits += 1
                cache._q4_ram_hits += 1
                ram_pending.append((expert_id, ram_entry))
                tiers[expert_id] = "ram"
                continue

            cache.misses += 1
            ssd_pending.append(expert_id)
            tiers[expert_id] = "missing"

    # Promote RAM hits without holding the metadata lock over H2D.
    for expert_id, ram_entry in ram_pending:
        entries[expert_id] = hierarchy._vram_insert(
            cache, root, int(layer), int(expert_id), ram_entry
        )

    # Cold Q4 files are independent. Read several in parallel so storage latency
    # is overlapped instead of paying one torch.load wall-time after another.
    ssd_found: list[tuple[int, Any]] = []
    if ssd_pending:
        jobs = [(root, int(layer), int(expert_id)) for expert_id in ssd_pending]
        workers = min(SSD_WORKERS, len(jobs))
        if workers == 1:
            results = [_ssd_load_one(job) for job in jobs]
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="q4-ssd") as pool:
                results = list(pool.map(_ssd_load_one, jobs))
        for expert_id, entry in results:
            if entry is None:
                continue
            _record_ssd_hit(cache, root, int(layer), int(expert_id))
            ssd_found.append((expert_id, entry))

    for expert_id, ssd_entry in ssd_found:
        hierarchy._ram_insert(cache, root, int(layer), int(expert_id), ssd_entry)
        entries[expert_id] = hierarchy._vram_insert(
            cache, root, int(layer), int(expert_id), ssd_entry
        )
        tiers[expert_id] = "ssd"

    # Genuine first-seen experts still use the canonical FP8 -> Q4 path.
    source_missing = [expert_id for expert_id in ssd_pending if expert_id not in entries]
    for expert_id in source_missing:
        q4 = hierarchy._source_to_q4(
            store, root, int(layer), int(expert_id), layer_prefix
        )
        hierarchy._ram_insert(cache, root, int(layer), int(expert_id), q4)
        entries[expert_id] = hierarchy._vram_insert(
            cache, root, int(layer), int(expert_id), q4
        )
        tiers[expert_id] = "source"
        with cache.lock:
            cache.loads += 1
            cache._q4_source_quantizations += 1
            cache.fp8_to_q4 += 1

    for expert_id in unique:
        policy.record(int(layer), int(expert_id), "q4")

    decoded = planner_v2._decode_q4(
        [(expert_id, "q4", entries[expert_id]) for expert_id in unique]
    )
    planner_v2._stat("q4_hits", sum(t in {"vram", "ram", "ssd"} for t in tiers.values()))
    planner_v2._stat("fp8_hits", 0)
    planner_v2._stat("miss_experts", sum(t == "source" for t in tiers.values()))
    planner_v2._stat("plans")
    planner_v2._stat("unique_experts", len(unique))
    return [decoded[expert_id] for expert_id in ids]


hierarchy._plan_layer_q4 = _plan_layer_batch
planner_v2._plan_layer = _plan_layer_batch
planner_v3._plan_layer = _plan_layer_batch

print(
    f"q4_batch_loader=enabled|ssd_workers={SSD_WORKERS}|"
    "tier_resolution=single-pass|ssd_reads=parallel|source_fallback=canonical"
)
