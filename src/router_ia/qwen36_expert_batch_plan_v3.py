from __future__ import annotations

"""Hot-path refinements for the Qwen3.6 batch planner.

Keeps v2's batched execution intact while removing two sources of avoidable
latency:
- routed expert raw tensors bypass the generic RAM tensor cache;
- Q4->FP8 promotion is never performed synchronously inside a token.
"""

from pathlib import Path

import torch

from . import qwen36_adaptive_experts as adaptive
from . import qwen36_cached_loop as cached
from . import qwen36_expert_batch_plan_v2 as planner


_ORIGINAL_LOAD_MISSING = planner._load_missing_grouped


def _load_missing_grouped_direct(
    root: Path,
    layer: int,
    layer_prefix: str,
    missing: list[int],
):
    """Load routed expert raw tensors directly from safetensor shards.

    Expert tensors are owned by the routed-expert cache; placing their raw FP8
    copies into the generic RAM cache only duplicates memory and competes with
    the Q4 expert bank.
    """
    if not missing:
        return {}

    store = cached._store(root)
    ids = list(dict.fromkeys(int(x) for x in missing))
    names = ("gate_proj", "up_proj", "down_proj")
    weights_cpu = {name: [] for name in names}
    scales_cpu = {name: [] for name in names}

    for expert_id in ids:
        expert_prefix = f"{layer_prefix}mlp.experts.{expert_id}"
        for name in names:
            prefix = f"{expert_prefix}.{name}"
            # Bypass _ShardStore.load(): routed expert storage should not also
            # occupy the generic RAM tensor cache.
            weights_cpu[name].append(store._load_ssd(prefix + ".weight"))
            scales_cpu[name].append(store._load_ssd(prefix + ".weight_scale_inv"))

    # Stay out of the GPU path entirely for non-FP8 checkpoints. The previous
    # implementation allocated H2D batches first and only then discovered that
    # it had to fall back to v2, which wasted VRAM and performed unnecessary work.
    if any(
        weight.dtype != torch.float8_e4m3fn
        for name in names
        for weight in weights_cpu[name]
    ):
        return _ORIGINAL_LOAD_MISSING(root, layer, layer_prefix, ids)

    gpu = {}
    for name in names:
        wb = torch.stack(weights_cpu[name], dim=0)
        sb = torch.stack(scales_cpu[name], dim=0)
        gpu[name] = (
            wb.to(device="cuda", non_blocking=True),
            sb.to(device="cuda", non_blocking=True),
        )
        planner._stat("weight_batches")
        planner._stat("scale_batches")
        planner._stat("cpu_projection_loads", len(ids))

    compact = {}
    for local, expert_id in enumerate(ids):
        compact[expert_id] = (
            (gpu["gate_proj"][0][local], gpu["gate_proj"][1][local]),
            (gpu["up_proj"][0][local], gpu["up_proj"][1][local]),
            (gpu["down_proj"][0][local], gpu["down_proj"][1][local]),
        )

    cache = planner._expert_cache(root)
    with cache.lock:
        for expert_id, entry in compact.items():
            cache._insert_fp8_locked(int(layer), int(expert_id), entry)
            cache.loads += 1

    return planner._decode_fp8(
        [(expert_id, "fp8", compact[expert_id]) for expert_id in ids]
    )


def _plan_layer_no_sync_promotion(
    root: Path,
    layer: int,
    layer_prefix: str,
    expert_ids: list[int],
):
    """Reuse v2 planning while making Q4 promotion non-blocking by omission."""
    ids = [int(x) for x in expert_ids]
    unique_ids = list(dict.fromkeys(ids))
    cache = planner._expert_cache(root)
    policy = adaptive._policy(root)
    output = {}
    fp8_found = []
    q4_found = []
    missing = []

    fp16_enabled = planner.fp16_cache_module.FP16_CACHE_BYTES > 0
    materialized = planner.fp16_cache_module._cache(root) if fp16_enabled else None

    for expert_id in unique_ids:
        if materialized is not None:
            hit = materialized.get((int(layer), expert_id))
            if hit is not None:
                output[expert_id] = hit
                planner._stat("fp16_hits")
                policy.record(layer, expert_id, "fp16")
                continue

        with cache.lock:
            fp8 = cache.fp8_entries.setdefault(int(layer), {}).get(expert_id)
            if fp8 is not None:
                cache.hits += 1
                cache.fp8_hits += 1
                cache.fp8_entries[int(layer)].move_to_end(expert_id)
                fp8_found.append((expert_id, "fp8", fp8))
                policy.record(layer, expert_id, "fp8")
                continue
            q4 = cache.q4_entries.setdefault(int(layer), {}).get(expert_id)
            if q4 is not None:
                cache.hits += 1
                cache.q4_hits += 1
                cache.q4_entries[int(layer)].move_to_end(expert_id)
                q4_found.append((expert_id, "q4", q4))
                policy.record(layer, expert_id, "q4")
                continue
            cache.misses += 1
        policy.record(layer, expert_id, "miss")
        missing.append(expert_id)

    output.update(planner._decode_fp8(fp8_found))
    planner._stat("fp8_hits", len(fp8_found))
    output.update(planner._decode_q4(q4_found))
    planner._stat("q4_hits", len(q4_found))

    # Intentionally do not call cache.put_fp16() here. That path requantizes a
    # fully dequantized FP16 triplet back into FP8 while the token is waiting.
    # Heat is still recorded above and can guide future retention/prefetch work.
    if missing:
        planner._stat("miss_experts", len(missing))
        output.update(_load_missing_grouped_direct(root, layer, layer_prefix, missing))

    if materialized is not None:
        for expert_id in unique_ids:
            materialized.put((int(layer), expert_id), output[expert_id])

    planner._stat("plans")
    planner._stat("unique_experts", len(unique_ids))
    return [output[expert_id] for expert_id in ids]


planner._load_missing_grouped = _load_missing_grouped_direct
planner._plan_layer = _plan_layer_no_sync_promotion

print(
    "expert_batch_plan_v3=enabled|raw-expert-cache=bypass|"
    "q4-promotion=synchronous-disabled|inherits=v2-batched-gemm"
)
