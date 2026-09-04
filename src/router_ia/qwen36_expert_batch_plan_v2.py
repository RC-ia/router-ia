from __future__ import annotations

"""Batch-first routed-expert planner for Qwen3.6.

Hot path:
    router -> complete route for this layer -> one planner -> grouped H2D -> batched GPU MoE

No per-expert ThreadPool is used for the routed experts. The planner operates
one layer at a time so it does not retain all 40 layers of FP16 materialized
weights simultaneously.
"""

from pathlib import Path
from threading import Lock

import torch
import torch.nn.functional as F

from . import qwen36_cached_loop as cached
from . import qwen36_chat_batch as chat
from . import qwen36_dequant as dequant
from . import qwen36_40layer_loop as base
from . import qwen36_expert_cache as expert_cache_module
from . import qwen36_fp16_expert_cache as fp16_cache_module
from . import qwen36_official_optimizations as official


_ORIGINAL_BATCHED_MOE_STEP = chat.batched_moe_step
_ORIGINAL_CACHE_STATS = chat.cache_stats
_STATS_LOCK = Lock()
_STATS = {
    "plans": 0,
    "unique_experts": 0,
    "fp16_hits": 0,
    "fp8_hits": 0,
    "q4_hits": 0,
    "miss_experts": 0,
    "weight_batches": 0,
    "scale_batches": 0,
    "cpu_projection_loads": 0,
}


def _stat(name: str, amount: int = 1) -> None:
    with _STATS_LOCK:
        _STATS[name] += int(amount)


def _expert_cache(root: Path):
    key = root.resolve()
    cache = official._EXPERT_CACHES.get(key)
    if cache is None:
        cache = expert_cache_module.RoutedExpertCache(official.EXPERT_VRAM_BUDGET_BYTES)
        official._EXPERT_CACHES[key] = cache
    return cache


def _decode_fp8(entries) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    output: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    if not entries:
        return output
    per_projection: list[list[torch.Tensor]] = []
    for projection in range(3):
        weights = torch.stack([entry[2][projection][0] for entry in entries], dim=0)
        scales = torch.stack([entry[2][projection][1] for entry in entries], dim=0)
        decoded = dequant.dequantize_fp8_blockwise_batch(weights, scales).to(dtype=torch.float16)
        per_projection.append(list(decoded.unbind(0)))
    for local, (expert_id, _tier, _entry) in enumerate(entries):
        output[expert_id] = (
            per_projection[0][local],
            per_projection[1][local],
            per_projection[2][local],
        )
    return output


def _decode_q4(entries) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    output: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    if not entries:
        return output
    per_projection: list[list[torch.Tensor]] = []
    for projection in range(3):
        per_projection.append(
            expert_cache_module._q4_dequantize_entry_batch(
                [entry[2] for entry in entries], projection
            )
        )
    for local, (expert_id, _tier, _entry) in enumerate(entries):
        output[expert_id] = (
            per_projection[0][local],
            per_projection[1][local],
            per_projection[2][local],
        )
    return output


def _load_missing_grouped(
    root: Path,
    layer: int,
    layer_prefix: str,
    missing: list[int],
) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Read raw CPU tensors, then perform grouped H2D transfers per projection."""
    if not missing:
        return {}

    store = cached._store(root)
    ids = list(dict.fromkeys(int(x) for x in missing))
    names = ("gate_proj", "up_proj", "down_proj")
    weights_cpu: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    scales_cpu: dict[str, list[torch.Tensor]] = {name: [] for name in names}

    for expert_id in ids:
        expert_prefix = f"{layer_prefix}mlp.experts.{expert_id}"
        for name in names:
            prefix = f"{expert_prefix}.{name}"
            weights_cpu[name].append(store.load(prefix + ".weight", device="cpu"))
            scales_cpu[name].append(store.load(prefix + ".weight_scale_inv", device="cpu"))
            _stat("cpu_projection_loads")

    gpu: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    all_fp8 = True
    for name in names:
        wb = torch.stack(weights_cpu[name], dim=0)
        sb = torch.stack(scales_cpu[name], dim=0)
        if wb.dtype == torch.float8_e4m3fn:
            gpu[name] = (
                wb.to(device="cuda", non_blocking=True),
                sb.to(device="cuda", non_blocking=True),
            )
            _stat("weight_batches")
            _stat("scale_batches")
        else:
            all_fp8 = False
            gpu[name] = (
                wb.to(device="cuda", dtype=torch.float16, non_blocking=True),
                sb,
            )
            _stat("weight_batches")

    if not all_fp8:
        # The active Qwen3.6 checkpoint is FP8. Keep a conservative fallback for
        # non-FP8 checkpoints rather than silently changing their math.
        cache = _expert_cache(root)
        decoded = cache.get_or_load_batch(store, int(layer), ids, layer_prefix)
        return {expert_id: value for expert_id, value in zip(ids, decoded)}

    compact = {}
    for local, expert_id in enumerate(ids):
        compact[expert_id] = (
            (gpu["gate_proj"][0][local], gpu["gate_proj"][1][local]),
            (gpu["up_proj"][0][local], gpu["up_proj"][1][local]),
            (gpu["down_proj"][0][local], gpu["down_proj"][1][local]),
        )

    cache = _expert_cache(root)
    with cache.lock:
        for expert_id, entry in compact.items():
            cache._insert_fp8_locked(int(layer), int(expert_id), entry)
            cache.loads += 1

    return _decode_fp8([(expert_id, "fp8", compact[expert_id]) for expert_id in ids])


def _plan_layer(
    root: Path,
    layer: int,
    layer_prefix: str,
    expert_ids: list[int],
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Plan one complete routed layer before execution."""
    ids = [int(x) for x in expert_ids]
    unique_ids = list(dict.fromkeys(ids))
    cache = _expert_cache(root)
    output: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    fp8_found = []
    q4_found = []
    missing = []

    fp16_enabled = fp16_cache_module.FP16_CACHE_BYTES > 0
    materialized = fp16_cache_module._cache(root) if fp16_enabled else None

    for expert_id in unique_ids:
        if materialized is not None:
            hit = materialized.get((int(layer), expert_id))
            if hit is not None:
                output[expert_id] = hit
                _stat("fp16_hits")
                continue

        with cache.lock:
            fp8 = cache.fp8_entries.setdefault(int(layer), {}).get(expert_id)
            if fp8 is not None:
                cache.hits += 1
                cache.fp8_hits += 1
                cache.fp8_entries[int(layer)].move_to_end(expert_id)
                fp8_found.append((expert_id, "fp8", fp8))
                continue
            q4 = cache.q4_entries.setdefault(int(layer), {}).get(expert_id)
            if q4 is not None:
                cache.hits += 1
                cache.q4_hits += 1
                cache.q4_entries[int(layer)].move_to_end(expert_id)
                q4_found.append((expert_id, "q4", q4))
                continue
            cache.misses += 1
        missing.append(expert_id)

    output.update(_decode_fp8(fp8_found))
    _stat("fp8_hits", len(fp8_found))
    output.update(_decode_q4(q4_found))
    _stat("q4_hits", len(q4_found))

    if missing:
        _stat("miss_experts", len(missing))
        output.update(_load_missing_grouped(root, layer, layer_prefix, missing))

    if materialized is not None:
        for expert_id in unique_ids:
            materialized.put((int(layer), expert_id), output[expert_id])

    _stat("plans")
    _stat("unique_experts", len(unique_ids))
    return [output[expert_id] for expert_id in ids]


def _gate_up_gemm(gate_w: torch.Tensor, up_w: torch.Tensor, x: torch.Tensor):
    n, out_features, in_features = map(int, gate_w.shape)
    combined = torch.cat((gate_w, up_w), dim=1).reshape(n * 2 * out_features, in_features)
    y = torch.mm(combined, x.reshape(in_features, 1)).reshape(n, 2 * out_features)
    return y[:, :out_features], y[:, out_features:]


def _down_batched(down_w: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    return torch.bmm(down_w, hidden.unsqueeze(-1)).squeeze(-1)


def _batched_moe(root: Path, layer: int, residual: torch.Tensor, top_k: int, device: str):
    if device != "cuda":
        return _ORIGINAL_BATCHED_MOE_STEP(root, layer, residual, top_k, device)

    prefix = base.layer_prefix(layer)
    post_norm = base.load_layer_weight(root, layer, "post_attention_layernorm.weight", device)
    moe_in = base.rmsnorm(residual, post_norm).reshape(1, base.HIDDEN).float()
    router_w = base.load_layer_weight(root, layer, "mlp.gate.weight", device).float()
    routed = base.route(moe_in.reshape(-1), router_w, top_k=top_k)
    expert_ids = [int(v) for v in routed.expert_ids.detach().cpu().tolist()]
    weights = [float(v) for v in routed.weights.detach().cpu().tolist()]

    # Router has now produced the complete route for the current layer. From
    # this point onward the routed weights are handled as one batch.
    triplets = _plan_layer(root, layer, prefix, expert_ids)
    gate_w = torch.stack([triplet[0] for triplet in triplets], dim=0)
    up_w = torch.stack([triplet[1] for triplet in triplets], dim=0)
    down_w = torch.stack([triplet[2] for triplet in triplets], dim=0)
    batch_x = moe_in.reshape(-1).to(dtype=torch.float16)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        gate, up = _gate_up_gemm(gate_w, up_w, batch_x)
        hidden = F.silu(gate) * up
        expert_out = _down_batched(down_w, hidden)
        routing = torch.as_tensor(weights, device="cuda", dtype=expert_out.dtype).reshape(-1, 1)
        routed_sum = (expert_out * routing).sum(dim=0, keepdim=True)

    shared_gate_w = base.load_layer_weight(root, layer, "mlp.shared_expert_gate.weight", device).float()
    shared_gate_proj = chat._projection(root, f"{prefix}mlp.shared_expert.gate_proj", device)
    shared_up_proj = chat._projection(root, f"{prefix}mlp.shared_expert.up_proj", device)
    shared_down_proj = chat._projection(root, f"{prefix}mlp.shared_expert.down_proj", device)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        shared_gate = torch.sigmoid(F.linear(moe_in, shared_gate_w))
        shared_hidden = F.silu(
            F.linear(moe_in.to(shared_gate_proj.dtype), shared_gate_proj)
        ) * F.linear(moe_in.to(shared_up_proj.dtype), shared_up_proj)
        shared_out = F.linear(shared_hidden, shared_down_proj) * shared_gate

    moe_out = routed_sum.float() + shared_out.float()
    layer_out = residual + moe_out
    shared_gate_value = float(shared_gate.float().item())
    moe_input_norm = float(torch.linalg.vector_norm(moe_in).item())
    return layer_out, expert_ids, weights, shared_gate_value, moe_input_norm


def _cache_stats(root: Path) -> dict[str, int | float]:
    result = dict(_ORIGINAL_CACHE_STATS(root))
    with _STATS_LOCK:
        snap = dict(_STATS)
    plans = max(int(snap["plans"]), 1)
    result.update({
        "expert_batch_plans": int(snap["plans"]),
        "expert_batch_unique_experts": int(snap["unique_experts"]),
        "expert_batch_fp16_hits": int(snap["fp16_hits"]),
        "expert_batch_fp8_hits": int(snap["fp8_hits"]),
        "expert_batch_q4_hits": int(snap["q4_hits"]),
        "expert_batch_miss_experts": int(snap["miss_experts"]),
        "expert_batch_weight_batches": int(snap["weight_batches"]),
        "expert_batch_scale_batches": int(snap["scale_batches"]),
        "expert_batch_grouped_h2d": int(snap["weight_batches"] + snap["scale_batches"]),
        "expert_batch_cpu_projection_loads": int(snap["cpu_projection_loads"]),
        "expert_batch_unique_per_plan": float(snap["unique_experts"] / plans),
        "expert_batch_h2d_per_plan": float(
            (snap["weight_batches"] + snap["scale_batches"]) / plans
        ),
    })
    return result


chat.batched_moe_step = _batched_moe
chat.cache_stats = _cache_stats

print(
    "expert_batch_plan_v2=enabled|route=complete-layer|"
    "planner=single-pass|h2d=grouped-per-projection|compute=batched-gemm|"
    "per-expert-threadpool=disabled"
)
