from __future__ import annotations

"""Batch-first routed-expert planner for Qwen3.6.

The normal hot path used one Python/cache operation per routed expert. This
patch changes the structure to:

    router -> all selected expert ids for this layer -> one plan -> grouped
    host loads -> grouped H2D transfers -> grouped GPU expert execution.

The planner is deliberately layered on top of the existing caches, so RAM/Q4,
FP8, and the prompt-scoped FP16 materialization cache remain usable.
"""

from pathlib import Path
from threading import Lock

import torch
import torch.nn.functional as F

from . import qwen36_chat_batch as chat
from . import qwen36_cached_loop as cached
from . import qwen36_dequant as dequant
from . import qwen36_40layer_loop as base
from . import qwen36_official_optimizations as official
from . import qwen36_expert_cache as expert_cache_module
from . import qwen36_fp16_expert_cache as fp16_cache_module


_ORIGINAL_BATCHED_MOE_STEP = chat.batched_moe_step
_STATS_LOCK = Lock()
_STATS = {
    "plans": 0,
    "layers": 0,
    "unique_experts": 0,
    "fp16_hits": 0,
    "compressed_hits": 0,
    "miss_experts": 0,
    "weight_batches": 0,
    "scale_batches": 0,
    "cpu_projection_loads": 0,
    "gpu_projection_views": 0,
}


def _add_stat(name: str, value: int = 1) -> None:
    with _STATS_LOCK:
        _STATS[name] += int(value)


def _expert_cache(root: Path):
    key = root.resolve()
    cache = official._EXPERT_CACHES.get(key)
    if cache is None:
        # The official optimizer owns the budget and cache registry.
        from .qwen36_expert_cache import RoutedExpertCache

        cache = RoutedExpertCache(official.EXPERT_VRAM_BUDGET_BYTES)
        official._EXPERT_CACHES[key] = cache
    return cache


def _raw_cpu_projection(store, prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Read one raw FP8 projection pair without performing H2D transfer."""
    weight = store.load(prefix + ".weight", device="cpu")
    scale = store.load(prefix + ".weight_scale_inv", device="cpu")
    _add_stat("cpu_projection_loads")
    return weight, scale


def _decode_compressed(
    cache,
    found: list[tuple[int, str, object]],
) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Decode already-cached FP8/Q4 entries as a grouped GPU operation."""
    if not found:
        return {}

    fp8_items = [(expert_id, entry) for expert_id, tier, entry in found if tier == "fp8"]
    q4_items = [(expert_id, entry) for expert_id, tier, entry in found if tier == "q4"]
    output: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    if fp8_items:
        per_projection: list[list[torch.Tensor]] = []
        for projection in range(3):
            weights = torch.stack([entry[projection][0] for _, entry in fp8_items], dim=0)
            scales = torch.stack([entry[projection][1] for _, entry in fp8_items], dim=0)
            decoded = dequant.dequantize_fp8_blockwise_batch(weights, scales).to(dtype=torch.float16)
            per_projection.append(list(decoded.unbind(0)))

        for local, (expert_id, _) in enumerate(fp8_items):
            output[expert_id] = (
                per_projection[0][local],
                per_projection[1][local],
                per_projection[2][local],
            )
        _add_stat("compressed_hits", len(fp8_items))

    if q4_items:
        per_projection_q4: list[list[torch.Tensor]] = []
        for projection in range(3):
            decoded = expert_cache_module._q4_dequantize_entry_batch(
                [entry for _, entry in q4_items], projection
            )
            per_projection_q4.append(decoded)

        for local, (expert_id, _) in enumerate(q4_items):
            output[expert_id] = (
                per_projection_q4[0][local],
                per_projection_q4[1][local],
                per_projection_q4[2][local],
            )
        _add_stat("compressed_hits", len(q4_items))

    return output


def _load_missing_as_fp8(
    root: Path,
    layer: int,
    layer_prefix: str,
    expert_ids: list[int],
) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Load missing experts from host RAM and transfer each projection as one batch."""
    if not expert_ids:
        return {}

    store = cached._store(root)
    ordered_ids = list(dict.fromkeys(int(x) for x in expert_ids))
    raw: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] = {
        name: ([], []) for name in ("gate_proj", "up_proj", "down_proj")
    }

    for expert_id in ordered_ids:
        prefix = f"{layer_prefix}mlp.experts.{expert_id}"
        for name in raw:
            weight, scale = _raw_cpu_projection(store, f"{prefix}.{name}")
            raw[name][0].append(weight)
            raw[name][1].append(scale)

    output: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    gpu_pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    for name, (weights, scales) in raw.items():
        weight_batch_cpu = torch.stack(weights, dim=0)
        scale_batch_cpu = torch.stack(scales, dim=0)
        if weight_batch_cpu.dtype == torch.float8_e4m3fn:
            # Exactly one grouped weight transfer and one grouped scale transfer
            # for this projection, instead of one pair per routed expert.
            weight_batch_gpu = weight_batch_cpu.to(device="cuda", non_blocking=True)
            scale_batch_gpu = scale_batch_cpu.to(device="cuda", non_blocking=True)
            gpu_pairs[name] = (weight_batch_gpu, scale_batch_gpu)
            _add_stat("weight_batches")
            _add_stat("scale_batches")
        else:
            weight_batch_gpu = weight_batch_cpu.to(device="cuda", dtype=torch.float16, non_blocking=True)
            scale_batch_gpu = scale_batch_cpu
            gpu_pairs[name] = (weight_batch_gpu, scale_batch_gpu)
            _add_stat("weight_batches")

    # The active checkpoint is FP8. Materialize a compact WarmEntry per expert
    # from grouped GPU views, then install them in the existing expert cache.
    if all(gpu_pairs[name][0].dtype == torch.float8_e4m3fn for name in gpu_pairs):
        loaded_compact = {}
        with expert_cache_module.RoutedExpertCache.__dict__.get("lock", Lock()):
            pass
        for local, expert_id in enumerate(ordered_ids):
            compact = (
                (gpu_pairs["gate_proj"][0][local], gpu_pairs["gate_proj"][1][local]),
                (gpu_pairs["up_proj"][0][local], gpu_pairs["up_proj"][1][local]),
                (gpu_pairs["down_proj"][0][local], gpu_pairs["down_proj"][1][local]),
            )
            loaded_compact[expert_id] = compact

        with cache.lock:
            for expert_id, compact in loaded_compact.items():
                cache._insert_fp8_locked(int(layer), int(expert_id), compact)
                cache.loads += 1
        return _decode_compressed(
            cache,
            [(expert_id, "fp8", loaded_compact[expert_id]) for expert_id in ordered_ids],
        )

    # Conservative fallback for a non-FP8 checkpoint: grouped transfer still
    # happens, but the existing cache loader remains responsible for fidelity.
    return {
        expert_id: triplet
        for expert_id, triplet in zip(
            ordered_ids,
            expert_cache_module.RoutedExpertCache.get_or_load_batch(
                cache, store, int(layer), ordered_ids, layer_prefix
            ),
        )
    }


def _plan_layer(
    root: Path,
    layer: int,
    layer_prefix: str,
    expert_ids: list[int],
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Build the complete route plan once, then return triplets in top-k order."""
    ordered_ids = [int(x) for x in expert_ids]
    unique_ids = list(dict.fromkeys(ordered_ids))
    materialized = fp16_cache_module._cache(root)
    cache = _expert_cache(root)

    output: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    compressed_found: list[tuple[int, str, object]] = []
    missing: list[int] = []

    for expert_id in unique_ids:
        fp16 = materialized.get((int(layer), expert_id))
        if fp16 is not None:
            output[expert_id] = fp16
            _add_stat("fp16_hits")
            continue

        with cache.lock:
            fp8 = cache.fp8_entries.setdefault(int(layer), {}).get(expert_id)
            if fp8 is not None:
                cache.fp8_entries[int(layer)].move_to_end(expert_id)
                compressed_found.append((expert_id, "fp8", fp8))
                continue
            q4 = cache.q4_entries.setdefault(int(layer), {}).get(expert_id)
            if q4 is not None:
                cache.q4_entries[int(layer)].move_to_end(expert_id)
                compressed_found.append((expert_id, "q4", q4))
                continue
        missing.append(expert_id)

    output.update(_decode_compressed(cache, compressed_found))
    if missing:
        _add_stat("miss_experts", len(missing))
        output.update(_load_missing_as_fp8(root, layer, layer_prefix, missing))

    # Cache only the requested expert triplets after the grouped decode. This
    # uses the same prompt-scoped FP16 pool as the existing FP16 patch.
    for expert_id in unique_ids:
        triplet = output.get(expert_id)
        if triplet is None:
            raise RuntimeError(f"Batch expert planner lost layer={layer} expert={expert_id}")
        materialized.put((int(layer), expert_id), triplet)
        _add_stat("gpu_projection_views", 3)

    _add_stat("plans")
    _add_stat("layers")
    _add_stat("unique_experts", len(unique_ids))
    return [output[expert_id] for expert_id in ordered_ids]


def _route_projection_batched(weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 3:
        raise ValueError(f"Expected [N,O,I] weight, got {tuple(weight.shape)}")
    if x.ndim == 1:
        n, out_features, in_features = map(int, weight.shape)
        if int(x.shape[0]) != in_features:
            raise ValueError(f"Input {tuple(x.shape)} incompatible with {tuple(weight.shape)}")
        result = torch.mm(weight.reshape(n * out_features, in_features), x.reshape(in_features, 1))
        return result.reshape(n, out_features)
    if x.ndim == 2:
        if int(x.shape[0]) != int(weight.shape[0]):
            raise ValueError(f"Batch mismatch: weights={weight.shape} input={x.shape}")
        return torch.bmm(weight, x.unsqueeze(-1)).squeeze(-1)
    raise ValueError(f"Unsupported input shape: {tuple(x.shape)}")


def _gate_up_single_gemm(
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n, out_features, in_features = map(int, gate_w.shape)
    combined = torch.cat((gate_w, up_w), dim=1).reshape(n * 2 * out_features, in_features)
    y = torch.mm(combined, x.reshape(in_features, 1)).reshape(n, 2 * out_features)
    return y[:, :out_features], y[:, out_features:]


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

    triplets = _plan_layer(root, layer, prefix, expert_ids)
    gate_w = torch.stack([triplet[0] for triplet in triplets], dim=0)
    up_w = torch.stack([triplet[1] for triplet in triplets], dim=0)
    down_w = torch.stack([triplet[2] for triplet in triplets], dim=0)
    batch_x = moe_in.reshape(-1).to(dtype=torch.float16)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        gate, up = _gate_up_single_gemm(gate_w, up_w, batch_x)
        hidden = F.silu(gate) * up
        expert_out = _route_projection_batched(down_w, hidden)
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


def _stats(root: Path) -> dict[str, int | float]:
    with _STATS_LOCK:
        snapshot = dict(_STATS)
    plans = max(int(snapshot["plans"]), 1)
    return {
        **snapshot,
        "unique_experts_per_plan": snapshot["unique_experts"] / plans,
        "grouped_h2d_batches_per_plan": (
            snapshot["weight_batches"] + snapshot["scale_batches"]
        ) / plans,
    }


def _cache_stats(root: Path) -> dict[str, int | float]:
    original = dict(chat.cache_stats(root))
    original["expert_batch_plans"] = int(_stats(root)["plans"])
    original["expert_batch_unique_experts"] = int(_stats(root)["unique_experts"])
    original["expert_batch_fp16_hits"] = int(_stats(root)["fp16_hits"])
    original["expert_batch_compressed_hits"] = int(_stats(root)["compressed_hits"])
    original["expert_batch_miss_experts"] = int(_stats(root)["miss_experts"])
    original["expert_batch_weight_batches"] = int(_stats(root)["weight_batches"])
    original["expert_batch_scale_batches"] = int(_stats(root)["scale_batches"])
    original["expert_batch_cpu_projection_loads"] = int(_stats(root)["cpu_projection_loads"])
    original["expert_batch_grouped_transfers"] = int(_stats(root)["weight_batches"] + _stats(root)["scale_batches"])
    return original


chat.batched_moe_step = _batched_moe
chat.cache_stats = _cache_stats

print(
    "expert_batch_plan=enabled|router=per-layer-full-route|"
    "planner=single-operation|transfers=grouped|compute=batched-GEMM|"
    "per-expert-threadpool=disabled"
)
