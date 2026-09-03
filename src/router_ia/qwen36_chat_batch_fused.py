from __future__ import annotations

"""Qwen3.6 chat runner with persistent compressed expert GPU cache."""

from pathlib import Path

import torch
import torch.nn.functional as F

from . import qwen36_cached_loop as cached
from . import qwen36_chat_batch as chat
from . import qwen36_40layer_loop as base
from .qwen36_expert_cache import RoutedExpertCache


_EXPERT_CACHES: dict[Path, RoutedExpertCache] = {}
_ORIGINAL_EXPERT_TRIPLET = chat._expert_projection_triplet
_ORIGINAL_CACHE_STATS = chat.cache_stats
_ORIGINAL_PRINT_CACHE = chat.print_cache


def _expert_cache(root: Path) -> RoutedExpertCache:
    key = root.resolve()
    cache = _EXPERT_CACHES.get(key)
    if cache is None:
        cache = RoutedExpertCache(cached.STREAM_BUDGET_BYTES)
        _EXPERT_CACHES[key] = cache
    return cache


def _cached_expert_projection_triplet(
    root: Path,
    layer_prefix: str,
    expert_id: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if device != "cuda":
        return _ORIGINAL_EXPERT_TRIPLET(root, layer_prefix, expert_id, device)
    layer_marker = ".layers."
    if layer_marker not in layer_prefix:
        return _ORIGINAL_EXPERT_TRIPLET(root, layer_prefix, expert_id, device)
    try:
        layer = int(layer_prefix.split(layer_marker, 1)[1].split(".", 1)[0])
    except (ValueError, IndexError):
        return _ORIGINAL_EXPERT_TRIPLET(root, layer_prefix, expert_id, device)
    return _expert_cache(root).get_or_load(cached._store(root), layer, expert_id, layer_prefix)


def _load_route_batch_preserving_duplicates(
    root: Path,
    layer: int,
    layer_prefix: str,
    expert_ids: list[int],
):
    """Load each unique routed expert once, then restore original top-k order.

    Routing can contain repeated expert IDs. The cache is keyed by (layer, expert),
    so duplicate IDs must not be allowed to shrink the compute batch.
    """
    expert_cache = _expert_cache(root)
    store = cached._store(root)

    unique_ids = list(dict.fromkeys(int(x) for x in expert_ids))
    loaded = {
        expert_id: expert_cache.get_or_load(store, layer, expert_id, layer_prefix)
        for expert_id in unique_ids
    }
    return [loaded[int(expert_id)] for expert_id in expert_ids]


def _batched_moe_step_gpu(
    root: Path,
    layer: int,
    residual: torch.Tensor,
    top_k: int,
    device: str,
):
    """MoE step whose routed expert weights are prepared on CUDA."""
    if device != "cuda":
        return _ORIGINAL_BATCHED_MOE_STEP(root, layer, residual, top_k, device)

    prefix = base.layer_prefix(layer)
    post_norm = base.load_layer_weight(root, layer, "post_attention_layernorm.weight", device)
    moe_in = base.rmsnorm(residual, post_norm).reshape(1, base.HIDDEN).float()
    router_w = base.load_layer_weight(root, layer, "mlp.gate.weight", device).float()
    routed = base.route(moe_in.reshape(-1), router_w, top_k=top_k)
    expert_ids = [int(v) for v in routed.expert_ids.detach().cpu().tolist()]
    weights = [float(v) for v in routed.weights.detach().cpu().tolist()]

    triplets = _load_route_batch_preserving_duplicates(root, layer, prefix, expert_ids)
    gate_w = torch.stack([triplet[0] for triplet in triplets], dim=0)
    up_w = torch.stack([triplet[1] for triplet in triplets], dim=0)
    down_w = torch.stack([triplet[2] for triplet in triplets], dim=0)
    batch_x = moe_in.expand(len(expert_ids), -1).to(dtype=torch.float16)

    if len(triplets) != len(expert_ids):
        raise RuntimeError(
            f"Expert route batch mismatch: requested {len(expert_ids)}, loaded {len(triplets)}"
        )

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        gate = torch.bmm(gate_w, batch_x.unsqueeze(-1)).squeeze(-1)
        up = torch.bmm(up_w, batch_x.unsqueeze(-1)).squeeze(-1)
        hidden = F.silu(gate) * up
        expert_out = torch.bmm(down_w, hidden.unsqueeze(-1)).squeeze(-1)
        routing = torch.tensor(weights, device=device, dtype=expert_out.dtype).reshape(-1, 1)
        routed_sum = (expert_out * routing).sum(dim=0, keepdim=True)

    shared_gate_w = base.load_layer_weight(root, layer, "mlp.shared_expert_gate.weight", device).float()
    shared_gate_proj = chat._projection(root, f"{prefix}mlp.shared_expert.gate_proj", device)
    shared_up_proj = chat._projection(root, f"{prefix}mlp.shared_expert.up_proj", device)
    shared_down_proj = chat._projection(root, f"{prefix}mlp.shared_expert.down_proj", device)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        shared_gate = torch.sigmoid(F.linear(moe_in, shared_gate_w))
        shared_hidden = F.silu(F.linear(moe_in.to(shared_gate_proj.dtype), shared_gate_proj)) * F.linear(moe_in.to(shared_up_proj.dtype), shared_up_proj)
        shared_out = F.linear(shared_hidden, shared_down_proj) * shared_gate

    moe_out = routed_sum.float() + shared_out.float()
    layer_out = residual + moe_out
    shared_gate_value = float(shared_gate.float().item())
    moe_input_norm = float(torch.linalg.vector_norm(moe_in).item())

    del post_norm, moe_in, router_w, routed
    del triplets, gate_w, up_w, down_w, batch_x
    del gate, up, hidden, expert_out, routing, routed_sum
    del shared_gate_w, shared_gate, shared_gate_proj, shared_up_proj, shared_down_proj
    del shared_hidden, shared_out, moe_out
    return layer_out, expert_ids, weights, shared_gate_value, moe_input_norm


def _cache_stats_with_experts(root: Path) -> dict[str, int | float]:
    stats = dict(_ORIGINAL_CACHE_STATS(root))
    cache = _EXPERT_CACHES.get(root.resolve())
    if cache is None:
        return stats
    expert = cache.snapshot()
    stats.update({
        "expert_cache_items": int(expert["items"]),
        "expert_cache_bytes": int(expert["bytes"]),
        "expert_cache_budget": int(expert["budget_bytes"]),
        "expert_cache_total_slots": int(expert["total_slots"]),
        "expert_cache_hits": int(expert["hits"]),
        "expert_cache_misses": int(expert["misses"]),
        "expert_cache_hit_rate": float(expert["hit_rate"]),
        "expert_cache_loads": int(expert["loads"]),
        "expert_cache_evictions": int(expert["evictions"]),
        "expert_cache_fp8_items": int(expert["warm_items"]),
        "expert_cache_q4_items": int(expert["cold_items"]),
        "expert_cache_fp8_hits": int(expert["fp8_hits"]),
        "expert_cache_q4_hits": int(expert["q4_hits"]),
        "expert_cache_fp16_to_fp8": int(expert["fp16_to_fp8"]),
        "expert_cache_fp8_to_q4": int(expert["fp8_to_q4"]),
        "expert_cache_q4_drops": int(expert["q4_drops"]),
        "expert_cache_stream_prefetch_hits": int(expert["stream_prefetch_hits"]),
        "expert_cache_stream_prefetch_misses": int(expert["stream_prefetch_misses"]),
    })
    return stats


def _print_cache_with_experts(root: Path, label: str) -> None:
    _ORIGINAL_PRINT_CACHE(root, label)
    cache = _EXPERT_CACHES.get(root.resolve())
    if cache is None:
        return
    expert = cache.snapshot()
    print(
        f"  expert_cache: entries={expert['items']} | "
        f"vram={expert['bytes'] / 1024**2:.1f}/{expert['budget_bytes'] / 1024**2:.1f} MiB | "
        f"hit_rate={expert['hit_rate']:.2f}% | hits={expert['hits']} | "
        f"misses={expert['misses']} | loads={expert['loads']} | evictions={expert['evictions']}"
    )
    print(
        f"    tiers: FP8={expert['warm_items']} | Q4={expert['cold_items']} | "
        f"hits FP8={expert['fp8_hits']} Q4={expert['q4_hits']} | "
        f"compressions FP8>Q4={expert['fp8_to_q4']} | drops={expert['q4_drops']} | "
        f"prefetch hits={expert['stream_prefetch_hits']} misses={expert['stream_prefetch_misses']}"
    )


chat._expert_projection_triplet = _cached_expert_projection_triplet
_ORIGINAL_BATCHED_MOE_STEP = chat.batched_moe_step
chat.batched_moe_step = _batched_moe_step_gpu
chat.cache_stats = _cache_stats_with_experts
chat.print_cache = _print_cache_with_experts


def main() -> None:
    cache = _expert_cache(Path("."))
    print("expert_cache=complete-layer-expert")
    print("expert_cache_key=(layer,expert)")
    print("expert_cache_policy=per-layer-tiered-8fp8-4q4")
    print("expert_cache_budget=full-stream-vram-budget")
    print("expert_cache_entry=FP8-resident|Q4-cold")
    print("expert_cache_eviction=FP8-to-Q4-then-drop")
    print("expert_cache_fp16_persistent=disabled")
    print("expert_cache_fp8_promotion=disabled")
    print("expert_cache_prefetch=parallel-raw-fp8-stream")
    print("expert_cache_compute=temporary-fp16")
    print("expert_cache_compute_batch=8-experts-preserve-duplicates")
    print(f"expert_cache_total_slots={cache.total_slots}")
    print(f"expert_cache_slots_per_layer={cache.slots_per_layer}")
    print(f"expert_cache_fp8_slots_per_layer={cache.fp8_slots}")
    print(f"expert_cache_q4_slots_per_layer={cache.q4_slots}")
    chat.main()


if __name__ == "__main__":
    main()
