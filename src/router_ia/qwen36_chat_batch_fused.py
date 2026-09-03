from __future__ import annotations

"""Qwen3.6 chat runner with persistent compressed expert GPU cache."""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

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
_ORIGINAL_BATCHED_MOE_STEP = chat.batched_moe_step
_ORIGINAL_RUN_GENERATED_TOKEN = chat.run_generated_token


class RoutingPredictor:
    """Learn recurring expert routes and speculatively prefetch the next token."""

    def __init__(self, top_n: int = 4, min_observations: int = 1) -> None:
        self.top_n = max(int(top_n), 1)
        self.min_observations = max(int(min_observations), 1)
        self._unigram: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
        self._bigram: dict[tuple[int, int, int], Counter[int]] = defaultdict(Counter)
        self._observations: dict[tuple[int, int], int] = defaultdict(int)
        self._pending: dict[tuple[int, int, int], set[int]] = {}
        self._predictions = 0
        self._predicted_experts = 0
        self._matched_experts = 0
        self._lock = Lock()

    def observe(self, previous_token: int | None, token_id: int, layer: int, expert_ids: list[int]) -> None:
        token_id = int(token_id)
        layer = int(layer)
        ids = [int(x) for x in expert_ids]
        with self._lock:
            key = (token_id, layer)
            self._unigram[key].update(ids)
            self._observations[key] += 1
            if previous_token is not None:
                sequence_key = (int(previous_token), token_id, layer)
                self._bigram[sequence_key].update(ids)
                pending = self._pending.pop(sequence_key, None)
                if pending:
                    self._matched_experts += len(pending.intersection(ids))

    def predict(self, previous_token: int | None, token_id: int, layer: int) -> list[int]:
        token_id = int(token_id)
        layer = int(layer)
        with self._lock:
            candidates: Counter[int] | None = None
            if previous_token is not None:
                bigram = self._bigram.get((int(previous_token), token_id, layer))
                if bigram and sum(bigram.values()) >= self.min_observations:
                    candidates = bigram
            if candidates is None:
                unigram = self._unigram.get((token_id, layer))
                if unigram and self._observations.get((token_id, layer), 0) >= self.min_observations:
                    candidates = unigram
            if not candidates:
                return []
            return [expert for expert, _ in candidates.most_common(self.top_n)]

    def predict_route(self, previous_token: int | None, token_id: int, layer: int) -> list[int]:
        predicted = self.predict(previous_token, token_id, layer)
        if not predicted:
            return []
        with self._lock:
            self._predictions += 1
            self._predicted_experts += len(predicted)
            if previous_token is not None:
                self._pending[(int(previous_token), int(token_id), int(layer))] = set(predicted)
        return predicted

    def snapshot(self) -> dict[str, int | float]:
        with self._lock:
            precision = self._matched_experts / self._predicted_experts * 100.0 if self._predicted_experts else 0.0
            return {
                "predictions": self._predictions,
                "predicted_experts": self._predicted_experts,
                "matched_experts": self._matched_experts,
                "expert_precision": precision,
                "top_n": self.top_n,
                "min_observations": self.min_observations,
                "contexts": len(self._unigram),
                "bigram_contexts": len(self._bigram),
            }


_ROUTING_PREDICTOR = RoutingPredictor(top_n=4, min_observations=1)
_LAST_INPUT_TOKEN: int | None = None


def _expert_cache(root: Path) -> RoutedExpertCache:
    key = root.resolve()
    cache = _EXPERT_CACHES.get(key)
    if cache is None:
        cache = RoutedExpertCache(cached.STREAM_BUDGET_BYTES)
        _EXPERT_CACHES[key] = cache
    return cache


def _cached_expert_projection_triplet(root: Path, layer_prefix: str, expert_id: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def _load_route_batch_preserving_duplicates(root: Path, layer: int, layer_prefix: str, expert_ids: list[int]):
    """Load each unique routed expert once, then restore original top-k order."""
    expert_cache = _expert_cache(root)
    store = cached._store(root)
    unique_ids = list(dict.fromkeys(int(x) for x in expert_ids))
    loaded = {expert_id: expert_cache.get_or_load(store, layer, expert_id, layer_prefix) for expert_id in unique_ids}
    return [loaded[int(expert_id)] for expert_id in expert_ids]


def _route_matvec_batched(weight: torch.Tensor, x: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Run N expert matvecs as one CUDA GEMM instead of N tiny batched GEMMs.

    ``weight`` is [N, out_features, in_features] and ``x`` is [in_features].
    Flattening the expert dimension lets cuBLAS process one larger GEMM while
    preserving the per-expert output layout expected by the router.
    """
    if weight.ndim != 3 or x.ndim != 1:
        raise ValueError(f"Expected [N,O,I] weight and [I] input, got {tuple(weight.shape)} and {tuple(x.shape)}")
    if int(weight.shape[0]) != int(batch_size):
        raise ValueError(f"Route batch size mismatch: {weight.shape[0]} != {batch_size}")
    out_features, in_features = map(int, weight.shape[1:])
    flattened = weight.reshape(batch_size * out_features, in_features)
    result = torch.mm(flattened, x.reshape(in_features, 1))
    return result.reshape(batch_size, out_features)


def _batched_moe_step_gpu(root: Path, layer: int, residual: torch.Tensor, top_k: int, device: str):
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
    if len(triplets) != len(expert_ids):
        raise RuntimeError(f"Expert route batch mismatch: requested {len(expert_ids)}, loaded {len(triplets)}")

    if _CURRENT_TOKEN_ID is not None:
        _ROUTING_PREDICTOR.observe(_LAST_INPUT_TOKEN, _CURRENT_TOKEN_ID, layer, expert_ids)

    gate_w = torch.stack([triplet[0] for triplet in triplets], dim=0)
    up_w = torch.stack([triplet[1] for triplet in triplets], dim=0)
    down_w = torch.stack([triplet[2] for triplet in triplets], dim=0)
    batch_x = moe_in.reshape(-1).to(dtype=torch.float16)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        gate = _route_matvec_batched(gate_w, batch_x, len(expert_ids))
        up = _route_matvec_batched(up_w, batch_x, len(expert_ids))
        hidden = F.silu(gate) * up
        expert_out = _route_matvec_batched(down_w, hidden, len(expert_ids))
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

    del post_norm, moe_in, router_w, routed, triplets, gate_w, up_w, down_w, batch_x
    del gate, up, hidden, expert_out, routing, routed_sum
    del shared_gate_w, shared_gate, shared_gate_proj, shared_up_proj, shared_down_proj
    del shared_hidden, shared_out, moe_out
    return layer_out, expert_ids, weights, shared_gate_value, moe_input_norm


def _prefetch_predicted_routes(root: Path, previous_token: int | None, token_id: int) -> tuple[int, int]:
    if not torch.cuda.is_available():
        return 0, 0
    store = cached._store(root)
    expert_cache = _expert_cache(root)
    jobs: list[tuple[str, int]] = []
    for layer in range(base.DEFAULT_LAYERS):
        predicted = _ROUTING_PREDICTOR.predict_route(previous_token, token_id, layer)
        if not predicted:
            continue
        prefix = base.layer_prefix(layer)
        for expert_id in predicted:
            jobs.append((prefix, int(expert_id)))
    if not jobs:
        return 0, 0

    with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as pool:
        futures = [pool.submit(expert_cache.prefetch_expert_raw, store, prefix, expert_id) for prefix, expert_id in jobs]
        for future in futures:
            future.result()
    return len(jobs), sum(1 for _ in jobs)


def _run_generated_token_with_predictor(root: Path, token_id: int, final_norm: torch.Tensor, lm_head: torch.Tensor, final_norm_name: str, lm_head_name: str, device: str, sampling_top_k: int, temperature: float):
    global _LAST_INPUT_TOKEN, _CURRENT_TOKEN_ID
    previous_token = _LAST_INPUT_TOKEN
    _CURRENT_TOKEN_ID = int(token_id)
    result = _ORIGINAL_RUN_GENERATED_TOKEN(root, token_id, final_norm, lm_head, final_norm_name, lm_head_name, device, sampling_top_k, temperature)
    if device == "cuda":
        _prefetch_predicted_routes(root, int(token_id), int(result[0]))
    _LAST_INPUT_TOKEN = int(token_id)
    _CURRENT_TOKEN_ID = None
    return result

_CURRENT_TOKEN_ID: int | None = None


def _cache_stats_with_experts(root: Path) -> dict[str, int | float]:
    stats = dict(_ORIGINAL_CACHE_STATS(root))
    cache = _EXPERT_CACHES.get(root.resolve())
    if cache is None:
        return stats
    expert = cache.snapshot()
    predictor = _ROUTING_PREDICTOR.snapshot()
    stats.update({
        "expert_cache_items": int(expert["items"]),
        "expert_cache_bytes": int(expert["bytes"]),
        "expert_cache_budget": int(expert["budget_bytes"]),
        "expert_cache_q4_ram_bytes": int(expert["q4_ram_bytes"]),
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
        "expert_cache_q4_ram_evictions": int(expert["q4_ram_evictions"]),
        "expert_cache_stream_prefetch_hits": int(expert["stream_prefetch_hits"]),
        "expert_cache_stream_prefetch_misses": int(expert["stream_prefetch_misses"]),
        "routing_predictor_predictions": int(predictor["predictions"]),
        "routing_predictor_predicted_experts": int(predictor["predicted_experts"]),
        "routing_predictor_matched_experts": int(predictor["matched_experts"]),
        "routing_predictor_precision": float(predictor["expert_precision"]),
        "routing_predictor_contexts": int(predictor["contexts"]),
        "routing_predictor_bigram_contexts": int(predictor["bigram_contexts"]),
    })
    return stats


def _print_cache_with_experts(root: Path, label: str) -> None:
    _ORIGINAL_PRINT_CACHE(root, label)
    cache = _EXPERT_CACHES.get(root.resolve())
    if cache is None:
        return
    expert = cache.snapshot()
    predictor = _ROUTING_PREDICTOR.snapshot()
    print(
        f"  expert_cache: fp8_vram_entries={expert['warm_items']} | "
        f"fp8_vram={expert['bytes'] / 1024**2:.1f}/{expert['budget_bytes'] / 1024**2:.1f} MiB | "
        f"q4_ram_entries={expert['cold_items']} | "
        f"q4_ram={expert['q4_ram_bytes'] / 1024**2:.1f} MiB | "
        f"hit_rate={expert['hit_rate']:.2f}% | hits={expert['hits']} | misses={expert['misses']} | loads={expert['loads']}"
    )
    print(
        f"    tiers: FP8=VRAM:{expert['warm_items']} | Q4=RAM:{expert['cold_items']} | "
        f"hits FP8={expert['fp8_hits']} Q4={expert['q4_hits']} | "
        f"GPU compressions FP8>Q4={expert['fp8_to_q4']} | Q4 RAM evictions={expert['q4_ram_evictions']}"
    )
    print(
        f"  routing_predictor: predictions={predictor['predictions']} | "
        f"predicted={predictor['predicted_experts']} | matched={predictor['matched_experts']} | "
        f"precision={predictor['expert_precision']:.2f}% | contexts={predictor['contexts']} | "
        f"bigrams={predictor['bigram_contexts']}"
    )


chat._expert_projection_triplet = _cached_expert_projection_triplet
chat.batched_moe_step = _batched_moe_step_gpu
chat.run_generated_token = _run_generated_token_with_predictor
chat.cache_stats = _cache_stats_with_experts
chat.print_cache = _print_cache_with_experts


def main() -> None:
    cache = _expert_cache(Path("."))
    print("expert_cache=complete-layer-expert")
    print("expert_cache_key=(layer,expert)")
    print("expert_cache_policy=per-layer-8fp8-vram-3q4-ram")
    print("expert_cache_budget=fp8-vram-stream-budget")
    print("expert_cache_entry=FP8-VRAM|Q4-RAM")
    print("expert_cache_eviction=FP8-to-Q4-RAM")
    print("expert_cache_fp16_persistent=disabled")
    print("expert_cache_fp8_promotion=disabled")
    print("expert_cache_prefetch=parallel-raw-fp8-stream")
    print("expert_cache_compute=temporary-fp16")
    print("expert_cache_compute_batch=single-gemm-per-projection")
    print("expert_cache_kernel_fused_dequant=not-yet")
    print("routing_predictor=enabled")
    print("routing_predictor_policy=bigram-with-unigram-fallback")
    print("routing_predictor_top_n=4")
    print("routing_predictor_prefetch=next-token-all-layers")
    print(f"expert_cache_total_slots={cache.total_slots}")
    print(f"expert_cache_slots_per_layer={cache.slots_per_layer}")
    print(f"expert_cache_fp8_slots_per_layer={cache.fp8_slots}")
    print(f"expert_cache_q4_ram_slots_per_layer={cache.q4_slots}")
    chat.main()


if __name__ == "__main__":
    main()
