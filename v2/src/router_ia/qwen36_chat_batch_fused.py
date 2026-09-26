from __future__ import annotations

"""Qwen3.6 chat runner with persistent compressed expert GPU cache."""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from threading import Lock

import torch
import torch.nn.functional as F

from . import qwen36_cached_loop as cached
from . import qwen36_dequant as dequant
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


def _route_projection_batched(weight: torch.Tensor, x: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Run routed expert projections on CUDA with the cheapest valid GEMM form.

    A shared input vector can be handled as one flattened GEMM across all
    experts. The down projection receives one hidden vector per expert and
    therefore uses a batched GEMM.
    """
    if weight.ndim != 3:
        raise ValueError(f"Expected [N,O,I] weight, got {tuple(weight.shape)}")
    if int(weight.shape[0]) != int(batch_size):
        raise ValueError(f"Route batch size mismatch: {weight.shape[0]} != {batch_size}")

    out_features, in_features = map(int, weight.shape[1:])
    if x.ndim == 1:
        if int(x.shape[0]) != in_features:
            raise ValueError(f"Expected [I] input with I={in_features}, got {tuple(x.shape)}")
        flattened = weight.reshape(batch_size * out_features, in_features)
        result = torch.mm(flattened, x.reshape(in_features, 1))
        return result.reshape(batch_size, out_features)

    if x.ndim == 2:
        if tuple(x.shape) != (batch_size, in_features):
            raise ValueError(
                f"Expected [{batch_size},{in_features}] input for batched projection, got {tuple(x.shape)}"
            )
        return torch.bmm(weight, x.unsqueeze(-1)).squeeze(-1)

    raise ValueError(f"Expected [I] or [N,I] input, got {tuple(x.shape)}")


def _route_gate_up_single_gemm(gate_w: torch.Tensor, up_w: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute all routed gate/up projections with one flattened CUDA GEMM."""
    if gate_w.shape != up_w.shape or gate_w.ndim != 3:
        raise ValueError(
            f"Gate/up shape mismatch: {tuple(gate_w.shape)} vs {tuple(up_w.shape)}"
        )
    if x.ndim != 1 or int(x.shape[0]) != int(gate_w.shape[-1]):
        raise ValueError(f"Expected [I] input matching gate/up weights, got {tuple(x.shape)}")

    experts, out_features, in_features = map(int, gate_w.shape)
    combined = torch.cat((gate_w, up_w), dim=1).reshape(experts * 2 * out_features, in_features)
    result = torch.mm(combined, x.reshape(in_features, 1)).reshape(experts, 2 * out_features)
    return result[:, :out_features], result[:, out_features:]


def _batched_moe_step_gpu(root: Path, layer: int, residual: torch.Tensor, top_k: int, device: str):
    """Mixed MoE execution: FP8/VRAM experts on GPU, Q4/RAM experts on CPU."""
    if device != "cuda":
        return _ORIGINAL_BATCHED_MOE_STEP(root, layer, residual, top_k, device)

    from . import qwen36_cpu_expert as cpu_expert
    prefix = base.layer_prefix(layer)
    post_norm = base.load_layer_weight(root, layer, "post_attention_layernorm.weight", device)
    moe_in = base.rmsnorm(residual, post_norm).reshape(1, base.HIDDEN).float()
    router_w = base.load_layer_weight(root, layer, "mlp.gate.weight", device).float()
    routed = base.route(moe_in.reshape(-1), router_w, top_k=top_k)
    expert_ids = [int(v) for v in routed.expert_ids.detach().cpu().tolist()]
    route_weights = [float(v) for v in routed.weights.detach().cpu().tolist()]

    tiered = _expert_cache(root).get_or_load_batch_tiered(
        cached._store(root), layer, expert_ids, prefix
    )

    if _CURRENT_TOKEN_ID is not None:
        _ROUTING_PREDICTOR.observe(_LAST_INPUT_TOKEN, _CURRENT_TOKEN_ID, layer, expert_ids)

    fp8_entries = [entry for tier, entry in tiered if tier == "fp8"]
    q4_entries = [entry for tier, entry in tiered if tier == "q4"]
    routed_sum = torch.zeros_like(moe_in)

    cpu_future = None
    cpu_pool = None
    if q4_entries:
        workers_raw = int(os.getenv("QWEN36_CPU_EXPERT_WORKERS", "2"))
        workers = min(max(workers_raw, 1), len(q4_entries))
        # Start RAM-resident experts first so CPU execution overlaps the GPU path.
        cpu_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ram-experts")
        cpu_future = cpu_pool.submit(
            cpu_expert.run_q4_expert_batch_cpu, q4_entries, moe_in, workers=workers
        )

    expert_out = None
    fp8_weights = fp8_scales = gate_w = up_w = down_w = batch_x = None
    if fp8_entries:
        fp8_weights = []
        fp8_scales = []
        for projection in range(3):
            fp8_weights.append(torch.stack([entry[projection][0] for entry in fp8_entries], dim=0))
            fp8_scales.append(torch.stack([entry[projection][1] for entry in fp8_entries], dim=0))
        gate_w = dequant.dequantize_fp8_blockwise_batch(fp8_weights[0], fp8_scales[0]).to(dtype=torch.float16)
        up_w = dequant.dequantize_fp8_blockwise_batch(fp8_weights[1], fp8_scales[1]).to(dtype=torch.float16)
        down_w = dequant.dequantize_fp8_blockwise_batch(fp8_weights[2], fp8_scales[2]).to(dtype=torch.float16)
        batch_x = moe_in.reshape(-1).to(dtype=torch.float16)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            gate, up = _route_gate_up_single_gemm(gate_w, up_w, batch_x)
            hidden = F.silu(gate) * up
            expert_out = _route_projection_batched(down_w, hidden, len(fp8_entries))

    cpu_result = cpu_future.result() if cpu_future is not None else None
    if cpu_pool is not None:
        cpu_pool.shutdown(wait=True)

    fp8_local = 0
    q4_local = 0
    for route_pos, (tier, _) in enumerate(tiered):
        weight = route_weights[route_pos]
        if tier == "fp8":
            out = expert_out[fp8_local:fp8_local + 1]
            fp8_local += 1
        else:
            out = cpu_result[0][q4_local].to(device="cuda", dtype=torch.float32)
            q4_local += 1
        routed_sum.add_(out.float(), alpha=weight)

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

    if cpu_result is not None:
        s = cpu_result[1]
        _CPU_RUNTIME["experts"] += s.experts
        _CPU_RUNTIME["seconds"] += s.seconds
        _CPU_RUNTIME["dequant_seconds"] += s.dequant_seconds
        _CPU_RUNTIME["matmul_seconds"] += s.matmul_seconds
        _CPU_RUNTIME["layers"] += 1
        _CPU_RUNTIME["last_cpu_experts"] = s.experts
        _CPU_RUNTIME["last_cpu_seconds"] = s.seconds
    else:
        _CPU_RUNTIME["last_cpu_experts"] = 0
        _CPU_RUNTIME["last_cpu_seconds"] = 0.0

    del post_norm, moe_in, router_w, routed, tiered, fp8_entries, q4_entries
    if expert_out is not None:
        del expert_out
    if gate_w is not None:
        del fp8_weights, fp8_scales, gate_w, up_w, down_w, batch_x
    del shared_gate_w, shared_gate, shared_gate_proj, shared_up_proj, shared_down_proj
    del shared_hidden, shared_out, moe_out
    return layer_out, expert_ids, route_weights, shared_gate_value, moe_input_norm

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
            # Do not warm a RAM-resident Q4 expert into VRAM. Its next route
            # should continue to use the CPU path unless it is evicted from RAM.
            with expert_cache.lock:
                if int(expert_id) in expert_cache.q4_entries.get(layer, {}):
                    continue
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
_CPU_RUNTIME = {"experts": 0, "seconds": 0.0, "dequant_seconds": 0.0, "matmul_seconds": 0.0, "layers": 0, "last_cpu_experts": 0, "last_cpu_seconds": 0.0}


def _cache_stats_with_experts(root: Path) -> dict[str, int | float]:
    stats = dict(_ORIGINAL_CACHE_STATS(root))
    cache = _EXPERT_CACHES.get(root.resolve())
    if cache is None:
        return stats
    expert = cache.snapshot()
    predictor = _ROUTING_PREDICTOR.snapshot()
    stats.update({
        "ram_expert_cpu_experts": int(_CPU_RUNTIME["experts"]),
        "ram_expert_cpu_seconds": float(_CPU_RUNTIME["seconds"]),
        "ram_expert_cpu_dequant_seconds": float(_CPU_RUNTIME["dequant_seconds"]),
        "ram_expert_cpu_matmul_seconds": float(_CPU_RUNTIME["matmul_seconds"]),
        "ram_expert_cpu_layers": int(_CPU_RUNTIME["layers"]),
        "ram_expert_cpu_last_experts": int(_CPU_RUNTIME["last_cpu_experts"]),
        "ram_expert_cpu_last_seconds": float(_CPU_RUNTIME["last_cpu_seconds"]),
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
        "routing_predictor_transition_contexts": int(predictor["transition_contexts"]),
        "routing_predictor_observations": int(predictor["observations"]),
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
        f"  ram_expert_cpu: experts={_CPU_RUNTIME['experts']} | "
        f"layers={_CPU_RUNTIME['layers']} | "
        f"total={_CPU_RUNTIME['seconds']:.3f}s | "
        f"dequant={_CPU_RUNTIME['dequant_seconds']:.3f}s | "
        f"matmul={_CPU_RUNTIME['matmul_seconds']:.3f}s | "
        f"last={_CPU_RUNTIME['last_cpu_experts']} experts/{_CPU_RUNTIME['last_cpu_seconds']:.3f}s"
    )
    print(
        f"  routing_predictor: predictions={predictor['predictions']} | "
        f"predicted={predictor['predicted_experts']} | matched={predictor['matched_experts']} | "
        f"precision={predictor['expert_precision']:.2f}% | transitions={predictor['transition_contexts']} | "
        f"observations={predictor['observations']}"
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
    print("expert_cache_compute=FP8-VRAM-GPU|Q4-RAM-CPU")
    print("ram_expert_cpu=enabled-by-default")
    print(f"ram_expert_cpu_workers={os.getenv('QWEN36_CPU_EXPERT_WORKERS', '2')}")
    print("expert_cache_compute_batch=single-gemm-gate-up-plus-batched-down")
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
