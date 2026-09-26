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
from . import qwen36_40layer_loop as base
from .qwen36_expert_cache import RoutedExpertCache


chat = None
_EXPERT_CACHES: dict[Path, RoutedExpertCache] = {}
_ORIGINAL_EXPERT_TRIPLET = None
_ORIGINAL_CACHE_STATS = None
_ORIGINAL_PRINT_CACHE = None
_ORIGINAL_BATCHED_MOE_STEP = None
_ORIGINAL_RUN_GENERATED_TOKEN = None
_ORIGINAL_RUN_FORWARD_TOKEN = None
_INSTALLED_CHAT_MODULES: set[int] = set()


class ExpertCrossLayerPredictor:
    """Predict current-layer experts from earlier layers of the same token."""

    def __init__(self, top_n: int = 4, min_observations: int = 1) -> None:
        self.top_n = max(int(top_n), 1)
        self.min_observations = max(int(min_observations), 1)
        self._transitions: dict[tuple[int, int, int], Counter[int]] = defaultdict(Counter)
        self._observations = 0
        self._predictions = 0
        self._predicted_experts = 0
        self._matched_experts = 0
        self._lock = Lock()

    def observe(
        self,
        layer: int,
        expert_ids: list[int],
        context_routes: dict[int, tuple[int, ...]],
    ) -> None:
        target_layer = int(layer)
        current = tuple(dict.fromkeys(int(x) for x in expert_ids))
        if not current or not context_routes:
            return
        with self._lock:
            for source_layer, source_ids in context_routes.items():
                source_layer = int(source_layer)
                if source_layer >= target_layer:
                    continue
                for source_expert in dict.fromkeys(int(x) for x in source_ids):
                    counter = self._transitions[
                        (target_layer, source_layer, source_expert)
                    ]
                    for current_expert in current:
                        counter[current_expert] += 1
            self._observations += 1

    def predict(
        self,
        layer: int,
        context_routes: dict[int, tuple[int, ...]],
    ) -> list[int]:
        target_layer = int(layer)
        aggregate: Counter[int] = Counter()
        with self._lock:
            for source_layer, source_ids in context_routes.items():
                source_layer = int(source_layer)
                if source_layer >= target_layer:
                    continue
                distance = max(target_layer - source_layer, 1)
                layer_weight = 1.0 / (distance ** 0.5)
                for source_expert in dict.fromkeys(int(x) for x in source_ids):
                    counter = self._transitions.get(
                        (target_layer, source_layer, source_expert)
                    )
                    if not counter or sum(counter.values()) < self.min_observations:
                        continue
                    for expert, count in counter.most_common(self.top_n):
                        aggregate[expert] += float(count) * layer_weight

            predicted = [expert for expert, _ in aggregate.most_common(self.top_n)]
            if predicted:
                self._predictions += 1
                self._predicted_experts += len(predicted)
            return predicted

    def record_matches(self, predicted: list[int], actual: list[int]) -> None:
        actual_set = {int(x) for x in actual}
        with self._lock:
            self._matched_experts += sum(
                1 for expert in predicted if int(expert) in actual_set
            )

    def snapshot(self) -> dict[str, int | float]:
        with self._lock:
            precision = (
                self._matched_experts / self._predicted_experts * 100.0
                if self._predicted_experts
                else 0.0
            )
            return {
                "predictions": self._predictions,
                "predicted_experts": self._predicted_experts,
                "matched_experts": self._matched_experts,
                "expert_precision": precision,
                "top_n": self.top_n,
                "min_observations": self.min_observations,
                "transition_contexts": len(self._transitions),
                "observations": self._observations,
            }

_ROUTING_PREDICTOR = ExpertCrossLayerPredictor(
    top_n=max(int(os.getenv("QWEN36_ROUTER_PREDICT_TOP_N", "2")), 1),
    min_observations=max(int(os.getenv("QWEN36_ROUTING_MIN_OBSERVATIONS", "2")), 1),
)
_CURRENT_TOKEN_ROUTES: dict[int, tuple[int, ...]] = {}
_CPU_POOL = ThreadPoolExecutor(
    max_workers=max(int(os.getenv("QWEN36_CPU_EXPERT_POOL_WORKERS", "2")), 1),
    thread_name_prefix="ram-experts",
)
_PREFETCH_POOL = ThreadPoolExecutor(
    max_workers=max(int(os.getenv("QWEN36_ROUTER_PREFETCH_WORKERS", "1")), 1),
    thread_name_prefix="expert-prefetch",
)
_PROMOTION_POOL = ThreadPoolExecutor(
    max_workers=max(int(os.getenv("QWEN36_ROUTER_PROMOTION_WORKERS", "2")), 1),
    thread_name_prefix="expert-promote",
)
_PREFETCH_MAX_PENDING = max(int(os.getenv("QWEN36_ROUTER_PREFETCH_MAX_PENDING", "6")), 1)
_PREFETCH_PENDING = set()
_PREFETCH_LOCK = Lock()
_COLLECT_LAYER_STATS = os.getenv("QWEN36_LAYER_STATS", "0") == "1"
_CPU_RUNTIME = {
    "experts": 0,
    "seconds": 0.0,
    "dequant_seconds": 0.0,
    "matmul_seconds": 0.0,
    "layers": 0,
    "last_cpu_experts": 0,
    "last_cpu_seconds": 0.0,
}


def _expert_cache(root: Path) -> RoutedExpertCache:
    key = root.resolve()
    cache = _EXPERT_CACHES.get(key)
    if cache is None:
        store = cached._store(root)
        cache = RoutedExpertCache(
            cached.CACHE_BUDGET_BYTES,
            shared_budget=store.shared_ram_budget,
        )
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


def _decoded_expert_triplet(
    root: Path,
    layer: int,
    expert_id: int,
    entry,
    store,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reuse FP16 expert matrices from the bounded VRAM stream cache."""
    decoded = []
    for projection in range(3):
        key = f"expert.fp16.{int(layer)}.{int(expert_id)}.{projection}"
        cached_weight = store.vram_cache.get_stream(key)
        if cached_weight is None:
            weight, scale = entry[projection]
            cached_weight = dequant.dequantize_fp8_blockwise(
                weight, scale
            ).to(dtype=torch.float16)
            store.vram_cache.put_stream(key, cached_weight)
        decoded.append(cached_weight)
    return decoded[0], decoded[1], decoded[2]


def _batched_moe_step_gpu(root: Path, layer: int, residual: torch.Tensor, top_k: int, device: str):
    """Mixed MoE execution: FP8/VRAM experts on GPU, Q4/RAM experts on CPU."""
    if device != "cuda":
        return _ORIGINAL_BATCHED_MOE_STEP(root, layer, residual, top_k, device)

    from . import qwen36_cpu_expert as cpu_expert
    prefix = base.layer_prefix(layer)
    post_norm = base.load_layer_weight(root, layer, "post_attention_layernorm.weight", device)
    moe_in = base.rmsnorm(residual, post_norm).reshape(1, base.HIDDEN).float()

    predicted_current = _ROUTING_PREDICTOR.predict(
        layer, _CURRENT_TOKEN_ROUTES
    )

    # Prefetch speculative candidates before resolving the current router
    # result so RAM staging overlaps router/SSD work.
    prefetch_cache = _expert_cache(root)
    prefetch_store = cached._store(root)
    for predicted_expert in predicted_current:
        with _PREFETCH_LOCK:
            done_prefetch = [
                future for future in tuple(_PREFETCH_PENDING) if future.done()
            ]
            for future in done_prefetch:
                _PREFETCH_PENDING.discard(future)
            if len(_PREFETCH_PENDING) >= _PREFETCH_MAX_PENDING:
                break
            future = _PREFETCH_POOL.submit(
                prefetch_cache.prefetch_expert_to_ram,
                prefetch_store,
                layer,
                int(predicted_expert),
                prefix,
            )
            _PREFETCH_PENDING.add(future)

    router_w = base.load_layer_weight(root, layer, "mlp.gate.weight", device).float()
    routed = base.route(moe_in.reshape(-1), router_w, top_k=top_k)
    expert_ids = [int(v) for v in routed.expert_ids.detach().cpu().tolist()]
    route_weights = [float(v) for v in routed.weights.detach().cpu().tolist()]

    _ROUTING_PREDICTOR.observe(
        layer, expert_ids, _CURRENT_TOKEN_ROUTES
    )
    if predicted_current:
        _ROUTING_PREDICTOR.record_matches(predicted_current, expert_ids)
    _CURRENT_TOKEN_ROUTES[int(layer)] = tuple(expert_ids)

    tiered = _expert_cache(root).get_or_load_batch_tiered(
        cached._store(root), layer, expert_ids, prefix
    )
        with _PREFETCH_LOCK:
            done_prefetch = [future for future in tuple(_PREFETCH_PENDING) if future.done()]
            for future in done_prefetch:
                _PREFETCH_PENDING.discard(future)
            if len(_PREFETCH_PENDING) >= _PREFETCH_MAX_PENDING:
                break
            future = _PREFETCH_POOL.submit(
                prefetch_cache.prefetch_expert_to_ram,
                prefetch_store,
                layer,
                int(predicted_expert),
                prefix,
            )
            _PREFETCH_PENDING.add(future)

    fp8_entries = [
        (expert_ids[pos], entry)
        for pos, (tier, entry) in enumerate(tiered)
        if tier == "fp8"
    ]
    q4_entries = [entry for tier, entry in tiered if tier == "q4"]
    routed_sum = torch.zeros_like(moe_in)

    cpu_future = None
    cpu_pool = None
    if q4_entries:
        workers_raw = int(os.getenv("QWEN36_CPU_EXPERT_WORKERS", "2"))
        workers = min(max(workers_raw, 1), len(q4_entries))
        # Start RAM-resident experts first so CPU execution overlaps the GPU path.
        cpu_future = _CPU_POOL.submit(
            cpu_expert.run_q4_expert_batch_cpu, q4_entries, moe_in, workers=workers
        )

    expert_out = None
    fp8_weights = fp8_scales = gate_w = up_w = down_w = batch_x = None
    if fp8_entries:
        decoded = [
            _decoded_expert_triplet(
                root,
                layer,
                int(expert_id),
                entry,
                cached._store(root),
            )
            for expert_id, entry in fp8_entries
        ]
        gate_w = torch.stack([item[0] for item in decoded], dim=0)
        up_w = torch.stack([item[1] for item in decoded], dim=0)
        down_w = torch.stack([item[2] for item in decoded], dim=0)
        batch_x = moe_in.reshape(-1).to(dtype=torch.float16)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            gate, up = _route_gate_up_single_gemm(gate_w, up_w, batch_x)
            hidden = F.silu(gate) * up
            expert_out = _route_projection_batched(down_w, hidden, len(fp8_entries))

    cpu_result = cpu_future.result() if cpu_future is not None else None
    if cpu_result is not None:
        promote_cache = _expert_cache(root)
        promote_store = cached._store(root)
        for route_pos, (tier, _) in enumerate(tiered):
            if tier == "q4":
                _PROMOTION_POOL.submit(
                    promote_cache.promote_q4_to_vram,
                    promote_store,
                    layer,
                    int(expert_ids[route_pos]),
                    prefix,
                )

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
    if _COLLECT_LAYER_STATS:
        shared_gate_value = float(shared_gate.float().item())
        moe_input_norm = float(torch.linalg.vector_norm(moe_in).item())
    else:
        shared_gate_value = 0.0
        moe_input_norm = 0.0

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

def _run_forward_token_with_predictor(
    root: Path,
    token_id: int,
    final_norm: torch.Tensor,
    lm_head: torch.Tensor,
    final_norm_name: str,
    lm_head_name: str,
    device: str,
    advance_state: bool = True,
):
    global _CURRENT_TOKEN_ROUTES
    _CURRENT_TOKEN_ROUTES = {}
    try:
        return _ORIGINAL_RUN_FORWARD_TOKEN(
            root,
            token_id,
            final_norm,
            lm_head,
            final_norm_name,
            lm_head_name,
            device,
            advance_state,
        )
    finally:
        _CURRENT_TOKEN_ROUTES = {}


def _run_generated_token_with_predictor(
    root: Path,
    token_id: int,
    final_norm: torch.Tensor,
    lm_head: torch.Tensor,
    final_norm_name: str,
    lm_head_name: str,
    device: str,
    sampling_top_k: int,
    temperature: float,
):
    return _ORIGINAL_RUN_GENERATED_TOKEN(
        root,
        token_id,
        final_norm,
        lm_head,
        final_norm_name,
        lm_head_name,
        device,
        sampling_top_k,
        temperature,
    )


def _cache_stats_with_experts(root: Path) -> dict[str, int | float]:
    stats = dict(_ORIGINAL_CACHE_STATS(root))
    cache = _EXPERT_CACHES.get(root.resolve())
    if cache is None:
        cache = _expert_cache(root)
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
        "expert_cache_host_ram_bytes": int(expert["host_ram_bytes"]),
        "expert_cache_host_ram_budget": int(expert["host_ram_budget"]),
        "expert_cache_host_ram_utilization": float(expert["host_ram_utilization"]),
        "shared_ram_bytes": int(cached._store(root).shared_ram_budget.snapshot()["used_bytes"]),
        "shared_ram_budget": int(cached._store(root).shared_ram_budget.snapshot()["total_bytes"]),
        "expert_cache_q4_promotions": int(expert["q4_promotions"]),
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
        cache = _expert_cache(root)
    expert = cache.snapshot()
    predictor = _ROUTING_PREDICTOR.snapshot()
    print(
        f"  expert_cache: fp8_vram_entries={expert['warm_items']} | "
        f"fp8_vram={expert['bytes'] / 1024**2:.1f}/{expert['budget_bytes'] / 1024**2:.1f} MiB | "
        f"q4_ram_entries={expert['cold_items']} | "
        f"q4_ram={expert['q4_ram_bytes'] / 1024**2:.1f} MiB | "
        f"predicted_ram={expert['predicted_ram_bytes'] / 1024**2:.1f} MiB | "
        f"host_ram={expert['host_ram_bytes'] / 1024**2:.1f}/{expert['host_ram_budget'] / 1024**2:.1f} MiB | "
        f"hit_rate={expert['hit_rate']:.2f}% | hits={expert['hits']} | misses={expert['misses']} | loads={expert['loads']}"
    )
    print(
        f"    tiers: FP8=VRAM:{expert['warm_items']} | Q4=RAM:{expert['cold_items']} | "
        f"hits FP8={expert['fp8_hits']} Q4={expert['q4_hits']} | "
        f"GPU compressions FP8>Q4={expert['fp8_to_q4']} | Q4 RAM evictions={expert['q4_ram_evictions']} | "
        f"Q4->VRAM promotions={expert['q4_promotions']}"
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


def install(target_chat_module) -> None:
    """Install the fused runtime onto the exact chat module being executed.

    This avoids the __main__ vs router_ia.qwen36_chat_batch duplicate-module
    trap when qwen36_chat_batch.py is launched with python -m.
    """
    global chat
    global _ORIGINAL_EXPERT_TRIPLET, _ORIGINAL_CACHE_STATS
    global _ORIGINAL_PRINT_CACHE, _ORIGINAL_BATCHED_MOE_STEP
    global _ORIGINAL_RUN_GENERATED_TOKEN, _ORIGINAL_RUN_FORWARD_TOKEN

    module_id = id(target_chat_module)
    if module_id in _INSTALLED_CHAT_MODULES:
        return

    chat = target_chat_module
    _ORIGINAL_EXPERT_TRIPLET = chat._expert_projection_triplet
    _ORIGINAL_CACHE_STATS = chat.cache_stats
    _ORIGINAL_PRINT_CACHE = chat.print_cache
    _ORIGINAL_BATCHED_MOE_STEP = chat.batched_moe_step
    _ORIGINAL_RUN_GENERATED_TOKEN = chat.run_generated_token
    _ORIGINAL_RUN_FORWARD_TOKEN = chat.run_forward_token

    chat._expert_projection_triplet = _cached_expert_projection_triplet
    chat.batched_moe_step = _batched_moe_step_gpu
    chat.run_forward_token = _run_forward_token_with_predictor
    chat.run_generated_token = _run_generated_token_with_predictor
    chat.cache_stats = _cache_stats_with_experts
    chat.print_cache = _print_cache_with_experts
    _INSTALLED_CHAT_MODULES.add(module_id)


def main() -> None:
    from . import qwen36_chat_batch as chat_module

    install(chat_module)
    cache = _expert_cache(Path("."))
    print("expert_cache=complete-layer-expert")
    print("expert_cache_key=(layer,expert)")
    print("expert_cache_policy=per-layer-4fp8-vram-3q4-ram")
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
    print("routing_predictor=expert-cross-layer")
    print("routing_predictor_policy=same-token-previous-layers-to-current-layer")
    print(f"routing_predictor_top_n={_ROUTING_PREDICTOR.top_n}")
    print("routing_predictor_prefetch=async-RAM-first")
    print("q4_to_vram_promotion=async-after-actual-use")
    print(f"layer_stats={'enabled' if _COLLECT_LAYER_STATS else 'disabled'}")
    print(f"expert_cache_total_slots={cache.total_slots}")
    print(f"expert_cache_slots_per_layer={cache.slots_per_layer}")
    print(f"expert_cache_fp8_slots_per_layer={cache.fp8_slots}")
    print(f"expert_cache_q4_ram_slots_per_layer={cache.q4_slots}")
    chat.main()


if __name__ == "__main__":
    main()
