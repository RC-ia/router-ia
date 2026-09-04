from __future__ import annotations

"""Opt-in stage profiler for Qwen3.6 token generation.

Enable with ``QWEN36_STAGE_PROFILE=1``.

The profiler is intentionally separate from the production path. It measures
wall-clock time around the major stages that can explain slow token generation:
expert load, Q4 H2D, Q4 dequantization, expert GEMMs, attention, RMSNorm,
lm_head and sampling.

CUDA stages use CUDA events where practical. Storage/cache operations use a
host timer because they may include CPU and file-system work. The profiler
adds synchronization, so timings are diagnostic rather than production-speed
benchmarks.
"""

import os
from collections import defaultdict
from time import perf_counter

import torch
import torch.nn.functional as F

from . import qwen36_attention_cache as attention_cache
from . import qwen36_chat_batch as chat
from . import qwen36_dequant as dequant
from . import qwen36_expert_batch_plan_v2 as planner_v2
from . import qwen36_expert_q4_hierarchy_fixed as hierarchy
from . import qwen36_40layer_loop as base

ENABLED = os.getenv("QWEN36_STAGE_PROFILE", "0").strip().lower() in {"1", "true", "yes", "on"}
TOP_LAYERS = max(int(os.getenv("QWEN36_STAGE_PROFILE_TOP_LAYERS", "8")), 1)


class _Stats:
    def __init__(self) -> None:
        self.total_ms = 0.0
        self.prefill_ms = 0.0
        self.phases = defaultdict(float)
        self.layers = defaultdict(lambda: defaultdict(float))
        self.h2d_bytes = 0
        self.h2d_calls = 0
        self.expert_load_calls = 0
        self.dequant_calls = 0
        self.gemm_calls = 0
        self.norm_calls = 0
        self.lm_head_calls = 0
        self.sample_calls = 0

    def phase(self, name: str, ms: float, layer: int | None = None) -> None:
        self.phases[name] += float(ms)
        if layer is not None:
            self.layers[int(layer)][name] += float(ms)


_STATS: _Stats | None = None

_ORIGINAL_RUN_FORWARD = chat.run_forward_token
_ORIGINAL_SAMPLE_NEXT = chat.sample_next
_ORIGINAL_ATTENTION = attention_cache.step_attention
_ORIGINAL_RMSNORM = base.rmsnorm
_ORIGINAL_TO_CUDA = hierarchy._to_cuda
_ORIGINAL_GET_OR_LOAD = hierarchy._get_or_load
_ORIGINAL_DECODE_Q4 = planner_v2._decode_q4
_ORIGINAL_GATE_UP = planner_v2._gate_up_gemm
_ORIGINAL_DOWN = planner_v2._down_batched
_ORIGINAL_DEQUANT_Q4 = None


def _cuda_elapsed(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    end.synchronize()
    return float(start.elapsed_time(end))


def _measure_cuda(fn):
    if not torch.cuda.is_available():
        t0 = perf_counter()
        value = fn()
        return value, (perf_counter() - t0) * 1000.0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    value = fn()
    end.record()
    return value, _cuda_elapsed(start, end)


def _layer_from_prefix(prefix: str) -> int | None:
    marker = ".layers."
    if marker not in prefix:
        return None
    try:
        return int(prefix.split(marker, 1)[1].split(".", 1)[0])
    except (ValueError, IndexError):
        return None


def _profile_attention(root, layer, x, device):
    stats = _STATS
    if not ENABLED or stats is None:
        return _ORIGINAL_ATTENTION(root, layer, x, device)
    value, elapsed = _measure_cuda(lambda: _ORIGINAL_ATTENTION(root, layer, x, device)) if device == "cuda" else _timed_host(lambda: _ORIGINAL_ATTENTION(root, layer, x, device))
    stats.phase("attention", elapsed, int(layer))
    return value


def _timed_host(fn):
    t0 = perf_counter()
    value = fn()
    return value, (perf_counter() - t0) * 1000.0


def _profile_rmsnorm(x, weight, *args, **kwargs):
    stats = _STATS
    if not ENABLED or stats is None:
        return _ORIGINAL_RMSNORM(x, weight, *args, **kwargs)
    if x.is_cuda:
        value, elapsed = _measure_cuda(lambda: _ORIGINAL_RMSNORM(x, weight, *args, **kwargs))
    else:
        value, elapsed = _timed_host(lambda: _ORIGINAL_RMSNORM(x, weight, *args, **kwargs))
    stats.phase("norm", elapsed)
    stats.norm_calls += 1
    return value


def _profile_get_or_load(cache, store, root, layer, expert_id, layer_prefix):
    stats = _STATS
    if not ENABLED or stats is None:
        return _ORIGINAL_GET_OR_LOAD(cache, store, root, layer, expert_id, layer_prefix)
    t0 = perf_counter()
    value = _ORIGINAL_GET_OR_LOAD(cache, store, root, layer, expert_id, layer_prefix)
    elapsed = (perf_counter() - t0) * 1000.0
    stats.phase("expert_load", elapsed, int(layer))
    stats.expert_load_calls += 1
    return value


def _profile_to_cuda(entry):
    stats = _STATS
    if not ENABLED or stats is None:
        return _ORIGINAL_TO_CUDA(entry)
    t0 = perf_counter()
    value = _ORIGINAL_TO_CUDA(entry)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (perf_counter() - t0) * 1000.0
    # This function receives one complete Q4 expert triplet. Count its total
    # CPU -> GPU transfer as the Q4 H2D stage.
    transferred = 0
    for packed, scale, _shape in entry:
        transferred += int(packed.numel()) * packed.element_size()
        transferred += int(scale.numel()) * scale.element_size()
    stats.phase("q4_h2d", elapsed)
    stats.h2d_bytes += transferred
    stats.h2d_calls += 1
    return value


def _profile_decode_q4(entries):
    stats = _STATS
    if not ENABLED or stats is None:
        return _ORIGINAL_DECODE_Q4(entries)
    if not entries:
        return _ORIGINAL_DECODE_Q4(entries)
    t0 = perf_counter()
    result = _ORIGINAL_DECODE_Q4(entries)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (perf_counter() - t0) * 1000.0
    stats.phase("q4_dequant")
    stats.phases["q4_dequant"] += 0.0
    # Re-time was already collected as host wall time including CUDA sync.
    stats.phases["q4_dequant"] += elapsed
    stats.dequant_calls += len(entries) * 3
    return result


def _profile_gate_up(gate_w, up_w, x):
    stats = _STATS
    if not ENABLED or stats is None:
        return _ORIGINAL_GATE_UP(gate_w, up_w, x)
    value, elapsed = _measure_cuda(lambda: _ORIGINAL_GATE_UP(gate_w, up_w, x)) if gate_w.is_cuda else _timed_host(lambda: _ORIGINAL_GATE_UP(gate_w, up_w, x))
    stats.phase("gemm")
    stats.phases["gemm"] += elapsed
    stats.gemm_calls += 1
    return value


def _profile_down(down_w, hidden):
    stats = _STATS
    if not ENABLED or stats is None:
        return _ORIGINAL_DOWN(down_w, hidden)
    value, elapsed = _measure_cuda(lambda: _ORIGINAL_DOWN(down_w, hidden)) if down_w.is_cuda else _timed_host(lambda: _ORIGINAL_DOWN(down_w, hidden))
    stats.phases["gemm"] += elapsed
    stats.gemm_calls += 1
    return value


def _profile_linear(input_tensor, weight, bias=None):
    stats = _STATS
    if not ENABLED or stats is None:
        return F._router_ia_original_linear(input_tensor, weight, bias)  # type: ignore[attr-defined]
    is_lm_head = weight is not None and weight.ndim == 2 and int(weight.shape[0]) == 248320 and int(weight.shape[1]) == 2048
    if not is_lm_head:
        return F._router_ia_original_linear(input_tensor, weight, bias)  # type: ignore[attr-defined]
    if input_tensor.is_cuda:
        value, elapsed = _measure_cuda(lambda: F._router_ia_original_linear(input_tensor, weight, bias))
    else:
        value, elapsed = _timed_host(lambda: F._router_ia_original_linear(input_tensor, weight, bias))
    stats.phase("lm_head", elapsed)
    stats.lm_head_calls += 1
    return value


def _profile_sample(logits, temperature, top_k):
    stats = _STATS
    if not ENABLED or stats is None:
        return _ORIGINAL_SAMPLE_NEXT(logits, temperature, top_k)
    if logits.is_cuda:
        value, elapsed = _measure_cuda(lambda: _ORIGINAL_SAMPLE_NEXT(logits, temperature, top_k))
    else:
        value, elapsed = _timed_host(lambda: _ORIGINAL_SAMPLE_NEXT(logits, temperature, top_k))
    stats.phase("sampling", elapsed)
    stats.sample_calls += 1
    return value


def _report(stats: _Stats) -> None:
    total = max(stats.total_ms, 1e-6)
    print("  stage_profile:")
    ordered = [
        "expert_load",
        "q4_h2d",
        "q4_dequant",
        "gemm",
        "attention",
        "norm",
        "lm_head",
        "sampling",
    ]
    for name in ordered:
        ms = float(stats.phases.get(name, 0.0))
        print(f"    {name}={ms / 1000.0:.4f}s | {ms / total * 100.0:.1f}%")
    accounted = sum(float(stats.phases.get(name, 0.0)) for name in ordered)
    other = max(stats.total_ms - accounted, 0.0)
    print(f"    other={other / 1000.0:.4f}s | {other / total * 100.0:.1f}%")
    print(
        f"    q4_h2d_transfer={stats.h2d_bytes / 1024**2:.1f}MiB | "
        f"q4_h2d_calls={stats.h2d_calls} | dequant_calls={stats.dequant_calls} | "
        f"gemm_calls={stats.gemm_calls} | norm_calls={stats.norm_calls} | "
        f"lm_head_calls={stats.lm_head_calls} | sample_calls={stats.sample_calls}"
    )
    by_layer = []
    for layer, values in stats.layers.items():
        total_layer = float(sum(values.values()))
        by_layer.append((total_layer, layer, values))
    by_layer.sort(reverse=True)
    if by_layer:
        print("    top_layers:")
        for total_layer, layer, values in by_layer[:TOP_LAYERS]:
            dominant_name, dominant_ms = max(values.items(), key=lambda item: item[1])
            print(
                f"      layer={layer:02d} | total={total_layer / 1000.0:.4f}s | "
                f"dominant={dominant_name} {dominant_ms / 1000.0:.4f}s"
            )


def _profile_run_forward(root, token_id, final_norm, lm_head, final_norm_name, lm_head_name, device, advance_state=True):
    global _STATS
    if not ENABLED:
        return _ORIGINAL_RUN_FORWARD(root, token_id, final_norm, lm_head, final_norm_name, lm_head_name, device, advance_state)
    _STATS = _Stats()
    t0 = perf_counter()
    try:
        result = _ORIGINAL_RUN_FORWARD(root, token_id, final_norm, lm_head, final_norm_name, lm_head_name, device, advance_state)
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        _STATS.total_ms = (perf_counter() - t0) * 1000.0
        _report(_STATS)
        return result
    finally:
        _STATS = None


if ENABLED:
    # Keep the profiler at the end of the runner import chain so these are the
    # already-patched Q4 hierarchy/planner functions actually used by inference.
    attention_cache.step_attention = _profile_attention
    base.rmsnorm = _profile_rmsnorm
    hierarchy._get_or_load = _profile_get_or_load
    hierarchy._to_cuda = _profile_to_cuda
    planner_v2._decode_q4 = _profile_decode_q4
    planner_v2._gate_up_gemm = _profile_gate_up
    planner_v2._down_batched = _profile_down

    F._router_ia_original_linear = F.linear  # type: ignore[attr-defined]
    F.linear = _profile_linear  # type: ignore[assignment]
    chat.sample_next = _profile_sample
    chat.run_forward_token = _profile_run_forward

    print(
        f"stage_profiler=enabled | top_layers={TOP_LAYERS} | "
        "warning=diagnostic-only-cuda-sync"
    )
