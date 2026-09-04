from __future__ import annotations

"""Opt-in detailed profiler for the official Qwen3.6 token path.

Enable with ``QWEN36_PROFILE=1``. Profiling synchronizes major phases and can
therefore reduce throughput; it is intended to identify bottlenecks, not to
measure production performance.
"""

import os
from dataclasses import dataclass, field
from threading import Lock
from time import perf_counter

import torch

from . import qwen36_attention_cache as attention_cache
from . import qwen36_chat_batch as chat

ENABLED = os.getenv("QWEN36_PROFILE", "0").strip().lower() in {"1", "true", "yes", "on"}
TOP_LAYERS = max(int(os.getenv("QWEN36_PROFILE_TOP_LAYERS", "5")), 1)


@dataclass
class _LayerTiming:
    attention_ms: float = 0.0
    moe_ms: float = 0.0
    prefetch_ms: float = 0.0
    expert_gpu_ms: float = 0.0
    calls: int = 0


@dataclass
class _TokenProfile:
    started: float
    layer: dict[int, _LayerTiming] = field(default_factory=dict)
    total_event: torch.cuda.Event | None = None

    def layer_for(self, index: int) -> _LayerTiming:
        timing = self.layer.get(int(index))
        if timing is None:
            timing = _LayerTiming()
            self.layer[int(index)] = timing
        return timing


_ACTIVE: _TokenProfile | None = None
_LOCK = Lock()
_ORIGINAL_STEP_ATTENTION = attention_cache.step_attention
_ORIGINAL_MOE = chat.batched_moe_step
_ORIGINAL_PREFETCH = chat._warm_expert_raw_cache
_ORIGINAL_EXPERT_TRIPLET = chat._expert_projection_triplet
_ORIGINAL_RUN_FORWARD = chat.run_forward_token


def _layer_from_prefix(layer_prefix: str) -> int | None:
    marker = ".layers."
    if marker not in layer_prefix:
        return None
    try:
        return int(layer_prefix.split(marker, 1)[1].split(".", 1)[0])
    except (ValueError, IndexError):
        return None


def _cuda_event_pair():
    if not torch.cuda.is_available():
        return None, None
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def _begin() -> None:
    global _ACTIVE
    with _LOCK:
        profile = _TokenProfile(started=perf_counter())
        if torch.cuda.is_available():
            profile.total_event = torch.cuda.Event(enable_timing=True)
            profile.total_event.record()
        _ACTIVE = profile


def _end_total(profile: _TokenProfile, device: str) -> float:
    if device == "cuda" and torch.cuda.is_available() and profile.total_event is not None:
        finish = torch.cuda.Event(enable_timing=True)
        finish.record()
        torch.cuda.synchronize()
        return float(profile.total_event.elapsed_time(finish))
    return (perf_counter() - profile.started) * 1000.0


def _print(profile: _TokenProfile, total_ms: float) -> None:
    layers = profile.layer
    attn_ms = sum(item.attention_ms for item in layers.values())
    moe_ms = sum(item.moe_ms for item in layers.values())
    prefetch_ms = sum(item.prefetch_ms for item in layers.values())
    expert_gpu_ms = sum(item.expert_gpu_ms for item in layers.values())
    accounted = attn_ms + moe_ms
    other_ms = max(total_ms - accounted, 0.0)

    bottlenecks = []
    for layer, timing in layers.items():
        dominant = "moe" if timing.moe_ms >= timing.attention_ms else "attention"
        value = max(timing.moe_ms, timing.attention_ms)
        bottlenecks.append((value, layer, dominant, timing))
    bottlenecks.sort(reverse=True)

    print(
        f"  profile total={total_ms / 1000.0:.3f}s | "
        f"attention={attn_ms / 1000.0:.3f}s | "
        f"moe={moe_ms / 1000.0:.3f}s | "
        f"other={other_ms / 1000.0:.3f}s"
    )
    print(
        f"  profile expert_prefetch_wall={prefetch_ms / 1000.0:.3f}s | "
        f"expert_triplet_gpu_sum={expert_gpu_ms / 1000.0:.3f}s"
    )

    if bottlenecks:
        print("  profile top_layers:")
        for _, layer, dominant, timing in bottlenecks[:TOP_LAYERS]:
            print(
                f"    layer={layer:02d} | dominant={dominant:<9} | "
                f"attention={timing.attention_ms:.1f}ms | moe={timing.moe_ms:.1f}ms | "
                f"prefetch={timing.prefetch_ms:.1f}ms | "
                f"expert_gpu_sum={timing.expert_gpu_ms:.1f}ms"
            )

    dominant_phase = max(
        ((attn_ms, "attention"), (moe_ms, "moe"), (other_ms, "other")),
        key=lambda item: item[0],
    )
    print(
        f"  profile bottleneck={dominant_phase[1]} "
        f"{dominant_phase[0] / 1000.0:.3f}s "
        f"({dominant_phase[0] / max(total_ms, 1e-9) * 100.0:.1f}%)"
    )


def _profile_attention(root, layer, x, device):
    profile = _ACTIVE
    if not ENABLED or profile is None:
        return _ORIGINAL_STEP_ATTENTION(root, layer, x, device)

    start, end = _cuda_event_pair()
    if device == "cuda" and start is not None and end is not None:
        start.record()
        result = _ORIGINAL_STEP_ATTENTION(root, layer, x, device)
        end.record()
        end.synchronize()
        elapsed = float(start.elapsed_time(end))
    else:
        t0 = perf_counter()
        result = _ORIGINAL_STEP_ATTENTION(root, layer, x, device)
        elapsed = (perf_counter() - t0) * 1000.0
    profile.layer_for(layer).attention_ms += elapsed
    return result


def _profile_moe(root, layer, residual, top_k, device):
    profile = _ACTIVE
    if not ENABLED or profile is None:
        return _ORIGINAL_MOE(root, layer, residual, top_k, device)

    start, end = _cuda_event_pair()
    timing = profile.layer_for(layer)
    timing.calls += 1
    if device == "cuda" and start is not None and end is not None:
        start.record()
        result = _ORIGINAL_MOE(root, layer, residual, top_k, device)
        end.record()
        end.synchronize()
        timing.moe_ms += float(start.elapsed_time(end))
    else:
        t0 = perf_counter()
        result = _ORIGINAL_MOE(root, layer, residual, top_k, device)
        timing.moe_ms += (perf_counter() - t0) * 1000.0
    return result


def _profile_prefetch(root, layer_prefix, expert_ids):
    profile = _ACTIVE
    if not ENABLED or profile is None:
        return _ORIGINAL_PREFETCH(root, layer_prefix, expert_ids)

    layer = _layer_from_prefix(layer_prefix)
    t0 = perf_counter()
    result = _ORIGINAL_PREFETCH(root, layer_prefix, expert_ids)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (perf_counter() - t0) * 1000.0
    if layer is not None:
        profile.layer_for(layer).prefetch_ms += elapsed
    return result


def _profile_expert_triplet(root, layer_prefix, expert_id, device):
    profile = _ACTIVE
    if not ENABLED or profile is None or device != "cuda" or not torch.cuda.is_available():
        return _ORIGINAL_EXPERT_TRIPLET(root, layer_prefix, expert_id, device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = _ORIGINAL_EXPERT_TRIPLET(root, layer_prefix, expert_id, device)
    end.record()
    layer = _layer_from_prefix(layer_prefix)
    if layer is not None:
        profile.layer_for(layer).expert_gpu_ms += float(start.elapsed_time(end))
    return result


def _profile_run_forward(*args, **kwargs):
    global _ACTIVE
    if not ENABLED:
        return _ORIGINAL_RUN_FORWARD(*args, **kwargs)

    _begin()
    device = str(kwargs.get("device", args[6] if len(args) > 6 else "cpu")).lower()
    try:
        result = _ORIGINAL_RUN_FORWARD(*args, **kwargs)
        profile = _ACTIVE
        if profile is not None:
            total_ms = _end_total(profile, device)
            _print(profile, total_ms)
        return result
    finally:
        with _LOCK:
            _ACTIVE = None


if ENABLED:
    attention_cache.step_attention = _profile_attention
    chat.batched_moe_step = _profile_moe
    chat._warm_expert_raw_cache = _profile_prefetch
    chat._expert_projection_triplet = _profile_expert_triplet
    chat.run_forward_token = _profile_run_forward
    print(
        f"profiler=enabled | top_layers={TOP_LAYERS} | "
        "warning=profiling synchronizes CUDA and is slower than normal"
    )
