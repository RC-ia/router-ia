from __future__ import annotations

"""Asynchronous next-layer expert prefetch scheduler for Qwen3.6.

The scheduler does not change model math. It speculatively prepares hot experts
for the next layer on a dedicated CUDA stream while the current layer computes.
The compute stream waits only for an expert that is actually requested.
"""

import math
import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

import torch

from . import qwen36_adaptive_experts as adaptive
from . import qwen36_chat_batch as chat
from . import qwen36_official_optimizations as official

ENABLED = os.getenv("QWEN36_ASYNC_EXPERTS", "1").strip().lower() not in {"0", "false", "no", "off"}
LOOKAHEAD_EXPERTS = max(int(os.getenv("QWEN36_ASYNC_LOOKAHEAD_EXPERTS", "2")), 1)
MAX_WORKERS = max(int(os.getenv("QWEN36_ASYNC_WORKERS", "2")), 1)


@dataclass
class _Pending:
    future: Future
    event: torch.cuda.Event | None
    layer: int
    expert_id: int


@dataclass
class _Stats:
    scheduled: int = 0
    completed: int = 0
    waited: int = 0
    overlapped: int = 0
    skipped_fp8: int = 0
    skipped_pending: int = 0
    errors: int = 0
    current_layer_prefetch: int = 0
    lookahead_prefetch: int = 0
    by_layer: dict[int, int] = field(default_factory=dict)


@dataclass
class _State:
    stream: torch.cuda.Stream
    pool: ThreadPoolExecutor
    pending: dict[tuple[int, int], _Pending] = field(default_factory=dict)
    stats: _Stats = field(default_factory=_Stats)


_STATES: dict[Path, _State] = {}
_STATES_LOCK = Lock()
_ORIGINAL_MOE = chat.batched_moe_step
_ORIGINAL_TRIPLET = chat._expert_projection_triplet
_ORIGINAL_WARM = chat._warm_expert_raw_cache
_ORIGINAL_CACHE_STATS = chat.cache_stats
_ORIGINAL_PRINT_CACHE = chat.print_cache


def _state(root: Path) -> _State | None:
    if not ENABLED or not torch.cuda.is_available():
        return None
    key = root.resolve()
    with _STATES_LOCK:
        value = _STATES.get(key)
        if value is None:
            value = _State(
                stream=torch.cuda.Stream(device="cuda"),
                pool=ThreadPoolExecutor(
                    max_workers=MAX_WORKERS,
                    thread_name_prefix="router-ia-prefetch",
                ),
            )
            _STATES[key] = value
        return value


def _layer_from_prefix(layer_prefix: str) -> int | None:
    marker = ".layers."
    if marker not in layer_prefix:
        return None
    try:
        return int(layer_prefix.split(marker, 1)[1].split(".", 1)[0])
    except (ValueError, IndexError):
        return None


def _hot_candidates(root: Path, layer: int) -> list[int]:
    policy = adaptive._POLICIES.get(root.resolve())
    expert_cache = official._EXPERT_CACHES.get(root.resolve())
    state = _state(root)
    if policy is None or expert_cache is None or state is None:
        return []

    with policy.lock:
        scored: list[tuple[float, int]] = []
        for (tracked_layer, expert_id), item in policy.usage.items():
            if tracked_layer != int(layer) or not item.accesses:
                continue
            age = max(policy.tick - item.last_tick, 0)
            recency = 1.0 / (1.0 + age / float(adaptive.RECENCY_WINDOW))
            score = 1.5 * math.log1p(item.accesses) + 2.0 * recency
            scored.append((score, int(expert_id)))

    scored.sort(reverse=True)
    selected: list[int] = []
    with expert_cache.lock:
        fp8_bank = expert_cache.fp8_entries.get(int(layer), {})
        pending = {
            expert
            for (pending_layer, expert) in state.pending
            if pending_layer == int(layer)
        }
        for score, expert_id in scored:
            if score < adaptive.PREFETCH_SCORE:
                break
            if expert_id in fp8_bank or expert_id in pending:
                continue
            selected.append(expert_id)
            if len(selected) >= LOOKAHEAD_EXPERTS:
                break
    return selected


def _do_prefetch(root: Path, layer: int, layer_prefix: str, expert_id: int, state: _State) -> None:
    cache = official._expert_cache(root)
    store = __import__("router_ia.qwen36_cached_loop", fromlist=["_store"])._store(root)
    with torch.cuda.stream(state.stream):
        cache.prefetch_expert_raw(store, layer_prefix, int(expert_id))


def _cleanup_done(state: _State) -> None:
    done_keys: list[tuple[int, int]] = []
    for key, pending in state.pending.items():
        if not pending.future.done():
            continue
        if pending.event is not None and not pending.event.query():
            # The Python worker is finished, but the CUDA stream may still be
            # copying/staging the tensors. Keep the dependency alive.
            continue
        done_keys.append(key)

    for key in done_keys:
        pending = state.pending.pop(key)
        try:
            pending.future.result()
            state.stats.completed += 1
        except Exception:
            state.stats.errors += 1


def _schedule(root: Path, layer: int, expert_id: int, source: str) -> None:
    state = _state(root)
    if state is None or layer >= 40:
        return
    key = (int(layer), int(expert_id))
    _cleanup_done(state)
    if key in state.pending:
        state.stats.skipped_pending += 1
        return

    expert_cache = official._expert_cache(root)
    with expert_cache.lock:
        if int(expert_id) in expert_cache.fp8_entries.get(int(layer), {}):
            state.stats.skipped_fp8 += 1
            return

    prefix = chat.base.layer_prefix(int(layer))
    event = torch.cuda.Event(enable_timing=True)

    def job() -> None:
        try:
            _do_prefetch(root, int(layer), prefix, int(expert_id), state)
            event.record(state.stream)
        except Exception:
            state.stats.errors += 1
            raise

    future = state.pool.submit(job)
    state.pending[key] = _Pending(future, event, int(layer), int(expert_id))
    state.stats.scheduled += 1
    state.stats.by_layer[int(layer)] = state.stats.by_layer.get(int(layer), 0) + 1
    if source == "current":
        state.stats.current_layer_prefetch += 1
    else:
        state.stats.lookahead_prefetch += 1


def _lookahead(root: Path, current_layer: int) -> None:
    next_layer = int(current_layer) + 1
    if next_layer >= 40:
        return
    for expert_id in _hot_candidates(root, next_layer):
        _schedule(root, next_layer, expert_id, "lookahead")


def _async_warm(root: Path, layer_prefix: str, expert_ids: list[int]) -> None:
    """Stage current-route experts asynchronously; wait only on actual use."""
    layer = _layer_from_prefix(layer_prefix)
    if layer is None or not expert_ids:
        return _ORIGINAL_WARM(root, layer_prefix, expert_ids)
    if not ENABLED or not torch.cuda.is_available():
        return _ORIGINAL_WARM(root, layer_prefix, expert_ids)

    for expert_id in dict.fromkeys(int(v) for v in expert_ids):
        _schedule(root, layer, expert_id, "current")


def _wait(root: Path, layer: int, expert_id: int) -> None:
    state = _state(root)
    if state is None:
        return
    key = (int(layer), int(expert_id))
    pending = state.pending.get(key)
    if pending is None:
        return

    state.stats.waited += 1
    if not pending.future.done():
        state.stats.overlapped += 1
    try:
        pending.future.result()
        if pending.event is not None:
            torch.cuda.current_stream().wait_event(pending.event)
        state.pending.pop(key, None)
        state.stats.completed += 1
    except Exception:
        state.stats.errors += 1
        state.pending.pop(key, None)
        raise


def _async_moe(root, layer, residual, top_k, device):
    if device == "cuda":
        _lookahead(root, int(layer))
    return _ORIGINAL_MOE(root, layer, residual, top_k, device)


def _async_triplet(root, layer_prefix, expert_id, device):
    if device == "cuda":
        layer = _layer_from_prefix(layer_prefix)
        if layer is not None:
            _wait(root, layer, int(expert_id))
    return _ORIGINAL_TRIPLET(root, layer_prefix, expert_id, device)


def _cache_stats(root: Path) -> dict[str, int | float]:
    stats = dict(_ORIGINAL_CACHE_STATS(root))
    state = _state(root)
    if state is None:
        return stats
    _cleanup_done(state)
    stats.update(
        {
            "async_scheduled": state.stats.scheduled,
            "async_completed": state.stats.completed,
            "async_waited": state.stats.waited,
            "async_overlapped": state.stats.overlapped,
            "async_skipped_fp8": state.stats.skipped_fp8,
            "async_skipped_pending": state.stats.skipped_pending,
            "async_current_prefetch": state.stats.current_layer_prefetch,
            "async_lookahead_prefetch": state.stats.lookahead_prefetch,
            "async_errors": state.stats.errors,
            "async_pending": len(state.pending),
        }
    )
    return stats


def _print_cache(root: Path, label: str) -> None:
    _ORIGINAL_PRINT_CACHE(root, label)
    state = _state(root)
    if state is None:
        return
    _cleanup_done(state)
    by_layer = sorted(state.stats.by_layer.items())
    lookahead_text = ", ".join(f"L{layer:02d}:{count}" for layer, count in by_layer[-8:]) or "none"
    print(
        f"  async experts: enabled=1 | scheduled={state.stats.scheduled} | "
        f"completed={state.stats.completed} | pending={len(state.pending)} | "
        f"waited={state.stats.waited} | overlapped={state.stats.overlapped} | "
        f"current_prefetch={state.stats.current_layer_prefetch} | "
        f"lookahead_prefetch={state.stats.lookahead_prefetch} | "
        f"skipped_fp8={state.stats.skipped_fp8} | "
        f"skipped_pending={state.stats.skipped_pending} | errors={state.stats.errors}"
    )
    print(f"  async lookahead: experts={LOOKAHEAD_EXPERTS} | layers={lookahead_text}")


chat.batched_moe_step = _async_moe
chat._warm_expert_raw_cache = _async_warm
chat._expert_projection_triplet = _async_triplet
chat.cache_stats = _cache_stats
chat.print_cache = _print_cache

if ENABLED:
    print(
        f"async_expert_scheduler=enabled | lookahead={LOOKAHEAD_EXPERTS} | "
        f"workers={MAX_WORKERS} | cuda_stream=dedicated"
    )
