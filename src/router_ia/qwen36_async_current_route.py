from __future__ import annotations

"""Higher-concurrency pool for the current routed experts.

The async scheduler's speculative lookahead intentionally stays small, but the
actual current route contains up to top-k experts and cannot afford to stage
those experts only two at a time. This patch gives current-route prefetches a
separate host worker pool while preserving the original lookahead pool.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import torch

from . import qwen36_async_scheduler as scheduler
from . import qwen36_chat_batch as chat
from . import qwen36_official_optimizations as official

CURRENT_WORKERS = max(int(os.getenv("QWEN36_ASYNC_CURRENT_WORKERS", "8")), 1)

_CURRENT_POOLS: dict[Path, ThreadPoolExecutor] = {}
_CURRENT_POOLS_LOCK = Lock()
_ORIGINAL_SCHEDULE = scheduler._schedule


def _current_pool(root: Path) -> ThreadPoolExecutor:
    key = root.resolve()
    with _CURRENT_POOLS_LOCK:
        pool = _CURRENT_POOLS.get(key)
        if pool is None:
            pool = ThreadPoolExecutor(
                max_workers=CURRENT_WORKERS,
                thread_name_prefix="router-ia-current",
            )
            _CURRENT_POOLS[key] = pool
        return pool


def _schedule(root: Path, layer: int, expert_id: int, source: str) -> None:
    if source != "current":
        return _ORIGINAL_SCHEDULE(root, layer, expert_id, source)

    state = scheduler._state(root)
    if state is None or layer >= 40:
        return
    key = (int(layer), int(expert_id))
    scheduler._cleanup_done(state)
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
            scheduler._do_prefetch(root, int(layer), prefix, int(expert_id), state)
            event.record(state.stream)
        except Exception:
            state.stats.errors += 1
            raise

    future = _current_pool(root).submit(job)
    state.pending[key] = scheduler._Pending(future, event, int(layer), int(expert_id))
    state.stats.scheduled += 1
    state.stats.by_layer[int(layer)] = state.stats.by_layer.get(int(layer), 0) + 1
    state.stats.current_layer_prefetch += 1


scheduler._schedule = _schedule

print(
    f"async_current_route=separate-pool|workers={CURRENT_WORKERS}|"
    f"lookahead_workers={scheduler.MAX_WORKERS}"
)
