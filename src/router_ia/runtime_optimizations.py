from __future__ import annotations

"""Low-risk runtime optimizations for the Qwen3.6 token loop.

These patches are intentionally conservative: they do not alter model math or
routing decisions. They remove allocator/GC churn from the hot path, avoid
re-detecting the hybrid attention layout for every generated token, and keep
the fused runner's VRAM cache budget bounded.
"""

import gc
import sys
from pathlib import Path
from typing import Callable, TypeVar

import torch

from . import qwen36_40layer_loop as base
from . import qwen36_chat_batch as chat
from . import qwen36_cached_loop as cached
from . import qwen36_expert_cache as expert_cache

_T = TypeVar("_T")


_ORIGINAL_ATTENTION_TYPE = base.attention_type
_ATTENTION_TYPES: dict[Path, tuple[str, ...]] = {}


def _cached_attention_type(root: Path, layer: int) -> str:
    key = root.resolve()
    cached_types = _ATTENTION_TYPES.get(key)
    if cached_types is None:
        detected = tuple(_ORIGINAL_ATTENTION_TYPE(key, index) for index in range(base.DEFAULT_LAYERS))
        _ATTENTION_TYPES[key] = detected
        cached_types = detected
    return cached_types[int(layer)]


def _without_allocator_flush(fn: Callable[..., _T]) -> Callable[..., _T]:
    """Run a hot-path function without forced Python/CUDA cache flushing."""
    def wrapped(*args, **kwargs):
        old_collect = gc.collect
        old_empty_cache = torch.cuda.empty_cache
        gc.collect = lambda: 0
        torch.cuda.empty_cache = lambda: None
        try:
            return fn(*args, **kwargs)
        finally:
            gc.collect = old_collect
            torch.cuda.empty_cache = old_empty_cache

    return wrapped


def _configure_fused_cache_budget() -> None:
    """Keep fused-runner caches within roughly 3 GiB of configured cache VRAM.

    The fused expert cache and the generic streaming cache are separate pools,
    so the old configuration could reserve the same ~1.2 GiB twice. Reserve
    about 1.0 GiB for the FP8 expert cache and give the generic stream only the
    remaining cache budget; Q4 backing stays exclusively in system RAM.
    """
    if "qwen36_chat_batch_fused" not in sys.argv[0]:
        return

    expert_budget = 1 * 1024**3
    stream_budget = max(
        cached.VRAM_CACHE_BUDGET_BYTES
        - cached.RESIDENT_VRAM_BUDGET_BYTES
        - expert_budget,
        0,
    )
    cached.STREAM_BUDGET_BYTES = stream_budget
    cached.STREAM_GB = stream_budget / 1024**3

    if getattr(expert_cache.RoutedExpertCache, "_router_ia_budget_patch", False):
        return

    original_init = expert_cache.RoutedExpertCache.__init__

    def bounded_init(self, budget_bytes: int, layers: int = expert_cache.MODEL_LAYERS) -> None:
        requested = max(int(budget_bytes), 0)
        effective = max(requested, expert_budget) if requested else 0
        original_init(self, effective, layers)
        if effective:
            # Q4 is host-RAM backing and therefore must not consume VRAM slots.
            self.q4_slots = expert_cache.Q4_SLOTS_PER_LAYER
            self.slots_per_layer = self.fp8_slots + self.q4_slots
            self.total_slots = self.slots_per_layer * self.layers

    expert_cache.RoutedExpertCache.__init__ = bounded_init
    expert_cache.RoutedExpertCache._router_ia_budget_patch = True


_configure_fused_cache_budget()

base.attention_type = _cached_attention_type
base.linear_attention_step = _without_allocator_flush(base.linear_attention_step)
base.full_attention_step = _without_allocator_flush(base.full_attention_step)


# qwen36_chat_batch.run_generated_token performs a gc.collect() after every
# generated token. Keep that cleanup out of the latency-critical path too.
_ORIGINAL_CHAT_RUN_GENERATED_TOKEN = chat.run_generated_token
chat.run_generated_token = _without_allocator_flush(_ORIGINAL_CHAT_RUN_GENERATED_TOKEN)
