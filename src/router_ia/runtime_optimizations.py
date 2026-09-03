from __future__ import annotations

"""Low-risk runtime optimizations for the Qwen3.6 token loop.

These patches are intentionally conservative: they do not alter model math or
routing decisions. They remove allocator/GC churn from the hot path and avoid
re-detecting the hybrid attention layout for every generated token.
"""

import gc
from pathlib import Path
from typing import Callable, TypeVar

import torch

from . import qwen36_40layer_loop as base
from . import qwen36_chat_batch as chat

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


base.attention_type = _cached_attention_type
base.linear_attention_step = _without_allocator_flush(base.linear_attention_step)
base.full_attention_step = _without_allocator_flush(base.full_attention_step)


# qwen36_chat_batch.run_generated_token performs a gc.collect() after every
# generated token. Keep that cleanup out of the latency-critical path too.
_ORIGINAL_CHAT_RUN_GENERATED_TOKEN = chat.run_generated_token
chat.run_generated_token = _without_allocator_flush(_ORIGINAL_CHAT_RUN_GENERATED_TOKEN)
