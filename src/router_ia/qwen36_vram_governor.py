from __future__ import annotations

"""VRAM governor for the optimized Qwen3.6 runner.

The governor treats compute headroom as protected memory. When unmanaged CUDA
allocations consume that headroom, only the rotating transfer window is
released. Persistent resident tensors and the dedicated routed-expert cache
are not flushed, and torch.cuda.empty_cache() is never used here.
"""

import os
from pathlib import Path

import torch

from . import qwen36_cached_loop as cached
from . import qwen36_chat_batch as chat


def _positive_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


COMPUTE_VRAM_GB = _positive_float("QWEN36_VRAM_COMPUTE_GB", 1.0)
COMPUTE_VRAM_RESERVE_BYTES = int(COMPUTE_VRAM_GB * 1024**3)

_ORIGINAL_WARM = chat._warm_expert_raw_cache
_ORIGINAL_TRIPLET = chat._expert_projection_triplet


def protect(root: Path) -> None:
    """Return transfer-window memory to the allocator without a global flush."""
    if not torch.cuda.is_available():
        return

    store = cached._store(root)
    generic_vram = int(store.vram_cache.snapshot()["bytes"])
    dedicated_expert = 0
    expert_caches = getattr(chat, "_EXPERT_CACHES", {})
    expert = expert_caches.get(root.resolve())
    if expert is not None:
        dedicated_expert = int(expert.snapshot()["bytes"])

    allocated = int(torch.cuda.memory_allocated())
    unmanaged = max(allocated - generic_vram - dedicated_expert, 0)
    if unmanaged <= COMPUTE_VRAM_RESERVE_BYTES:
        return

    # Transfer/stream is the sacrificial tier. Clearing it releases the tensor
    # references immediately; PyTorch can reuse the allocator blocks without an
    # allocator-wide empty_cache synchronization.
    store.clear_stream()


def _warm(root: Path, layer_prefix: str, expert_ids: list[int]) -> None:
    protect(root)
    _ORIGINAL_WARM(root, layer_prefix, expert_ids)


def _triplet(root: Path, layer_prefix: str, expert_id: int, device: str):
    if device == "cuda":
        protect(root)
    return _ORIGINAL_TRIPLET(root, layer_prefix, expert_id, device)


chat._warm_expert_raw_cache = _warm
chat._expert_projection_triplet = _triplet
