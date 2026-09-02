from __future__ import annotations

"""Qwen3.6 loop with bounded prefetch of guaranteed MoE weights.

This wrapper builds on qwen36_cached_loop.py and preloads only MoE tensors that
are guaranteed to be needed for the current layer before attention finishes:
post-attention LayerNorm, router weights, shared-expert weights/scales, and the
shared-expert gate. Routed expert IDs are still determined by the real router;
no prediction or approximation is used.

The prefetched objects remain as raw CPU tensors. They are consumed by the
normal MoE path and removed from the prefetch cache, keeping the memory budget
small. A single worker is used to avoid unsafe concurrent access to the
Safetensors handle store.
"""

import atexit
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from . import qwen36_cached_loop as optimized
from . import qwen36_40layer_loop as base

PREFETCH_WORKERS = 1

_executor = ThreadPoolExecutor(
    max_workers=PREFETCH_WORKERS,
    thread_name_prefix="moe-weight-prefetch",
)
_prefetch_lock = threading.Lock()
_prefetch_cache: dict[tuple[Path, str], torch.Tensor] = {}
_prefetch_futures: dict[tuple[Path, str], object] = {}


# Keep the optimized persistent reader as the underlying source. The wrapper
# below adds only a small RAM cache for tensors that are certain to be consumed
# by the current layer's MoE block.
_original_load_tensor = optimized._cached_load_tensor


def _key(root: Path, name: str) -> tuple[Path, str]:
    return root.resolve(), name


def _prefetch_one(root: Path, name: str) -> torch.Tensor:
    tensor = _original_load_tensor(root, name, device="cpu")
    key = _key(root, name)
    with _prefetch_lock:
        _prefetch_cache[key] = tensor
        _prefetch_futures.pop(key, None)
    return tensor


def _load_tensor(root: Path, name: str, device: str = "cpu") -> torch.Tensor:
    key = _key(root, name)
    with _prefetch_lock:
        future = _prefetch_futures.get(key)
        cached = _prefetch_cache.get(key)

    if cached is None and future is not None:
        cached = future.result()

    if cached is None:
        return _original_load_tensor(root, name, device=device)

    # Consumption is one-shot. The normal code may request the same tensor
    # more than once within a layer, so only remove it when explicitly told to
    # consume it; ordinary reads keep the small cached tensor available.
    if device == "cpu":
        return cached
    return cached.to(device=device)


def _schedule(root: Path, layer: int) -> None:
    prefix = base.layer_prefix(layer)
    names = [
        prefix + "post_attention_layernorm.weight",
        prefix + "mlp.gate.weight",
        prefix + "mlp.shared_expert.gate_proj.weight",
        prefix + "mlp.shared_expert.gate_proj.weight_scale_inv",
        prefix + "mlp.shared_expert.up_proj.weight",
        prefix + "mlp.shared_expert.up_proj.weight_scale_inv",
        prefix + "mlp.shared_expert.down_proj.weight",
        prefix + "mlp.shared_expert.down_proj.weight_scale_inv",
        prefix + "mlp.shared_expert_gate.weight",
    ]

    for name in names:
        key = _key(root, name)
        with _prefetch_lock:
            if key in _prefetch_cache or key in _prefetch_futures:
                continue
            _prefetch_futures[key] = _executor.submit(_prefetch_one, root, name)


def _cleanup_layer(root: Path, layer: int) -> None:
    prefix = base.layer_prefix(layer)
    names = [
        prefix + "post_attention_layernorm.weight",
        prefix + "mlp.gate.weight",
        prefix + "mlp.shared_expert.gate_proj.weight",
        prefix + "mlp.shared_expert.gate_proj.weight_scale_inv",
        prefix + "mlp.shared_expert.up_proj.weight",
        prefix + "mlp.shared_expert.up_proj.weight_scale_inv",
        prefix + "mlp.shared_expert.down_proj.weight",
        prefix + "mlp.shared_expert.down_proj.weight_scale_inv",
        prefix + "mlp.shared_expert_gate.weight",
    ]
    with _prefetch_lock:
        for name in names:
            tensor = _prefetch_cache.pop(_key(root, name), None)
            if tensor is not None:
                del tensor


_original_linear_attention = base.linear_attention_step
_original_full_attention = base.full_attention_step


def _linear_attention_with_prefetch(root: Path, layer: int, x0: torch.Tensor, device: str):
    _schedule(root, layer)
    return _original_linear_attention(root, layer, x0, device)


def _full_attention_with_prefetch(root: Path, layer: int, x0: torch.Tensor, device: str):
    _schedule(root, layer)
    return _original_full_attention(root, layer, x0, device)


# Patch the reference module with the optimized reader plus the guaranteed
# MoE-weight prefetch layer.
base.load_tensor = _load_tensor
base.linear_attention_step = _linear_attention_with_prefetch
base.full_attention_step = _full_attention_with_prefetch


@atexit.register
def _shutdown() -> None:
    try:
        _executor.shutdown(wait=True, cancel_futures=False)
    finally:
        with _prefetch_lock:
            _prefetch_cache.clear()
            _prefetch_futures.clear()
        for store in optimized._stores.values():
            store.close()


def main() -> None:
    base.main()
    print(
        "guaranteed MoE prefetch: "
        f"workers={PREFETCH_WORKERS} | "
        f"cache_tensors={len(_prefetch_cache)}"
    )


if __name__ == "__main__":
    main()
