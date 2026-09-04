from __future__ import annotations

"""Keep startup/loading from materializing large model tensors unnecessarily.

This module is intentionally small and late-loaded by the canonical runner.
It fixes two legacy paths that conflict with the Q4 hierarchy:

1. Attention-type detection must inspect tensor names, not materialize the
   full tensors just to discover whether a layer is linear-attention or
   full-attention.
2. The old FP8 raw expert prefetch path is disabled while the Q4 hierarchy is
   active; otherwise the router keeps a second FP8 representation alive in
   CUDA in addition to Q4.
"""

import os
from pathlib import Path

from safetensors import safe_open

from . import qwen36_40layer_loop as base
from . import qwen36_cached_loop as cached
from . import qwen36_chat_batch as chat

_TRACE = os.getenv("QWEN36_MEMORY_LOAD_TRACE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

_ATTENTION_TYPES: dict[Path, tuple[str, ...]] = {}


def _has_tensor_name(root: Path, name: str) -> bool:
    store = cached._store(root.resolve())
    if store.weight_map:
        return name in store.weight_map

    # Fallback for checkpoints without an index: inspect safetensors metadata
    # only. Do not call get_tensor(), which would materialize the full tensor.
    for shard in sorted(root.glob("*.safetensors")):
        if not shard.is_file():
            continue
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            if name in handle.keys():
                return True
    return False


def _detect_attention_types(root: Path) -> tuple[str, ...]:
    root = root.resolve()
    result: list[str] = []
    for layer in range(base.DEFAULT_LAYERS):
        prefix = base.layer_prefix(layer)
        if _has_tensor_name(root, prefix + "linear_attn.in_proj_qkv.weight"):
            result.append("linear_attention")
        elif _has_tensor_name(root, prefix + "self_attn.q_proj.weight"):
            result.append("full_attention")
        else:
            raise KeyError(f"Could not identify attention type for layer {layer}")
    return tuple(result)


def _lazy_attention_type(root: Path, layer: int) -> str:
    key = root.resolve()
    detected = _ATTENTION_TYPES.get(key)
    if detected is None:
        if _TRACE:
            print("memory_load_trace=attention-layout|mode=metadata-only|begin")
        detected = _detect_attention_types(key)
        _ATTENTION_TYPES[key] = detected
        if _TRACE:
            linear = sum(value == "linear_attention" for value in detected)
            full = sum(value == "full_attention" for value in detected)
            print(
                "memory_load_trace=attention-layout|mode=metadata-only|done|"
                f"linear={linear}|full={full}"
            )
    return detected[int(layer)]


def _disabled_legacy_fp8_prefetch(root: Path, layer_prefix: str, expert_ids: list[int]) -> None:
    if _TRACE and expert_ids:
        print(
            "memory_load_trace=legacy-fp8-prefetch|disabled|"
            f"layer_prefix={layer_prefix}|experts={len(set(int(v) for v in expert_ids))}"
        )
    return None


base.attention_type = _lazy_attention_type
chat._warm_expert_raw_cache = _disabled_legacy_fp8_prefetch

if _TRACE:
    print("memory_loading_fix=enabled|attention=metadata-only|legacy_fp8_prefetch=disabled")
