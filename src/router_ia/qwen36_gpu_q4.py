from __future__ import annotations

"""GPU-only Q4 decode path for routed experts.

Q4 experts live in host RAM as packed uint8 nibbles plus FP16 scales. The host
side is only responsible for owning those raw buffers. For CUDA inference,
this patch avoids building a large temporary Q4 batch on CPU: every packed
buffer is copied to CUDA first, then stacking, nibble unpacking, scaling and
FP16 materialization all happen on the GPU.
"""

from typing import Any

import torch

from . import qwen36_expert_cache as expert_cache


def _cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU Q4 path requires CUDA")
    return torch.device("cuda")


def _to_cuda_raw(tensor: torch.Tensor) -> torch.Tensor:
    """Transfer only the raw packed/scalar buffer; no arithmetic is done on CPU."""
    if tensor.device.type == "cuda":
        return tensor
    return tensor.to(device=_cuda_device(), non_blocking=True)


def _gpu_q4_dequantize_matrix(matrix: Any, device: str = "cuda") -> torch.Tensor:
    if str(device) != "cuda":
        # Preserve the reference CPU mode for non-CUDA diagnostics.
        return expert_cache._q4_dequantize_matrix(matrix, device=device)

    packed, scale, shape = matrix
    packed_gpu = _to_cuda_raw(packed)
    scale_gpu = _to_cuda_raw(scale)

    low = (packed_gpu & 0x0F).to(torch.int16) - 8
    high = ((packed_gpu >> 4) & 0x0F).to(torch.int16) - 8
    q = torch.stack((low, high), dim=1).reshape(-1)[: shape[0] * shape[1]]
    return (q.float() * scale_gpu.float()).reshape(shape).to(torch.float16)


def _gpu_q4_dequantize_entry_batch(
    entries: list[Any], projection: int
) -> list[torch.Tensor]:
    if not entries:
        return []

    shapes = [entry[projection][2] for entry in entries]
    rows, cols = shapes[0]
    if any(shape != (rows, cols) for shape in shapes):
        return [
            _gpu_q4_dequantize_matrix(entry[projection], device="cuda")
            for entry in entries
        ]

    # Important: stack happens AFTER the raw RAM -> VRAM copies.
    # No Q4 arithmetic is performed on the CPU.
    packed = torch.stack(
        [_to_cuda_raw(entry[projection][0]) for entry in entries], dim=0
    )
    scales = torch.stack(
        [_to_cuda_raw(entry[projection][1]) for entry in entries], dim=0
    )

    low = (packed & 0x0F).to(torch.int16) - 8
    high = ((packed >> 4) & 0x0F).to(torch.int16) - 8
    q = torch.stack((low, high), dim=2).reshape(len(entries), -1)
    q = q[:, : rows * cols]
    return (q.float() * scales.float().reshape(len(entries), 1)).reshape(
        len(entries), rows, cols
    ).to(torch.float16)


# Patch the active cache implementation. qwen36_chat_batch imports and uses
# this module through the canonical runner, so one patch covers all Q4 hits.
expert_cache._q4_dequantize_matrix = _gpu_q4_dequantize_matrix
expert_cache._q4_dequantize_entry_batch = _gpu_q4_dequantize_entry_batch

print("q4_decode=gpu-only|host=raw-buffer-transfer|dequant=CUDA")
