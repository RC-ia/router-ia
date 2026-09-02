from __future__ import annotations

"""Memory-conscious FP8 blockwise dequantization for Qwen3.6.

The previous implementation expanded the block scale matrix to the full
weight shape with repeat_interleave(). That creates an additional large
intermediate tensor. This implementation broadcasts the scales over 128x128
blocks instead, avoiding the expanded scale tensor.
"""

import torch

BLOCK = 128


def dequantize_fp8_blockwise(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
) -> torch.Tensor:
    if weight.ndim != 2 or scale_inv.ndim != 2:
        raise ValueError(
            "Expected 2-D weight/scale tensors, "
            f"got {tuple(weight.shape)} and {tuple(scale_inv.shape)}"
        )

    out_features, in_features = map(int, weight.shape)
    expected = (
        (out_features + BLOCK - 1) // BLOCK,
        (in_features + BLOCK - 1) // BLOCK,
    )
    if tuple(scale_inv.shape) != expected:
        raise ValueError(
            f"Scale shape {tuple(scale_inv.shape)} does not match "
            f"weight {tuple(weight.shape)}; expected {expected}"
        )

    # Fast path for the Qwen3.6 expert/projection matrices used here, whose
    # dimensions are aligned to the 128x128 quantization blocks.
    if out_features % BLOCK == 0 and in_features % BLOCK == 0:
        w = weight.float().reshape(
            out_features // BLOCK,
            BLOCK,
            in_features // BLOCK,
            BLOCK,
        )
        s = scale_inv.float().reshape(
            out_features // BLOCK,
            1,
            in_features // BLOCK,
            1,
        )
        return (w * s).reshape(out_features, in_features)

    # General fallback for partially filled edge blocks. Padding is avoided so
    # this remains safe for arbitrary 2-D FP8 matrices.
    expanded = scale_inv.float().repeat_interleave(BLOCK, dim=0)
    expanded = expanded.repeat_interleave(BLOCK, dim=1)
    return weight.float() * expanded[:out_features, :in_features]
