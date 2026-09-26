from __future__ import annotations

"""Memory-conscious FP8 blockwise dequantization for Qwen3.6.

The implementation broadcasts scales over 128x128 blocks and also exposes a
batched path used by the routed-expert streamer. The batched path keeps the
expert matrices grouped on the GPU, reducing CUDA launch and transfer
overhead from one call per projection to one call per expert triplet.
"""

import torch

BLOCK = 128


def _validate_2d(weight: torch.Tensor, scale_inv: torch.Tensor) -> tuple[int, int]:
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
    return out_features, in_features


def dequantize_fp8_blockwise(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
) -> torch.Tensor:
    out_features, in_features = _validate_2d(weight, scale_inv)

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

    expanded = scale_inv.float().repeat_interleave(BLOCK, dim=0)
    expanded = expanded.repeat_interleave(BLOCK, dim=1)
    return weight.float() * expanded[:out_features, :in_features]


def dequantize_fp8_blockwise_batch(
    weights: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Dequantize a stack of equal-shaped FP8 matrices in one vectorized pass.

    ``weights`` has shape ``[N, out_features, in_features]`` and ``scales``
    has shape ``[N, ceil(out/128), ceil(in/128)]``. The leading batch dimension
    is preserved, making this suitable for grouped gate/up/down expert loads.
    """
    if weights.ndim != 3 or scales.ndim != 3:
        raise ValueError(
            "Expected 3-D batched weight/scale tensors, "
            f"got {tuple(weights.shape)} and {tuple(scales.shape)}"
        )

    n, out_features, in_features = map(int, weights.shape)
    expected = (
        n,
        (out_features + BLOCK - 1) // BLOCK,
        (in_features + BLOCK - 1) // BLOCK,
    )
    if tuple(scales.shape) != expected:
        raise ValueError(
            f"Scale shape {tuple(scales.shape)} does not match weights "
            f"{tuple(weights.shape)}; expected {expected}"
        )

    if out_features % BLOCK == 0 and in_features % BLOCK == 0:
        w = weights.float().reshape(
            n,
            out_features // BLOCK,
            BLOCK,
            in_features // BLOCK,
            BLOCK,
        )
        s = scales.float().reshape(
            n,
            out_features // BLOCK,
            1,
            in_features // BLOCK,
            1,
        )
        return (w * s).reshape(n, out_features, in_features)

    expanded = scales.float().repeat_interleave(BLOCK, dim=1)
    expanded = expanded.repeat_interleave(BLOCK, dim=2)
    return weights.float() * expanded[:, :out_features, :in_features]
