from __future__ import annotations

"""Bound CPU temporary memory during FP8 -> Q4 expert conversion.

The legacy hierarchy materialized each expert as a full FP32 dequantized
matrix, then a full FP16 matrix, then Q4. Repeating that hundreds of times can
inflate Windows process commit even though the tensors are later released.

This replacement keeps the exact scalar-per-matrix Q4 format, but computes and
packs it in small row chunks directly from the FP8 source + block scales.
"""

import os
from typing import Any

import torch

from . import qwen36_expert_q4_hierarchy_fixed as hierarchy
from . import qwen36_expert_cache as expert_cache

Q4_BLOCK = 128
Q4_CHUNK_ROWS = max(int(os.getenv("QWEN36_Q4_CHUNK_ROWS", "2048")), Q4_BLOCK)
_ORIGINAL_SOURCE_TO_Q4 = hierarchy._source_to_q4


def _scale_rows(scale: torch.Tensor, row_start: int, row_end: int, cols: int) -> torch.Tensor:
    block_start = row_start // Q4_BLOCK
    block_end = (row_end + Q4_BLOCK - 1) // Q4_BLOCK
    rows = scale[block_start:block_end].float().repeat_interleave(Q4_BLOCK, dim=0)
    rows = rows[: row_end - row_start, :]
    return rows.repeat_interleave(Q4_BLOCK, dim=1)[:, :cols]


def _q4_from_fp8(weight: torch.Tensor, scale: torch.Tensor) -> expert_cache.Q4Matrix:
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError(f"Expected 2-D FP8 weight/scale, got {tuple(weight.shape)} / {tuple(scale.shape)}")

    rows, cols = map(int, weight.shape)

    # First pass: recover only the scalar max(abs(x)) needed by the existing
    # per-matrix Q4 format. No full dequantized matrix is allocated.
    max_abs = torch.tensor(0.0, dtype=torch.float32)
    for row_start in range(0, rows, Q4_CHUNK_ROWS):
        row_end = min(row_start + Q4_CHUNK_ROWS, rows)
        block_weight = weight[row_start:row_end].float()
        row_scale = _scale_rows(scale, row_start, row_end, cols)
        local = (block_weight * row_scale).abs().amax()
        max_abs = torch.maximum(max_abs, local.cpu())
        del block_weight, row_scale, local

    q4_scale = torch.clamp(max_abs / 7.0, min=torch.finfo(torch.float32).tiny).to(torch.float16)
    total = rows * cols
    packed = torch.empty((total + 1) // 2, dtype=torch.uint8)

    # Second pass: pack directly into the final Q4 buffer using bounded chunks.
    flat_offset = 0
    for row_start in range(0, rows, Q4_CHUNK_ROWS):
        row_end = min(row_start + Q4_CHUNK_ROWS, rows)
        block_weight = weight[row_start:row_end].float()
        row_scale = _scale_rows(scale, row_start, row_end, cols)
        values = torch.round((block_weight * row_scale) / q4_scale.float()).clamp(-7, 7).to(torch.int16) + 8
        flat = values.reshape(-1)
        if flat.numel() & 1:
            flat_for_pack = torch.cat((flat, torch.full((1,), 8, dtype=torch.int16)))
        else:
            flat_for_pack = flat
        packed[flat_offset : flat_offset + flat_for_pack.numel() // 2] = (
            flat_for_pack[0::2].to(torch.uint8)
            | (flat_for_pack[1::2].to(torch.uint8) << 4)
        )
        flat_offset += flat.numel() // 2
        del block_weight, row_scale, values, flat, flat_for_pack

    return packed, q4_scale, (rows, cols)


def _compact_source_to_q4(store: Any, root, layer: int, expert_id: int, layer_prefix: str):
    prefix = f"{layer_prefix}mlp.experts.{int(expert_id)}"
    matrices = []
    for name in ("gate_proj", "up_proj", "down_proj"):
        weight = store._load_ssd(prefix + "." + name + ".weight")
        scale = store._load_ssd(prefix + "." + name + ".weight_scale_inv")
        if weight.dtype == torch.float8_e4m3fn:
            matrix = _q4_from_fp8(weight, scale)
        else:
            # Non-FP8 checkpoints are not the current target; preserve the
            # existing path for compatibility.
            matrix = expert_cache._q4_quantize_matrix(weight.float())
        matrices.append(matrix)
        del weight, scale
    return tuple(matrices)


hierarchy._source_to_q4 = _compact_source_to_q4

# Give Q4 RAM a useful per-layer residency window. The total Q4 RAM budget is
# still the hard ceiling, but 24 slots caused every VRAM eviction to spill
# straight to SSD before other layers could reuse the host copy.
hierarchy.Q4_RAM_SLOTS_PER_LAYER = max(int(os.getenv("QWEN36_Q4_RAM_SLOTS_PER_LAYER", "128")), 1)

print(
    f"q4_source_compact=enabled|chunk_rows={Q4_CHUNK_ROWS}|"
    f"ram_slots_per_layer={hierarchy.Q4_RAM_SLOTS_PER_LAYER}|"
    "temp_full_matrix=disabled"
)
