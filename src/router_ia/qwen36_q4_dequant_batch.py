from __future__ import annotations

"""Batch Q4 dequantization for the Qwen3.6 routed-expert hot path."""

import torch

from . import qwen36_expert_batch_plan_v2 as planner_v2

_ORIGINAL = planner_v2._decode_q4


def _decode_q4_batched(entries):
    if not entries:
        return {}

    cold_entries = [entry[2] for entry in entries]
    projection_count = 3
    shapes = [cold_entries[0][p][2] for p in range(projection_count)]

    # Preserve correctness for unusual checkpoints with mixed projection shapes.
    if any(
        cold_entries[i][p][2] != shapes[p]
        for i in range(len(cold_entries))
        for p in range(projection_count)
    ):
        return _ORIGINAL(entries)
    if any(
        cold_entries[i][p][0].numel() != cold_entries[0][p][0].numel()
        for i in range(len(cold_entries))
        for p in range(projection_count)
    ):
        return _ORIGINAL(entries)

    n = len(cold_entries)
    packed = torch.stack(
        [cold_entries[i][p][0] for i in range(n) for p in range(projection_count)],
        dim=0,
    ).to(device="cuda", non_blocking=True)
    scales = torch.stack(
        [cold_entries[i][p][1] for i in range(n) for p in range(projection_count)],
        dim=0,
    ).to(device="cuda", non_blocking=True)

    low = (packed & 0x0F).to(torch.int16) - 8
    high = ((packed >> 4) & 0x0F).to(torch.int16) - 8
    q = torch.stack((low, high), dim=2).reshape(n * projection_count, -1)
    decoded = (
        q.float() * scales.float().reshape(n * projection_count, 1)
    ).reshape(
        n * projection_count,
        shapes[0][0],
        shapes[0][1],
    ).to(torch.float16)

    output = {}
    for local, (expert_id, _tier, _entry) in enumerate(entries):
        base = local * projection_count
        output[int(expert_id)] = (
            decoded[base], decoded[base + 1], decoded[base + 2]
        )
    return output


planner_v2._decode_q4 = _decode_q4_batched

print(
    "q4_dequant_batch=enabled|projections-per-call=3|"
    "cuda-launch-chain=batched|math=unchanged|fallback=mixed-shapes"
)
