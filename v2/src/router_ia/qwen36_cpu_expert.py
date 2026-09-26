from __future__ import annotations

"""CPU execution for Q4 experts that are resident in host RAM.

The Q4 representation is intentionally kept packed in RAM. A RAM hit is
therefore executed locally: packed Q4 -> FP32 weights -> expert MLP -> small
output transfer back to the GPU. Full expert matrices never cross PCIe.
"""

from dataclasses import dataclass
from time import perf_counter

import torch
import torch.nn.functional as F

from .qwen36_expert_cache import ColdEntry

CPU_DTYPE = torch.float32

@dataclass(frozen=True)
class CPUExpertStats:
    experts: int
    seconds: float
    dequant_seconds: float
    matmul_seconds: float

def _dequantize_q4_matrix_cpu(matrix) -> torch.Tensor:
    packed, scale, shape = matrix
    packed_cpu = packed.detach().to(device="cpu", dtype=torch.uint8)
    scale_cpu = scale.detach().to(device="cpu", dtype=CPU_DTYPE)
    low = packed_cpu.bitwise_and(0x0F).to(torch.int16) - 8
    high = torch.bitwise_right_shift(packed_cpu, 4).bitwise_and(0x0F).to(torch.int16) - 8
    q = torch.stack((low, high), dim=1).reshape(-1)[: shape[0] * shape[1]]
    return (q.to(CPU_DTYPE) * scale_cpu).reshape(shape)

def run_q4_expert_cpu(entry: ColdEntry, x_cpu: torch.Tensor) -> tuple[torch.Tensor, float, float, float]:
    """Execute one Q4 routed expert entirely on CPU."""
    if x_cpu.device.type != "cpu":
        raise ValueError(f"x_cpu must be on CPU, got {x_cpu.device}")
    if x_cpu.ndim != 2 or x_cpu.shape[0] != 1:
        raise ValueError(f"Expected x_cpu=[1,hidden], got {tuple(x_cpu.shape)}")
    started = perf_counter()
    dequant_started = started
    gate_w = _dequantize_q4_matrix_cpu(entry[0])
    up_w = _dequantize_q4_matrix_cpu(entry[1])
    down_w = _dequantize_q4_matrix_cpu(entry[2])
    dequant_seconds = perf_counter() - dequant_started
    matmul_started = perf_counter()
    gate = F.linear(x_cpu, gate_w)
    up = F.linear(x_cpu, up_w)
    hidden = F.silu(gate) * up
    out = F.linear(hidden, down_w).contiguous()
    matmul_seconds = perf_counter() - matmul_started
    return out, perf_counter() - started, dequant_seconds, matmul_seconds

def run_q4_expert_batch_cpu(entries: list[ColdEntry], x: torch.Tensor, *, workers: int = 1):
    """Run Q4 experts from RAM without moving their weights to the GPU."""
    if not entries:
        return [], CPUExpertStats(0, 0.0, 0.0, 0.0)
    x_cpu = x.detach().to(device="cpu", dtype=CPU_DTYPE)
    started = perf_counter()
    if workers <= 1 or len(entries) == 1:
        results = [run_q4_expert_cpu(entry, x_cpu) for entry in entries]
    else:
        from concurrent.futures import ThreadPoolExecutor
        worker_count = min(max(int(workers), 1), len(entries))
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="q4-cpu") as pool:
            futures = [pool.submit(run_q4_expert_cpu, entry, x_cpu) for entry in entries]
            results = [future.result() for future in futures]
    outputs = [item[0] for item in results]
    total = perf_counter() - started
    dequant_seconds = sum(item[2] for item in results)
    matmul_seconds = sum(item[3] for item in results)
    return outputs, CPUExpertStats(len(entries), total, dequant_seconds, matmul_seconds)

__all__ = ["CPUExpertStats", "run_q4_expert_cpu", "run_q4_expert_batch_cpu"]
