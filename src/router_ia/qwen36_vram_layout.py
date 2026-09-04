from __future__ import annotations

"""Explicit VRAM layout for the Qwen3.6 runtime.

The layout keeps compute headroom separate from persistent cache capacity.
Persistent cache is split into resident tensors, routed experts and a rotating
transfer window. The transfer window is always the first tier sacrificed under
compute pressure; allocator-wide cache flushing is intentionally avoided.
"""

import os
from dataclasses import dataclass

import torch


def _positive_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


@dataclass(frozen=True)
class VRAMLayout:
    """Independent VRAM regions, all expressed in bytes."""

    compute_reserve_bytes: int
    resident_bytes: int
    expert_bytes: int
    transfer_bytes: int

    @property
    def persistent_cache_bytes(self) -> int:
        return self.resident_bytes + self.expert_bytes + self.transfer_bytes

    @property
    def total_managed_bytes(self) -> int:
        return self.compute_reserve_bytes + self.persistent_cache_bytes

    def as_dict(self) -> dict[str, int | float]:
        return {
            "compute_reserve_bytes": self.compute_reserve_bytes,
            "resident_bytes": self.resident_bytes,
            "expert_bytes": self.expert_bytes,
            "transfer_bytes": self.transfer_bytes,
            "persistent_cache_bytes": self.persistent_cache_bytes,
            "total_managed_bytes": self.total_managed_bytes,
        }


def build_layout(persistent_cache_bytes: int) -> VRAMLayout:
    persistent = max(int(persistent_cache_bytes), 0)
    resident_ratio = min(_positive_float("QWEN36_RESIDENT_VRAM_RATIO", 0.60), 0.95)
    resident = min(int(persistent * resident_ratio), persistent)
    remaining = max(persistent - resident, 0)

    requested_expert = int(_positive_float("QWEN36_EXPERT_VRAM_GB", 1.0) * 1024**3)
    expert = min(requested_expert, remaining)
    remaining -= expert

    stream_raw = os.getenv("QWEN36_VRAM_STREAM_GB")
    if stream_raw is None:
        transfer = remaining
    else:
        try:
            transfer = min(max(int(float(stream_raw) * 1024**3), 0), remaining)
        except ValueError:
            transfer = remaining

    compute = int(_positive_float("QWEN36_VRAM_COMPUTE_GB", 1.0) * 1024**3)
    return VRAMLayout(compute, resident, expert, transfer)


def configure_allocator(layout: VRAMLayout, explicit_gb: float | None = None) -> None:
    """Cap the process allocator to compute reserve + persistent cache when safe."""
    if not torch.cuda.is_available():
        return
    props = torch.cuda.get_device_properties(0)
    total_gib = props.total_memory / 1024**3
    target_gib = explicit_gb if explicit_gb is not None else layout.total_managed_bytes / 1024**3
    if target_gib <= 0 or target_gib >= total_gib:
        return
    torch.cuda.set_per_process_memory_fraction(target_gib / total_gib, 0)
