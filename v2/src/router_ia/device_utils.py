"""Device parsing and capability helpers for runtime entry points."""

from __future__ import annotations

import argparse

import torch

DeviceLike = str | torch.device


def parse_device(value: str) -> torch.device:
    """Parse a PyTorch device, preserving an optional CUDA device index."""
    try:
        return torch.device(value)
    except (RuntimeError, TypeError) as exc:
        raise argparse.ArgumentTypeError(f"invalid PyTorch device: {value!r}") from exc


def is_cuda(device: DeviceLike) -> bool:
    """Return whether *device* is any CUDA device, including ``cuda:N``."""
    return torch.device(device).type == "cuda"


def is_cpu(device: DeviceLike) -> bool:
    """Return whether *device* is a CPU device."""
    return torch.device(device).type == "cpu"
