from __future__ import annotations

"""Persistent attention state for Qwen3.6 autoregressive decoding."""

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from . import qwen36_40layer_loop as base
from . import qwen36_cached_loop as cached
from .qwen36_gated_norm_probe import gated_rmsnorm
from .qwen36_op_probe import rmsnorm

ROPE_THETA = 10_000_000.0
ROPE_DIM = int(base.FULL_HEAD_DIM * 0.25)
LINEAR_CONV_KERNEL = 4
LINEAR_CONV_STATE = LINEAR_CONV_KERNEL - 1
LINEAR_CONV_DIM = base.LINEAR_KEY_DIM * 2 + base.LINEAR_VALUE_DIM


@dataclass
class AttentionState:
    full_keys: dict[int, torch.Tensor] = field(default_factory=dict)
    full_values: dict[int, torch.Tensor] = field(default_factory=dict)
    linear_states: dict[int, torch.Tensor] = field(default_factory=dict)
    linear_conv_states: dict[int, torch.Tensor] = field(default_factory=dict)
    tokens_seen: int = 0
    device: str | None = None

    def reset(self) -> None:
        self.full_keys.clear()
        self.full_values.clear()
        self.linear_states.clear()
        self.linear_conv_states.clear()
        self.tokens_seen = 0

    def bind(self, device: str) -> None:
        if self.device != device:
            self.reset()
            self.device = device

    def snapshot(self) -> dict[str, int | float]:
        # KV cache layout is (batch, kv_heads, sequence, head_dim), so the
        # sequence length lives at dimension -2, not -1 (head_dim=256).
        full_tokens = sum(int(value.shape[-2]) for value in self.full_keys.values())
        linear_bytes = sum(int(value.numel() * value.element_size()) for value in self.linear_states.values())
        conv_bytes = sum(int(value.numel() * value.element_size()) for value in self.linear_conv_states.values())
        full_bytes = sum(
            int(tensor.numel() * tensor.element_size())
            for tensor in [*self.full_keys.values(), *self.full_values.values()]
        )
        return {
            "tokens_seen": int(self.tokens_seen),
            "full_layers_cached": len(self.full_keys),
            "full_tokens": full_tokens,
            "full_bytes": full_bytes,
            "linear_layers_cached": len(self.linear_states),
            "linear_bytes": linear_bytes,
            "linear_conv_layers_cached": len(self.linear_conv_states),
            "linear_conv_bytes": conv_bytes,
            "bytes": full_bytes + linear_bytes + conv_bytes,
        }


_STATES: dict[Path, AttentionState] = {}
_ACTIVE_STATES: dict[Path, AttentionState] = {}


def state_for(root: Path, device: str) -> AttentionState:
    key = root.resolve()
    state = _STATES.get(key)
    if state is None:
        state = AttentionState()
        _STATES[key] = state
    state.bind(device)
    return state


def reset(root: Path) -> AttentionState:
    key = root.resolve()
    state = _STATES.get(key)
    if state is None:
        state = AttentionState()
        _STATES[key] = state
    state.reset()
    return state


def activate(root: Path, state: AttentionState) -> None:
    _ACTIVE_STATES[root.resolve()] = state
