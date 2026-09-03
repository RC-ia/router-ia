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


@dataclass
class AttentionState:
    full_keys: dict[int, torch.Tensor] = field(default_factory=dict)
    full_values: dict[int, torch.Tensor] = field(default_factory=dict)
    linear_states: dict[int, torch.Tensor] = field(default_factory=dict)
    tokens_seen: int = 0
    device: str | None = None

    def reset(self) -> None:
        self.full_keys.clear()
        self.full_values.clear()
        self.linear_states.clear()
        self.tokens_seen = 0

    def bind(self, device: str) -> None:
        if self.device != device:
            self.reset()
            self.device = device

    def snapshot(self) -> dict[str, int | float]:
        full_tokens = sum(int(value.shape[-1]) for value in self.full_keys.values())
        linear_bytes = sum(int(value.numel() * value.element_size()) for value in self.linear_states.values())
        full_bytes = sum(int(tensor.numel() * tensor.element_size()) for tensor in [*self.full_keys.values(), *self.full_values.values()])
        return {"tokens_seen": int(self.tokens_seen), "full_layers_cached": len(self.full_keys), "full_tokens": full_tokens, "full_bytes": full_bytes, "linear_layers_cached": len(self.linear_states), "linear_bytes": linear_bytes, "bytes": full_bytes + linear_bytes}


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


def deactivate(root: Path) -> None:
    _ACTIVE_STATES.pop(root.resolve(), None)


def active(root: Path, device: str) -> AttentionState:
    key = root.resolve()
    state = _ACTIVE_STATES.get(key)
    if state is None:
        state = state_for(key, device)
        _ACTIVE_STATES[key] = state
    state.bind(device)
    return state


def _rope(position: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    if ROPE_DIM <= 0 or ROPE_DIM % 2:
        raise ValueError(f"Invalid rotary dimension: {ROPE_DIM}")
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, ROPE_DIM, 2, device=device, dtype=torch.float32) / ROPE_DIM))
    angles = float(position) * inv_freq
    emb = torch.cat((angles, angles), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, position: int) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = _rope(position, q.device, q.dtype)
    cos = cos.view(1, 1, 1, ROPE_DIM)
    sin = sin.view(1, 1, 1, ROPE_DIM)
    q_rot, q_pass = q[..., :ROPE_DIM], q[..., ROPE_DIM:]
    k_rot, k_pass = k[..., :ROPE_DIM], k[..., ROPE_DIM:]
    q_rot = q_rot * cos + _rotate_half(q_rot) * sin
    k_rot = k_rot * cos + _rotate_half(k_rot) * sin
    return torch.cat((q_rot, q_pass), dim=-1), torch.cat((k_rot, k_pass), dim=-1)


def _projection(root: Path, prefix: str, device: str) -> torch.Tensor:
    return cached._cached_load_projection(root, prefix, device)


def _linear_stateful(root: Path, layer: int, x0: torch.Tensor, device: str) -> torch.Tensor:
    state = active(root, device)
    prefix = base.layer_prefix(layer)
    input_norm = base.load_layer_weight(root, layer, "input_layernorm.weight", device)
    h = rmsnorm(x0, input_norm)
    compute_dtype = torch.float16 if device == "cuda" else torch.float32
    h_compute = h.to(dtype=compute_dtype)
    qkv_w = _projection(root, prefix + "linear_attn.in_proj_qkv", device)
    mixed = F.linear(h_compute.to(dtype=qkv_w.dtype), qkv_w).reshape(1, base.LINEAR_KEY_DIM * 2 + base.LINEAR_VALUE_DIM)
    conv_w = base.load_layer_weight(root, layer, "linear_attn.conv1d.weight", device).float()
    mixed = F.silu(mixed * conv_w[:, 0, -1].reshape(1, -1).to(dtype=mixed.dtype))
    q, k, v = torch.split(mixed, [base.LINEAR_KEY_DIM, base.LINEAR_KEY_DIM, base.LINEAR_VALUE_DIM], dim=-1)
    q = q.reshape(1, base.LINEAR_NUM_K_HEADS, 128).repeat_interleave(2, dim=1)
    k = k.reshape(1, base.LINEAR_NUM_K_HEADS, 128).repeat_interleave(2, dim=1)
    v = v.reshape(1, base.LINEAR_NUM_VALUE_HEADS, 128)
    a_w = _projection(root, prefix + "linear_attn.in_proj_a", device)
    b_w = _projection(root, prefix + "linear_attn.in_proj_b", device)
    a_log = base.load_layer_weight(root, layer, "linear_attn.A_log", device).float().reshape(1, base.LINEAR_NUM_VALUE_HEADS)
    dt_bias = base.load_layer_weight(root, layer, "linear_attn.dt_bias", device).float().reshape(1, base.LINEAR_NUM_VALUE_HEADS)
    a_raw = F.linear(h_compute.to(dtype=a_w.dtype), a_w).reshape(1, base.LINEAR_NUM_VALUE_HEADS).float()
    b_raw = F.linear(h_compute.to(dtype=b_w.dtype), b_w).reshape(1, base.LINEAR_NUM_VALUE_HEADS).float()
    beta = torch.sigmoid(b_raw)
    g = -torch.exp(a_log) * F.softplus(a_raw + dt_bias)
    decay = torch.exp(g)
    q = F.normalize(q.float(), dim=-1, eps=base.EPS) * (128 ** -0.5)
    k = F.normalize(k.float(), dim=-1, eps=base.EPS)
    linear_state = state.linear_states.get(int(layer))
    if linear_state is None or linear_state.device != x0.device:
        linear_state = torch.zeros(1, base.LINEAR_NUM_VALUE_HEADS, 128, 128, device=device, dtype=torch.float32)
    linear_state = linear_state * decay.unsqueeze(-1).unsqueeze(-1)
    retrieved = (linear_state * k.unsqueeze(-1)).sum(dim=-2)
    delta = (v.float() - retrieved) * beta.unsqueeze(-1)
    linear_state = linear_state + k.unsqueeze(-1) * delta.unsqueeze(-2)
    state.linear_states[int(layer)] = linear_state.detach()
    attn = (linear_state * q.unsqueeze(-1)).sum(dim=-2)
    z_w = _projection(root, prefix + "linear_attn.in_proj_z", device)
    z = F.linear(h_compute.to(dtype=z_w.dtype), z_w).reshape(1, base.LINEAR_NUM_VALUE_HEADS, 128)
    norm_w = base.load_layer_weight(root, layer, "linear_attn.norm.weight", device)
    gated, _, _ = gated_rmsnorm(attn, z, norm_w)
    out_w = _projection(root, prefix + "linear_attn.out_proj", device)
    gated_compute = gated.reshape(1, base.LINEAR_VALUE_DIM).to(dtype=out_w.dtype if device == "cuda" else compute_dtype)
    attn_projected = F.linear(gated_compute, out_w).float()
    residual = x0.reshape(1, base.HIDDEN).float() + attn_projected
    del input_norm, h, h_compute, qkv_w, mixed, conv_w, q, k, v, a_w, b_w, a_log, dt_bias, a_raw, b_raw, beta, g, decay, retrieved, delta, attn, z_w, z, norm_w, gated, out_w, gated_compute, attn_projected
    return residual


def _full_stateful(root: Path, layer: int, x0: torch.Tensor, device: str) -> torch.Tensor:
    state = active(root, device)
    prefix = base.layer_prefix(layer)
    position = int(state.tokens_seen)
    input_norm = base.load_layer_weight(root, layer, "input_layernorm.weight", device)
    h = rmsnorm(x0, input_norm)
    compute_dtype = torch.float16 if device == "cuda" else torch.float32
    h_compute = h.to(dtype=compute_dtype)
    q_w = _projection(root, prefix + "self_attn.q_proj", device)
    k_w = _projection(root, prefix + "self_attn.k_proj", device)
    v_w = _projection(root, prefix + "self_attn.v_proj", device)
    q_gate = F.linear(h_compute.to(dtype=q_w.dtype), q_w).reshape(1, base.FULL_NUM_HEADS, base.FULL_HEAD_DIM * 2)
    q, gate = torch.chunk(q_gate, 2, dim=-1)
    k = F.linear(h_compute.to(dtype=k_w.dtype), k_w).reshape(1, base.FULL_NUM_KV_HEADS, base.FULL_HEAD_DIM)
    v = F.linear(h_compute.to(dtype=v_w.dtype), v_w).reshape(1, base.FULL_NUM_KV_HEADS, base.FULL_HEAD_DIM)
    q_norm_w = base.load_layer_weight(root, layer, "self_attn.q_norm.weight", device)
    k_norm_w = base.load_layer_weight(root, layer, "self_attn.k_norm.weight", device)
    q = rmsnorm(q, q_norm_w).float().unsqueeze(2)
    k = rmsnorm(k, k_norm_w).float().unsqueeze(2)
    q, k_token = _apply_rope(q, k, position)
    v_token = v.float().unsqueeze(2)
    full_k = state.full_keys.get(int(layer))
    full_v = state.full_values.get(int(layer))
    if full_k is None or full_v is None or full_k.device != k_token.device:
        full_k = k_token.detach()
        full_v = v_token.detach()
    else:
        full_k = torch.cat((full_k, k_token.detach()), dim=2)
        full_v = torch.cat((full_v, v_token.detach()), dim=2)
    state.full_keys[int(layer)] = full_k
    state.full_values[int(layer)] = full_v
    k_expanded = full_k.repeat_interleave(base.FULL_NUM_KV_GROUPS, dim=1).float()
    v_expanded = full_v.repeat_interleave(base.FULL_NUM_KV_GROUPS, dim=1).float()
    q_now = q.squeeze(2)
    scores = torch.einsum("bhd,bhld->bhl", q_now, k_expanded) * (base.FULL_HEAD_DIM ** -0.5)
    attn_weights = torch.softmax(scores, dim=-1)
    attn = torch.einsum("bhl,bhld->bhd", attn_weights, v_expanded)
    if attn.shape != (1, base.FULL_NUM_HEADS, base.FULL_HEAD_DIM):
        raise RuntimeError(f"Unexpected full-attention output shape: {tuple(attn.shape)}")
    attn = attn * torch.sigmoid(gate.float())
    attn_flat = attn.reshape(1, base.FULL_Q_DIM).to(dtype=compute_dtype)
    out_w = _projection(root, prefix + "self_attn.o_proj", device)
    attn_projected = F.linear(attn_flat.to(dtype=out_w.dtype), out_w).float()
    residual = x0.reshape(1, base.HIDDEN).float() + attn_projected
    del input_norm, h, h_compute, q_w, k_w, v_w, q_gate, q, gate, k, v, q_norm_w, k_norm_w, k_token, v_token, k_expanded, v_expanded, q_now, scores, attn_weights, attn, attn_flat, out_w, attn_projected
    return residual


def step_attention(root: Path, layer: int, x0: torch.Tensor, device: str) -> torch.Tensor:
    if base.attention_type(root, layer) == "linear_attention":
        return _linear_stateful(root, layer, x0, device)
    return _full_stateful(root, layer, x0, device)


def stats(root: Path) -> dict[str, int | float]:
    state = _STATES.get(root.resolve())
    return state.snapshot() if state is not None else {"tokens_seen": 0, "full_layers_cached": 0, "full_tokens": 0, "full_bytes": 0, "linear_layers_cached": 0, "linear_bytes": 0, "bytes": 0}
