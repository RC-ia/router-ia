from __future__ import annotations

"""Persistent attention state for Qwen3.6 autoregressive decoding.

The model uses hybrid attention: periodic full-attention layers and Gated
DeltaNet/linear-attention layers. The reference runner previously rebuilt the
full-attention K/V tensors from scratch for every token and reset every
DeltaNet state to zeros. This module keeps the per-layer state alive across
prompt prefill and generated tokens.
"""

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from . import qwen36_40layer_loop as base
from .qwen36_gated_norm_probe import gated_rmsnorm
from .qwen36_op_probe import load_projection, rmsnorm


@dataclass
class AttentionState:
    """Autoregressive state shared by all 40 layers for one model root."""

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
            "bytes": full_bytes + linear_bytes,
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


def _linear_stateful(root: Path, layer: int, x0: torch.Tensor, device: str) -> torch.Tensor:
    state = active(root, device)
    prefix = base.layer_prefix(layer)
    input_norm = base.load_layer_weight(root, layer, "input_layernorm.weight", device)
    h = rmsnorm(x0, input_norm)
    compute_dtype = torch.float16 if device == "cuda" else torch.float32
    h_compute = h.to(dtype=compute_dtype)

    qkv_w = load_projection(root, prefix + "linear_attn.in_proj_qkv", device)
    h_qkv = h_compute.to(dtype=qkv_w.dtype)
    mixed = F.linear(h_qkv, qkv_w).reshape(1, base.LINEAR_KEY_DIM * 2 + base.LINEAR_VALUE_DIM)

    conv_w = base.load_layer_weight(root, layer, "linear_attn.conv1d.weight", device).float()
    mixed = F.silu(mixed * conv_w[:, 0, -1].reshape(1, -1).to(dtype=mixed.dtype))
    q, k, v = torch.split(mixed, [base.LINEAR_KEY_DIM, base.LINEAR_KEY_DIM, base.LINEAR_VALUE_DIM], dim=-1)
    q = q.reshape(1, base.LINEAR_NUM_K_HEADS, 128).repeat_interleave(2, dim=1)
    k = k.reshape(1, base.LINEAR_NUM_K_HEADS, 128).repeat_interleave(2, dim=1)
    v = v.reshape(1, base.LINEAR_NUM_V_HEADS, 128)

    a_w = load_projection(root, prefix + "linear_attn.in_proj_a", device)
    b_w = load_projection(root, prefix + "linear_attn.in_proj_b", device)
    a_log = base.load_layer_weight(root, layer, "linear_attn.A_log", device).float().reshape(1, base.LINEAR_NUM_V_HEADS)
    dt_bias = base.load_layer_weight(root, layer, "linear_attn.dt_bias", device).float().reshape(1, base.LINEAR_NUM_V_HEADS)
    h_a = h_compute.to(dtype=a_w.dtype)
    h_b = h_compute.to(dtype=b_w.dtype)
    a_raw = F.linear(h_a, a_w).reshape(1, base.LINEAR_NUM_V_HEADS).float()
    b_raw = F.linear(h_b, b_w).reshape(1, base.LINEAR_NUM_V_HEADS).float()
    beta = torch.sigmoid(b_raw)
    g = -torch.exp(a_log) * F.softplus(a_raw + dt_bias)
    decay = torch.exp(g)

    qn = F.normalize(q.float(), dim=-1, eps=base.EPS) * (128 ** -0.5)
    kn = F.normalize(k.float(), dim=-1, eps=base.EPS)

    linear_state = state.linear_states.get(int(layer))
    if linear_state is None or linear_state.device != x0.device:
        linear_state = torch.zeros(
            1,
            base.LINEAR_NUM_V_HEADS,
            128,
            128,
            device=device,
            dtype=torch.float32,
        )

    linear_state = linear_state * decay.unsqueeze(-1).unsqueeze(-1)
    retrieved = torch.einsum("bhkd,bhk->bhd", linear_state, kn)
    delta = (v.float() - retrieved) * beta.unsqueeze(-1)
    linear_state = linear_state + kn.unsqueeze(-1) * delta.unsqueeze(-2)
    state.linear_states[int(layer)] = linear_state.detach()
    attn = torch.einsum("bhkd,bhk->bhd", linear_state, qn)

    z_w = load_projection(root, prefix + "linear_attn.in_proj_z", device)
    z = F.linear(h_compute.to(dtype=z_w.dtype), z_w).reshape(1, base.LINEAR_NUM_V_HEADS, 128)
    norm_w = base.load_layer_weight(root, layer, "linear_attn.norm.weight", device)
    gated, _, _ = gated_rmsnorm(attn, z, norm_w)

    out_w = load_projection(root, prefix + "linear_attn.out_proj", device)
    gated_compute = gated.reshape(1, base.LINEAR_VALUE_DIM).to(dtype=out_w.dtype if device == "cuda" else compute_dtype)
    attn_projected = F.linear(gated_compute, out_w).float()
    residual = x0.reshape(1, base.HIDDEN).float() + attn_projected

    del input_norm, h, h_compute, qkv_w, h_qkv, mixed, conv_w, q, k, v
    del a_w, b_w, h_a, h_b, a_log, dt_bias, a_raw, b_raw, beta, g, decay, qn, kn
    del retrieved, delta, attn, z_w, z, norm_w, gated, out_w, gated_compute, attn_projected
    state.tokens_seen += 1
    return residual


def _full_stateful(root: Path, layer: int, x0: torch.Tensor, device: str) -> torch.Tensor:
    state = active(root, device)
    prefix = base.layer_prefix(layer)
    input_norm = base.load_layer_weight(root, layer, "input_layernorm.weight", device)
    h = rmsnorm(x0, input_norm)
    compute_dtype = torch.float16 if device == "cuda" else torch.float32
    h_compute = h.to(dtype=compute_dtype)

    q_w = load_projection(root, prefix + "self_attn.q_proj", device)
    k_w = load_projection(root, prefix + "self_attn.k_proj", device)
    v_w = load_projection(root, prefix + "self_attn.v_proj", device)

    q_gate = F.linear(h_compute.to(dtype=q_w.dtype), q_w).reshape(1, base.FULL_NUM_HEADS, base.FULL_HEAD_DIM * 2)
    q, gate = torch.chunk(q_gate, 2, dim=-1)
    k = F.linear(h_compute.to(dtype=k_w.dtype), k_w).reshape(1, base.FULL_NUM_KV_HEADS, base.FULL_HEAD_DIM)
    v = F.linear(h_compute.to(dtype=v_w.dtype), v_w).reshape(1, base.FULL_NUM_KV_HEADS, base.FULL_HEAD_DIM)

    q_norm_w = base.load_layer_weight(root, layer, "self_attn.q_norm.weight", device)
    k_norm_w = base.load_layer_weight(root, layer, "self_attn.k_norm.weight", device)
    q = rmsnorm(q, q_norm_w).float()
    k = rmsnorm(k, k_norm_w).float()

    full_k = state.full_keys.get(int(layer))
    full_v = state.full_values.get(int(layer))
    k_token = k.unsqueeze(2)
    v_token = v.unsqueeze(2)
    if full_k is None or full_v is None or full_k.device != k.device:
        full_k = k_token.detach()
        full_v = v_token.detach()
    else:
        full_k = torch.cat((full_k, k_token), dim=2)
        full_v = torch.cat((full_v, v_token), dim=2)

    state.full_keys[int(layer)] = full_k.detach()
    state.full_values[int(layer)] = full_v.detach()

    k_expanded = full_k.repeat_interleave(base.FULL_NUM_KV_GROUPS, dim=1)
    v_expanded = full_v.repeat_interleave(base.FULL_NUM_KV_GROUPS, dim=1)
    scores = torch.matmul(q.unsqueeze(2), k_expanded.transpose(-1, -2)).squeeze(-2) * (base.FULL_HEAD_DIM ** -0.5)
    attn_weights = torch.softmax(scores.float(), dim=-1)
    attn = torch.matmul(attn_weights.unsqueeze(-2), v_expanded).squeeze(-2)
    attn = attn * torch.sigmoid(gate.float())
    attn_flat = attn.reshape(1, base.FULL_Q_DIM).to(dtype=compute_dtype)

    out_w = load_projection(root, prefix + "self_attn.o_proj", device)
    attn_projected = F.linear(attn_flat.to(dtype=out_w.dtype), out_w).float()
    residual = x0.reshape(1, base.HIDDEN).float() + attn_projected

    del input_norm, h, h_compute, q_w, k_w, v_w, q_gate, q, gate, k, v
    del k_token, v_token, k_expanded, v_expanded, scores, attn_weights, attn
    del attn_flat, out_w, attn_projected
    return residual


def step_attention(root: Path, layer: int, x0: torch.Tensor, device: str) -> torch.Tensor:
    """Run the correct attention type using persistent per-layer state."""
    if base.attention_type(root, layer) == "linear_attention":
        return _linear_stateful(root, layer, x0, device)
    return _full_stateful(root, layer, x0, device)


def stats(root: Path) -> dict[str, int | float]:
    state = _STATES.get(root.resolve())
    return state.snapshot() if state is not None else {
        "tokens_seen": 0,
        "full_layers_cached": 0,
        "full_tokens": 0,
        "full_bytes": 0,
        "linear_layers_cached": 0,
        "linear_bytes": 0,
        "bytes": 0,
    }
