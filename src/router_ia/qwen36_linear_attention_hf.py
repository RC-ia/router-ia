from __future__ import annotations

"""Reference-style Qwen3.6 linear attention implementation.

This module follows the PyTorch fallback structure used by Hugging Face for
Qwen-family Gated DeltaNet: project -> causal depthwise conv -> reshape/repeat
-> beta/g -> recurrent gated-delta update -> gated RMSNorm -> out projection.

It is intentionally independent from qwen36_attention_cache.py so it can be
used as a clean reference candidate before changing the production runtime.
Weights are dequantized directly to BF16 on CUDA when stored as FP8, avoiding
the FP16 materialization used by the generic cached projection path.
"""

from pathlib import Path
import json

import torch
import torch.nn.functional as F
from safetensors import safe_open

from . import qwen36_40layer_loop as base
from .qwen36_dequant import dequantize_fp8_blockwise
from .qwen36_gated_norm_probe import gated_rmsnorm
from .qwen36_op_probe import rmsnorm


CONV_KERNEL = 4
HEAD_DIM = 128


def _raw_weight(root: Path, name: str) -> tuple[torch.Tensor, torch.Tensor | None]:
    index = root / "model.safetensors.index.json"
    if index.is_file():
        payload = json.loads(index.read_text(encoding="utf-8"))
        shard_names = [payload["weight_map"][name]]
    else:
        shard_names = [p.name for p in sorted(root.glob("*.safetensors"))]

    scale_name = name.replace(".weight", ".weight_scale_inv")
    for shard_name in shard_names:
        shard = root / shard_name
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            if name not in handle.keys():
                continue
            weight = handle.get_tensor(name)
            scale = handle.get_tensor(scale_name) if scale_name in handle.keys() else None
            return weight, scale
    raise KeyError(f"Missing tensor: {name}")


def load_linear_weight(root: Path, prefix: str, device: str, *, dtype: torch.dtype) -> torch.Tensor:
    """Load one linear weight, preserving BF16 instead of routing through FP16."""
    name = prefix + ".weight"
    weight, scale = _raw_weight(root, name)
    if weight.dtype == torch.float8_e4m3fn:
        if scale is None:
            raise RuntimeError(f"Missing FP8 scale for {name}")
        weight = dequantize_fp8_blockwise(
            weight.to(device=device),
            scale.to(device=device),
        )
    else:
        weight = weight.to(device=device)
    return weight.to(dtype=dtype)


def load_vector(root: Path, layer: int, suffix: str, device: str) -> torch.Tensor:
    return base.load_layer_weight(root, layer, suffix, device).float().reshape(-1)


def causal_conv_step(
    mixed_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """HF-style one-token causal depthwise convolution."""
    if mixed_qkv.ndim != 2 or mixed_qkv.shape[0] != 1:
        raise ValueError(f"Expected mixed_qkv [1,C], got {tuple(mixed_qkv.shape)}")
    if conv_weight.ndim != 3 or conv_weight.shape[1] != 1:
        raise ValueError(f"Expected depthwise conv weight [C,1,K], got {tuple(conv_weight.shape)}")
    if conv_state.ndim != 3:
        raise ValueError(f"Expected conv state [B,C,K], got {tuple(conv_state.shape)}")

    state_len = int(conv_state.shape[-1])
    x = mixed_qkv.reshape(1, mixed_qkv.shape[-1], 1).to(dtype=conv_weight.dtype)
    history = torch.cat((conv_state.to(dtype=conv_weight.dtype), x), dim=-1)
    new_state = history[:, :, -state_len:].detach()
    out = F.conv1d(
        history,
        conv_weight,
        bias=None,
        stride=1,
        padding=0,
        groups=history.shape[1],
    )[:, :, -1:]
    out = F.silu(out)
    return out[:, :, 0].to(dtype=mixed_qkv.dtype), new_state


def gated_delta_recurrent(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """HF Torch-fallback recurrent gated-delta rule for one or more tokens.

    Inputs use [B,S,H,D] for q/k/v and [B,S,H] for g/beta. Internally all
    tensors and the recurrent state are FP32, matching the HF fallback.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query/key/value must be [B,S,H,D]")
    if g.ndim != 3 or beta.ndim != 3:
        raise ValueError("g/beta must be [B,S,H]")

    batch, seq_len, num_heads, k_dim = query.shape
    v_heads = value.shape[2]
    v_dim = value.shape[3]

    q = query.transpose(1, 2).contiguous().float()
    k = key.transpose(1, 2).contiguous().float()
    v = value.transpose(1, 2).contiguous().float()
    decay = g.transpose(1, 2).contiguous().float()
    beta_f = beta.transpose(1, 2).contiguous().float()

    q = q / (k_dim ** 0.5)
    q = q / torch.sqrt((q * q).sum(dim=-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt((k * k).sum(dim=-1, keepdim=True) + 1e-6)

    if state is None:
        recurrent = torch.zeros(
            (batch, v_heads, k_dim, v_dim),
            device=value.device,
            dtype=torch.float32,
        )
    else:
        recurrent = state.to(device=value.device, dtype=torch.float32)

    outputs = torch.empty_like(v, dtype=torch.float32)
    for i in range(seq_len):
        q_t = q[:, :, i]
        k_t = k[:, :, i]
        v_t = v[:, :, i]
        decay_t = decay[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta_f[:, :, i].unsqueeze(-1)

        recurrent = recurrent * decay_t
        kv_mem = (recurrent * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        recurrent = recurrent + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs[:, :, i] = (recurrent * q_t.unsqueeze(-1)).sum(dim=-2)

    return outputs.transpose(1, 2).contiguous().to(query.dtype), recurrent.detach()


def linear_attention_step(
    root: Path,
    layer: int,
    x0: torch.Tensor,
    conv_state: torch.Tensor | None,
    recurrent_state: torch.Tensor | None,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a complete reference-style one-token Qwen3.6 linear-attention step.

    Returns (linear_output, new_conv_state, new_recurrent_state). The returned
    linear output does not include the residual connection.
    """
    if x0.shape != (1, base.HIDDEN):
        raise ValueError(f"Expected x0 [1,{base.HIDDEN}], got {tuple(x0.shape)}")

    compute_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    x = x0.to(device=device, dtype=compute_dtype)
    prefix = base.layer_prefix(layer)

    input_norm = base.load_layer_weight(root, layer, "input_layernorm.weight", device)
    h = rmsnorm(x, input_norm)

    qkv_w = load_linear_weight(root, prefix + "linear_attn.in_proj_qkv", device, dtype=compute_dtype)
    z_w = load_linear_weight(root, prefix + "linear_attn.in_proj_z", device, dtype=compute_dtype)
    b_w = load_linear_weight(root, prefix + "linear_attn.in_proj_b", device, dtype=compute_dtype)
    a_w = load_linear_weight(root, prefix + "linear_attn.in_proj_a", device, dtype=compute_dtype)
    out_w = load_linear_weight(root, prefix + "linear_attn.out_proj", device, dtype=compute_dtype)

    mixed = F.linear(h, qkv_w).reshape(1, base.LINEAR_CONV_DIM)
    conv_w = base.load_layer_weight(root, layer, "linear_attn.conv1d.weight", device).to(dtype=compute_dtype)
    if conv_state is None or tuple(conv_state.shape) != (1, base.LINEAR_CONV_DIM, CONV_KERNEL):
        conv_state = torch.zeros(
            (1, base.LINEAR_CONV_DIM, CONV_KERNEL),
            device=x.device,
            dtype=compute_dtype,
        )
    mixed, conv_state_new = causal_conv_step(mixed, conv_state, conv_w)

    q_flat, k_flat, v_flat = torch.split(
        mixed,
        [base.LINEAR_KEY_DIM, base.LINEAR_KEY_DIM, base.LINEAR_VALUE_DIM],
        dim=-1,
    )
    q = q_flat.reshape(1, base.LINEAR_NUM_K_HEADS, HEAD_DIM)
    k = k_flat.reshape(1, base.LINEAR_NUM_K_HEADS, HEAD_DIM)
    v = v_flat.reshape(1, base.LINEAR_NUM_V_HEADS, HEAD_DIM)

    q = q.repeat_interleave(base.LINEAR_NUM_V_HEADS // base.LINEAR_NUM_K_HEADS, dim=1)
    k = k.repeat_interleave(base.LINEAR_NUM_V_HEADS // base.LINEAR_NUM_K_HEADS, dim=1)

    a = F.linear(h, a_w).reshape(1, base.LINEAR_NUM_V_HEADS).float()
    b = F.linear(h, b_w).reshape(1, base.LINEAR_NUM_V_HEADS)
    beta = torch.sigmoid(b)

    a_log = load_vector(root, layer, "linear_attn.A_log", device).reshape(1, base.LINEAR_NUM_V_HEADS)
    dt_bias = load_vector(root, layer, "linear_attn.dt_bias", device).reshape(1, base.LINEAR_NUM_V_HEADS)
    g = -torch.exp(a_log) * F.softplus(a + dt_bias)

    recurrent_q = q.unsqueeze(1)
    recurrent_k = k.unsqueeze(1)
    recurrent_v = v.unsqueeze(1)
    recurrent_g = g.unsqueeze(1)
    recurrent_beta = beta.unsqueeze(1)

    core, recurrent_state_new = gated_delta_recurrent(
        recurrent_q,
        recurrent_k,
        recurrent_v,
        recurrent_g,
        recurrent_beta,
        recurrent_state,
    )
    attn = core[:, 0]

    z = F.linear(h, z_w).reshape(1, base.LINEAR_NUM_V_HEADS, HEAD_DIM)
    gated, _, _ = gated_rmsnorm(attn, z, base.load_layer_weight(root, layer, "linear_attn.norm.weight", device))

    projected = F.linear(gated.reshape(1, base.LINEAR_VALUE_DIM), out_w).float()
    return projected, conv_state_new, recurrent_state_new


__all__ = [
    "gated_delta_recurrent",
    "linear_attention_step",
    "causal_conv_step",
    "load_linear_weight",
]
