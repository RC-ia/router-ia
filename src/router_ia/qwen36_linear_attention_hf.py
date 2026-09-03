from __future__ import annotations

"""Reference-style Qwen3.6 linear attention implementation."""

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
LINEAR_CONV_DIM = base.LINEAR_KEY_DIM * 2 + base.LINEAR_VALUE_DIM


def _raw_weight(root: Path, name: str) -> tuple[torch.Tensor, torch.Tensor | None]:
    index = root / "model.safetensors.index.json"
    if index.is_file():
        payload = json.loads(index.read_text(encoding="utf-8"))
        shard_names = [payload["weight_map"][name]]
    else:
        shard_names = [p.name for p in sorted(root.glob("*.safetensors"))]
    scale_name = name.replace(".weight", ".weight_scale_inv")
    for shard_name in shard_names:
        with safe_open(str(root / shard_name), framework="pt", device="cpu") as handle:
            if name in handle.keys():
                weight = handle.get_tensor(name)
                scale = handle.get_tensor(scale_name) if scale_name in handle.keys() else None
                return weight, scale
    raise KeyError(f"Missing tensor: {name}")


def _raw_tensor(root: Path, name: str) -> torch.Tensor:
    index = root / "model.safetensors.index.json"
    if index.is_file():
        payload = json.loads(index.read_text(encoding="utf-8"))
        shard_names = [payload["weight_map"][name]]
    else:
        shard_names = [p.name for p in sorted(root.glob("*.safetensors"))]
    for shard_name in shard_names:
        with safe_open(str(root / shard_name), framework="pt", device="cpu") as handle:
            if name in handle.keys():
                return handle.get_tensor(name)
    raise KeyError(f"Missing tensor: {name}")


def load_linear_weight(root: Path, prefix: str, device: str, *, dtype: torch.dtype) -> torch.Tensor:
    name = prefix + ".weight"
    weight, scale = _raw_weight(root, name)
    if weight.dtype == torch.float8_e4m3fn:
        if scale is None:
            raise RuntimeError(f"Missing FP8 scale for {name}")
        weight = dequantize_fp8_blockwise(weight.to(device=device), scale.to(device=device))
    else:
        weight = weight.to(device=device)
    return weight.to(dtype=dtype)


def load_vector(root: Path, layer: int, suffix: str, device: str) -> torch.Tensor:
    """Load scalar/vector parameters without the generic FP16 projection cast."""
    return _raw_tensor(root, base.layer_prefix(layer) + suffix).to(device=device)


def causal_conv_step(mixed_qkv: torch.Tensor, conv_state: torch.Tensor, conv_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    state_len = int(conv_state.shape[-1])
    x = mixed_qkv.reshape(1, mixed_qkv.shape[-1], 1).to(dtype=conv_weight.dtype)
    history = torch.cat((conv_state.to(dtype=conv_weight.dtype), x), dim=-1)
    new_state = history[:, :, -state_len:].detach()
    out = F.conv1d(history, conv_weight, bias=None, stride=1, padding=0, groups=history.shape[1])[:, :, -1:]
    out = F.silu(out)
    return out[:, :, 0].to(dtype=mixed_qkv.dtype), new_state


def gated_delta_chunk_initial(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use Transformers' exact chunk implementation for the first token.

    Qwen3.6 starts its linear-attention cache through the chunk kernel. Even for
    a single token, its reduction order can differ from the scalar recurrent
    loop enough to change the cached FP32 state at ~1e-3 scale.
    """
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf

    output, final_state = hf.torch_chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    if final_state is None:
        raise RuntimeError("HF chunk rule did not return final recurrent state")
    return output, final_state.detach()


def gated_delta_recurrent(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """HF-style recurrent gated-delta update using FP32 internal state/math."""
    batch, seq_len, num_heads, k_dim = query.shape
    v_heads, v_dim = value.shape[2], value.shape[3]
    initial_dtype = query.dtype
    q = query.transpose(1, 2).contiguous().float()
    k = key.transpose(1, 2).contiguous().float()
    v = value.transpose(1, 2).contiguous().float()
    g = g.transpose(1, 2).contiguous().float()
    beta = beta.transpose(1, 2).contiguous().float()

    q = q / torch.sqrt((q * q).sum(dim=-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt((k * k).sum(dim=-1, keepdim=True) + 1e-6)
    q = q * (k_dim ** -0.5)

    if state is None:
        recurrent = torch.zeros((batch, v_heads, k_dim, v_dim), device=value.device, dtype=torch.float32)
    else:
        recurrent = state.to(device=value.device, dtype=torch.float32)

    outputs = torch.empty((batch, v_heads, seq_len, v_dim), device=value.device, dtype=torch.float32)
    for i in range(seq_len):
        q_t = q[:, :, i]
        k_t = k[:, :, i]
        v_t = v[:, :, i]
        decay_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)
        recurrent = recurrent * decay_t
        kv_mem = (recurrent * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        recurrent = recurrent + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs[:, :, i] = (recurrent * q_t.unsqueeze(-1)).sum(dim=-2)
    return outputs.transpose(1, 2).contiguous().to(dtype=initial_dtype), recurrent.detach()


def linear_attention_step(root: Path, layer: int, x0: torch.Tensor, conv_state: torch.Tensor | None, recurrent_state: torch.Tensor | None, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    compute_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    x = x0.to(device=device, dtype=compute_dtype)
    prefix = base.layer_prefix(layer)

    h = rmsnorm(x, base.load_layer_weight(root, layer, "input_layernorm.weight", device))
    qkv_w = load_linear_weight(root, prefix + "linear_attn.in_proj_qkv", device, dtype=compute_dtype)
    z_w = load_linear_weight(root, prefix + "linear_attn.in_proj_z", device, dtype=compute_dtype)
    b_w = load_linear_weight(root, prefix + "linear_attn.in_proj_b", device, dtype=compute_dtype)
    a_w = load_linear_weight(root, prefix + "linear_attn.in_proj_a", device, dtype=compute_dtype)
    out_w = load_linear_weight(root, prefix + "linear_attn.out_proj", device, dtype=compute_dtype)

    mixed = F.linear(h, qkv_w).reshape(1, LINEAR_CONV_DIM)
    conv_w = base.load_layer_weight(root, layer, "linear_attn.conv1d.weight", device).to(dtype=compute_dtype)
    if conv_state is None or tuple(conv_state.shape) != (1, LINEAR_CONV_DIM, CONV_KERNEL):
        conv_state = torch.zeros((1, LINEAR_CONV_DIM, CONV_KERNEL), device=x.device, dtype=compute_dtype)
    mixed, conv_state_new = causal_conv_step(mixed, conv_state, conv_w)

    q_flat, k_flat, v_flat = torch.split(mixed, [base.LINEAR_KEY_DIM, base.LINEAR_KEY_DIM, base.LINEAR_VALUE_DIM], dim=-1)
    q = q_flat.reshape(1, base.LINEAR_NUM_K_HEADS, HEAD_DIM).repeat_interleave(base.LINEAR_NUM_V_HEADS // base.LINEAR_NUM_K_HEADS, dim=1)
    k = k_flat.reshape(1, base.LINEAR_NUM_K_HEADS, HEAD_DIM).repeat_interleave(base.LINEAR_NUM_V_HEADS // base.LINEAR_NUM_K_HEADS, dim=1)
    v = v_flat.reshape(1, base.LINEAR_NUM_V_HEADS, HEAD_DIM)

    a = F.linear(h, a_w).reshape(1, base.LINEAR_NUM_V_HEADS).float()
    b = F.linear(h, b_w).reshape(1, base.LINEAR_NUM_V_HEADS)
    beta = torch.sigmoid(b)
    a_log = load_vector(root, layer, "linear_attn.A_log", device).reshape(1, base.LINEAR_NUM_V_HEADS)
    dt_bias = load_vector(root, layer, "linear_attn.dt_bias", device).reshape(1, base.LINEAR_NUM_V_HEADS)
    g = -a_log.float().exp() * F.softplus(a + dt_bias)

    if recurrent_state is None:
        core, recurrent_state_new = gated_delta_chunk_initial(
            q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), g.unsqueeze(1), beta.unsqueeze(1)
        )
    else:
        core, recurrent_state_new = gated_delta_recurrent(
            q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), g.unsqueeze(1), beta.unsqueeze(1), recurrent_state
        )
    attn = core[:, 0]

    z = F.linear(h, z_w).reshape(1, base.LINEAR_NUM_V_HEADS, HEAD_DIM)
    norm_w = base.load_layer_weight(root, layer, "linear_attn.norm.weight", device)
    gated, _, _ = gated_rmsnorm(attn, z, norm_w)
    projected = F.linear(gated.reshape(1, base.LINEAR_VALUE_DIM).to(dtype=out_w.dtype), out_w).float()
    return projected, conv_state_new, recurrent_state_new


__all__ = ["gated_delta_recurrent", "gated_delta_chunk_initial", "linear_attention_step", "causal_conv_step", "load_linear_weight"]
