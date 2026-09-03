from __future__ import annotations

"""Isolated Qwen3.6 Gated RMSNorm implementation/probe."""

import argparse
import gc
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from .qwen36_delta_sequence_probe import token_params
from .qwen36_op_probe import HEAD_DIM, LAYER_PREFIX, NUM_V_HEADS, load_projection, load_tensor

EPS = 1e-6


def gated_rmsnorm(x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match Qwen3.5/3.6 RMSNormGated dtype/order semantics."""
    if x.shape != z.shape:
        raise ValueError(f"x/z shape mismatch: {tuple(x.shape)} vs {tuple(z.shape)}")
    if x.ndim != 3 or x.shape[1:] != (NUM_V_HEADS, HEAD_DIM):
        raise ValueError(f"Expected (batch, {NUM_V_HEADS}, {HEAD_DIM}), got {tuple(x.shape)}")
    if weight.numel() != HEAD_DIM:
        raise ValueError(f"Expected norm weight with {HEAD_DIM} values, got {weight.numel()}")

    input_dtype = x.dtype
    x_fp32 = x.float()
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    normalized_fp32 = x_fp32 * torch.rsqrt(variance + EPS)
    normalized = normalized_fp32.to(input_dtype)
    weighted = weight.reshape(1, 1, HEAD_DIM) * normalized
    gate = F.silu(z.float())
    out = (weighted * gate).to(input_dtype)
    return out, weighted, gate


def build_inputs(root: Path, token_id: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    norm_weight = load_tensor(root, LAYER_PREFIX + "input_layernorm.weight", device=device)
    a_weight = load_projection(root, LAYER_PREFIX + "linear_attn.in_proj_a", device)
    b_weight = load_projection(root, LAYER_PREFIX + "linear_attn.in_proj_b", device)
    a_log = load_tensor(root, LAYER_PREFIX + "linear_attn.A_log", device=device).float().reshape(1, NUM_V_HEADS)
    dt_bias = load_tensor(root, LAYER_PREFIX + "linear_attn.dt_bias", device=device).float().reshape(1, NUM_V_HEADS)
    q, k, v, beta, g, decay = token_params(root, token_id, device, norm_weight, a_weight, b_weight, a_log, dt_bias)
    state = torch.zeros(1, NUM_V_HEADS, HEAD_DIM, HEAD_DIM, device=device, dtype=torch.float32)
    state = state * decay.unsqueeze(-1).unsqueeze(-1)
    retrieved = torch.einsum("bhkd,bhk->bhd", state, k)
    delta = (v - retrieved) * beta.unsqueeze(-1)
    state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
    attn = torch.einsum("bhkd,bhk->bhd", state, q)
    z_weight = load_projection(root, LAYER_PREFIX + "linear_attn.in_proj_z", device)
    from .qwen36_op_probe import load_embedding_row, rmsnorm
    h = rmsnorm(load_embedding_row(root, token_id).to(device), norm_weight)
    z = F.linear(h.float(), z_weight.float()).reshape(1, NUM_V_HEADS, HEAD_DIM)
    out_norm_weight = load_tensor(root, LAYER_PREFIX + "linear_attn.norm.weight", device=device)
    del q, k, v, beta, g, decay, state, retrieved, delta, norm_weight, a_weight, b_weight, a_log, dt_bias, z_weight, h
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return attn, z, out_norm_weight


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.6 isolated Gated RMSNorm")
    parser.add_argument("root", type=Path)
    parser.add_argument("--token-id", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    root = args.root.resolve()
    attn, z, norm_weight = build_inputs(root, args.token_id, args.device)
    print("op=gated_rmsnorm")
    print(f"token id: {args.token_id}")
    print(f"attention input shape: {tuple(attn.shape)}")
    print(f"z gate shape: {tuple(z.shape)}")
    print(f"norm weight shape: {tuple(norm_weight.shape)}")
    start = perf_counter()
    out, normalized, gate = gated_rmsnorm(attn, z, norm_weight)
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (perf_counter() - start) * 1000.0
    def show(name: str, x: torch.Tensor) -> None:
        y = x.detach().float().cpu()
        print(f"{name}: shape={tuple(x.shape)} norm={torch.linalg.vector_norm(y).item():.8f} mean={y.mean().item():.8f} std={y.std().item():.8f} min={y.min().item():.8f} max={y.max().item():.8f}")
    show("attention input", attn)
    show("z", z)
    show("silu(z)", gate)
    show("normalized", normalized)
    show("gated rmsnorm output", out)
    print(f"op=gated_rmsnorm time={elapsed_ms:.3f} ms")
    del attn, z, norm_weight, out, normalized, gate
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
