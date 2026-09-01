from __future__ import annotations

"""Isolated Qwen3.6 Layer-0 out_proj probe."""

import argparse
import gc
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from .qwen36_gated_norm_probe import gated_rmsnorm
from .qwen36_delta_sequence_probe import token_params
from .qwen36_op_probe import HEAD_DIM, LAYER_PREFIX, NUM_V_HEADS, load_embedding_row, load_projection, load_tensor, rmsnorm

HIDDEN = 2048
VALUE_DIM = NUM_V_HEADS * HEAD_DIM


def build_gated_input(root: Path, token_id: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_norm_weight = load_tensor(root, LAYER_PREFIX + "input_layernorm.weight", device=device)
    a_weight = load_projection(root, LAYER_PREFIX + "linear_attn.in_proj_a", device)
    b_weight = load_projection(root, LAYER_PREFIX + "linear_attn.in_proj_b", device)
    a_log = load_tensor(root, LAYER_PREFIX + "linear_attn.A_log", device=device).float().reshape(1, NUM_V_HEADS)
    dt_bias = load_tensor(root, LAYER_PREFIX + "linear_attn.dt_bias", device=device).float().reshape(1, NUM_V_HEADS)

    q, k, v, beta, g, decay = token_params(root, token_id, device, input_norm_weight, a_weight, b_weight, a_log, dt_bias)
    state = torch.zeros(1, NUM_V_HEADS, HEAD_DIM, HEAD_DIM, device=device, dtype=torch.float32)
    state = state * decay.unsqueeze(-1).unsqueeze(-1)
    retrieved = torch.einsum("bhkd,bhk->bhd", state, k)
    delta = (v - retrieved) * beta.unsqueeze(-1)
    state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
    attn = torch.einsum("bhkd,bhk->bhd", state, q)

    h = rmsnorm(load_embedding_row(root, token_id).to(device), input_norm_weight)
    z_weight = load_projection(root, LAYER_PREFIX + "linear_attn.in_proj_z", device)
    z = F.linear(h.float(), z_weight.float()).reshape(1, NUM_V_HEADS, HEAD_DIM)
    norm_weight = load_tensor(root, LAYER_PREFIX + "linear_attn.norm.weight", device=device)
    gated, _, _ = gated_rmsnorm(attn, z, norm_weight)

    del input_norm_weight, a_weight, b_weight, a_log, dt_bias, q, k, v, beta, g, decay, state, retrieved, delta, h, z_weight, norm_weight, attn, z
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return gated.reshape(1, VALUE_DIM), gated


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.6 isolated Layer-0 out_proj")
    parser.add_argument("root", type=Path)
    parser.add_argument("--token-id", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")

    root = args.root.resolve()
    out_weight = load_projection(root, LAYER_PREFIX + "linear_attn.out_proj", args.device)
    x_flat, x_heads = build_gated_input(root, args.token_id, args.device)

    if out_weight.ndim != 2 or tuple(out_weight.shape) != (HIDDEN, VALUE_DIM):
        raise ValueError(f"Unexpected out_proj weight shape: {tuple(out_weight.shape)}; expected {(HIDDEN, VALUE_DIM)}")

    print("op=out_proj")
    print(f"token id: {args.token_id}")
    print(f"input head shape: {tuple(x_heads.shape)}")
    print(f"flattened input shape: {tuple(x_flat.shape)}")
    print(f"out_proj weight shape: {tuple(out_weight.shape)} dtype={out_weight.dtype}")

    start = perf_counter()
    y = F.linear(x_flat.float(), out_weight.float())
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (perf_counter() - start) * 1000.0

    def show(name: str, x: torch.Tensor) -> None:
        t = x.detach().float().cpu()
        print(
            f"{name}: shape={tuple(x.shape)} norm={torch.linalg.vector_norm(t).item():.8f} "
            f"mean={t.mean().item():.8f} std={t.std().item():.8f} "
            f"min={t.min().item():.8f} max={t.max().item():.8f}"
        )

    show("gated input", x_flat)
    show("out_proj output", y)
    print(f"op=out_proj compute time={elapsed_ms:.3f} ms")

    del out_weight, x_flat, x_heads, y
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
