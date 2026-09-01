from __future__ import annotations

"""Isolated Layer-0 Qwen3.6 Gated RMSNorm probe."""

import argparse
import gc
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from .qwen36_delta_sequence_probe import build_token_step

EPS = 1e-6
NUM_V_HEADS = 32
HEAD_DIM = 128


def gated_rmsnorm(x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.shape != z.shape:
        raise ValueError(f"x/z shape mismatch: {tuple(x.shape)} vs {tuple(z.shape)}")
    if x.ndim != 3 or x.shape[1:] != (NUM_V_HEADS, HEAD_DIM):
        raise ValueError(f"Expected (batch, {NUM_V_HEADS}, {HEAD_DIM}), got {tuple(x.shape)}")
    if weight.numel() != HEAD_DIM:
        raise ValueError(f"Expected norm weight with {HEAD_DIM} values, got {weight.numel()}")

    x = x.float()
    z = z.float()
    w = weight.float().reshape(1, 1, HEAD_DIM)
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    normalized = (x / rms) * w
    gate = F.silu(z)
    out = normalized * gate
    return out, normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.6 isolated Layer-0 Gated RMSNorm")
    parser.add_argument("root", type=Path)
    parser.add_argument("--token-id", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")

    root = args.root.resolve()
    step = build_token_step(root, args.token_id, args.device)
    attn = step["delta_output"]
    z = step["z"]
    norm_weight = step["norm_weight"]

    print("op=gated_rmsnorm")
    print(f"token id: {args.token_id}")
    print(f"attention input shape: {tuple(attn.shape)}")
    print(f"z gate shape: {tuple(z.shape)}")
    print(f"norm weight shape: {tuple(norm_weight.shape)}")

    start = perf_counter()
    out, normalized = gated_rmsnorm(attn, z, norm_weight)
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (perf_counter() - start) * 1000.0

    def show(name: str, x: torch.Tensor) -> None:
        y = x.detach().float().cpu()
        print(
            f"{name}: shape={tuple(x.shape)} norm={torch.linalg.vector_norm(y).item():.8f} "
            f"mean={y.mean().item():.8f} std={y.std().item():.8f} "
            f"min={y.min().item():.8f} max={y.max().item():.8f}"
        )

    show("attention input", attn)
    show("z", z)
    show("silu(z)", F.silu(z))
    show("normalized", normalized)
    show("gated rmsnorm output", out)
    print(f"op=gated_rmsnorm time={elapsed_ms:.3f} ms")

    del step, attn, z, norm_weight, out, normalized
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
