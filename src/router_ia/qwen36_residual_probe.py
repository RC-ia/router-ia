from __future__ import annotations

"""Isolated Qwen3.6 Layer-0 residual probe."""

import argparse
import gc
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from .qwen36_out_proj_probe import build_gated_input
from .qwen36_op_probe import HIDDEN, LAYER_PREFIX, load_embedding_row, load_projection


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.6 isolated Layer-0 residual")
    parser.add_argument("root", type=Path)
    parser.add_argument("--token-id", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")

    root = args.root.resolve()
    x0 = load_embedding_row(root, args.token_id).to(args.device).reshape(1, HIDDEN)
    out_weight = load_projection(root, LAYER_PREFIX + "linear_attn.out_proj", args.device)
    gated_flat, _ = build_gated_input(root, args.token_id, args.device)

    if tuple(out_weight.shape) != (HIDDEN, 4096):
        raise ValueError(f"Unexpected out_proj weight shape: {tuple(out_weight.shape)}")

    start = perf_counter()
    mixer_out = F.linear(gated_flat.float(), out_weight.float())
    residual = x0.float() + mixer_out.float()
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

    print("op=residual")
    print(f"token id: {args.token_id}")
    show("input hidden", x0)
    show("attention projected", mixer_out)
    show("residual output", residual)
    print(f"residual/input norm ratio={torch.linalg.vector_norm(residual).item() / max(torch.linalg.vector_norm(x0).item(), 1e-12):.8f}")
    print(f"op=residual compute time={elapsed_ms:.3f} ms")

    del x0, out_weight, gated_flat, mixer_out, residual
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
