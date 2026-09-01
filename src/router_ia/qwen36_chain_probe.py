from __future__ import annotations

"""Execute consecutive Qwen3.6 layers for a single token."""

import argparse
import gc
from pathlib import Path
from time import perf_counter

import torch

from .qwen36_layer_executor import execute_layer, HIDDEN
from .qwen36_op_probe import load_embedding_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen3.6 Layer 0 -> Layer N")
    parser.add_argument("root", type=Path)
    parser.add_argument("--token-id", type=int, default=0)
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--end-layer", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    if args.start_layer < 0 or args.end_layer < args.start_layer:
        raise SystemExit("Intervalo de camadas inválido")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")

    root = args.root.resolve()
    x = load_embedding_row(root, args.token_id).reshape(1, HIDDEN).to(args.device)

    print("op=chain")
    print(f"token id: {args.token_id}")
    print(f"layers: {args.start_layer}..{args.end_layer}")
    print(f"device: {args.device}")
    print(f"input shape: {tuple(x.shape)}")
    print(f"input norm: {torch.linalg.vector_norm(x).item():.8f}")

    for layer in range(args.start_layer, args.end_layer + 1):
        start = perf_counter()
        out, info = execute_layer(root, layer, x, args.device, args.top_k)
        if args.device == "cuda":
            torch.cuda.synchronize()
        elapsed = (perf_counter() - start) * 1000.0

        print(f"layer {layer}:")
        print(f"  router top-{args.top_k}: {info['expert_ids']}")
        print(f"  router weights: {[round(w, 8) for w in info['router_weights']]}")
        print(f"  shared gate: {info['shared_gate']:.8f}")
        print(f"  moe input norm: {info['moe_input_norm']:.8f}")
        print(f"  output shape: {tuple(out.shape)}")
        print(f"  output norm: {torch.linalg.vector_norm(out).item():.8f}")
        print(f"  output mean: {out.mean().item():.8f}")
        print(f"  time: {elapsed:.3f} ms")

        del x
        x = out
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    print(f"final output shape: {tuple(x.shape)}")
    print(f"final output norm: {torch.linalg.vector_norm(x).item():.8f}")
    print(f"final output mean: {x.mean().item():.8f}")
    print(f"final output min: {x.min().item():.8f}")
    print(f"final output max: {x.max().item():.8f}")

    del x
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
