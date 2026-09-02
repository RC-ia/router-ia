from __future__ import annotations

"""Profile the existing Qwen3.6 reference loop without changing its math.

The profiler wraps the existing tensor loader and FP8 dequantizer, then runs
exactly the same layer functions as qwen36_40layer_loop.py. For CPU runs,
this lets us split each layer into:
  - I/O/materialization time from load_tensor()
  - FP8 dequantization time
  - remaining layer calculation time

The existing reference runner is not modified.
"""

import argparse
import gc
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import torch

from . import qwen36_40layer_loop as loop


class Profile:
    def __init__(self) -> None:
        self.io_s = 0.0
        self.dequant_s = 0.0
        self.io_calls = 0
        self.dequant_calls = 0
        self.io_bytes = 0
        self.dequant_bytes = 0
        self.layer = defaultdict(lambda: {"io": 0.0, "dequant": 0.0})
        self.active_layer: int | None = None


PROFILE = Profile()


_original_load_tensor = loop.load_tensor
_original_dequant = loop.dequantize_fp8_blockwise


def profiled_load_tensor(root: Path, name: str, device: str):
    start = perf_counter()
    tensor = _original_load_tensor(root, name, device=device)
    elapsed = perf_counter() - start
    PROFILE.io_s += elapsed
    PROFILE.io_calls += 1
    PROFILE.io_bytes += tensor.numel() * tensor.element_size()
    if PROFILE.active_layer is not None:
        PROFILE.layer[PROFILE.active_layer]["io"] += elapsed
    return tensor


def profiled_dequant(weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    start = perf_counter()
    out = _original_dequant(weight, scale_inv)
    elapsed = perf_counter() - start
    PROFILE.dequant_s += elapsed
    PROFILE.dequant_calls += 1
    PROFILE.dequant_bytes += out.numel() * out.element_size()
    if PROFILE.active_layer is not None:
        PROFILE.layer[PROFILE.active_layer]["dequant"] += elapsed
    return out


# The loop imported these functions directly, so wrapping its module symbols
# is enough to instrument every existing load/dequant call without changing
# the implementation being profiled.
loop.load_tensor = profiled_load_tensor
loop.dequantize_fp8_blockwise = profiled_dequant


def mib(value: int) -> float:
    return value / (1024.0 * 1024.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile Qwen3.6 reference loop")
    parser.add_argument("root", type=Path)
    parser.add_argument("--token-id", type=int, default=0)
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--end-layer", type=int, default=loop.DEFAULT_LAYERS - 1)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    if not 0 <= args.start_layer <= args.end_layer < loop.DEFAULT_LAYERS:
        raise SystemExit(f"layer range must be inside 0..{loop.DEFAULT_LAYERS - 1}")

    root = args.root.resolve()
    x = loop.load_embedding_row(root, args.token_id).reshape(1, loop.HIDDEN).to(args.device).float()

    print("op=profile")
    print(f"token id: {args.token_id}")
    print(f"layers: {args.start_layer}..{args.end_layer}")
    print(f"device: {args.device}")
    print(f"input shape: {tuple(x.shape)}")
    print(f"input norm: {torch.linalg.vector_norm(x).item():.8f}")
    print()
    print("per-layer breakdown: io + dequant + compute = total")

    total_start = perf_counter()
    total_layer_s = 0.0
    total_io_s = 0.0
    total_dequant_s = 0.0
    total_compute_s = 0.0
    total_type_s = 0.0

    for layer in range(args.start_layer, args.end_layer + 1):
        x_before = x
        PROFILE.active_layer = layer

        type_start = perf_counter()
        kind = loop.attention_type(root, layer)
        type_s = perf_counter() - type_start
        total_type_s += type_s

        layer_start = perf_counter()
        if kind == "linear_attention":
            residual = loop.linear_attention_step(root, layer, x_before, args.device)
        else:
            residual = loop.full_attention_step(root, layer, x_before, args.device)
        x, expert_ids, weights, shared_gate, moe_input_norm = loop.moe_step(
            root, layer, residual, args.top_k, args.device
        )
        if args.device == "cuda":
            torch.cuda.synchronize()
        layer_s = perf_counter() - layer_start

        io_s = PROFILE.layer[layer]["io"]
        dequant_s = PROFILE.layer[layer]["dequant"]
        # The remaining sequential time is the actual layer work plus small
        # Python/GC overhead. This is the useful "compute" bucket for finding
        # the dominant bottleneck before deeper instrumentation.
        compute_s = max(0.0, layer_s - io_s - dequant_s)

        total_layer_s += layer_s
        total_io_s += io_s
        total_dequant_s += dequant_s
        total_compute_s += compute_s

        print(
            f"layer {layer:02d} ({kind}): "
            f"io={io_s * 1000.0:.3f} ms | "
            f"dequant={dequant_s * 1000.0:.3f} ms | "
            f"compute={compute_s * 1000.0:.3f} ms | "
            f"total={layer_s * 1000.0:.3f} ms | "
            f"experts={expert_ids}"
        )
        del x_before, residual
        gc.collect()

    PROFILE.active_layer = None
    if args.device == "cuda":
        torch.cuda.synchronize()
    wall_s = perf_counter() - total_start

    print()
    print("summary:")
    print(f"layers: {args.end_layer - args.start_layer + 1}")
    print(f"I/O: {total_io_s * 1000.0:.3f} ms ({100.0 * total_io_s / total_layer_s:.2f}%)")
    print(f"dequantization: {total_dequant_s * 1000.0:.3f} ms ({100.0 * total_dequant_s / total_layer_s:.2f}%)")
    print(f"compute/other: {total_compute_s * 1000.0:.3f} ms ({100.0 * total_compute_s / total_layer_s:.2f}%)")
    print(f"layer total: {total_layer_s * 1000.0:.3f} ms")
    print(f"wall time: {wall_s * 1000.0:.3f} ms")
    print(f"load_tensor calls: {PROFILE.io_calls}")
    print(f"materialized bytes: {mib(PROFILE.io_bytes):.2f} MiB")
    print(f"dequant calls: {PROFILE.dequant_calls}")
    print(f"dequant output: {mib(PROFILE.dequant_bytes):.2f} MiB")
    print(f"attention-type detection: {total_type_s * 1000.0:.3f} ms")
    print(f"final output norm: {torch.linalg.vector_norm(x).item():.8f}")

    del x
    gc.collect()


if __name__ == "__main__":
    main()
