from __future__ import annotations

"""Execute one Qwen3.6 FP8 routed expert.

This is an isolated correctness/benchmark harness, not the full model.
The official FP8 checkpoint stores 2-D FP8 E4M3 weights with per-block
128x128 inverse scales. We load one expert through FP8ExpertCache, dequantize
its three projections to a normal floating-point CUDA tensor, then execute:

    gate = x @ gate_proj.T
    up   = x @ up_proj.T
    h    = silu(gate) * up
    y    = h @ down_proj.T

The runner intentionally keeps the router and shared expert out of this test.
"""

import argparse
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

try:
    from .fp8_expert_cache import FP8ExpertCache
except ImportError:  # direct ``python fp8_expert_runner.py`` execution
    from fp8_expert_cache import FP8ExpertCache


BLOCK = 128
INPUT_SIZE = 2048
HIDDEN_SIZE = 512


def _dequantize_blockwise(weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """Dequantize a block-wise FP8 E4M3 weight to float32."""
    if weight.ndim != 2 or scale_inv.ndim != 2:
        raise ValueError(
            f"Expected 2-D weight/scale tensors, got {tuple(weight.shape)} and {tuple(scale_inv.shape)}"
        )

    out_features, in_features = map(int, weight.shape)
    expected_scales = (
        (out_features + BLOCK - 1) // BLOCK,
        (in_features + BLOCK - 1) // BLOCK,
    )
    if tuple(scale_inv.shape) != expected_scales:
        raise ValueError(
            f"Scale shape {tuple(scale_inv.shape)} does not match weight "
            f"shape {tuple(weight.shape)}; expected {expected_scales}"
        )

    values = weight.float()
    expanded = scale_inv.float().repeat_interleave(BLOCK, dim=0).repeat_interleave(BLOCK, dim=1)
    return values * expanded[:out_features, :in_features]


def _make_input(seed: int, device: torch.device) -> torch.Tensor:
    """Create a deterministic input on CPU, then transfer it to CUDA.

    Using a CPU generator avoids the ``Expected a 'cuda' device type for
    generator but found 'cpu'`` error on builds where a CPU Generator cannot
    be passed to a CUDA random operation.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    cpu_x = torch.randn(INPUT_SIZE, generator=generator, dtype=torch.float32)
    return cpu_x.to(device, non_blocking=False)


def run_one(
    root: Path,
    layer: int,
    expert: int,
    *,
    ram_gb: float,
    vram_gb: float,
    seed: int,
) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this Python environment.")

    device = torch.device("cuda")
    cache = FP8ExpertCache(
        root,
        ram_limit_bytes=int(ram_gb * 1024**3),
        vram_limit_bytes=int(vram_gb * 1024**3),
        device="cuda",
    )

    expert_blob = cache.get(layer, expert, tier="vram")

    start = perf_counter()
    gate = _dequantize_blockwise(expert_blob.weights["gate_proj"], expert_blob.scales["gate_proj"])
    up = _dequantize_blockwise(expert_blob.weights["up_proj"], expert_blob.scales["up_proj"])
    down = _dequantize_blockwise(expert_blob.weights["down_proj"], expert_blob.scales["down_proj"])
    dequant_ms = (perf_counter() - start) * 1000.0

    x = _make_input(seed, device)

    if gate.shape != (HIDDEN_SIZE, INPUT_SIZE):
        raise ValueError(f"Unexpected gate shape: {tuple(gate.shape)}")
    if up.shape != (HIDDEN_SIZE, INPUT_SIZE):
        raise ValueError(f"Unexpected up shape: {tuple(up.shape)}")
    if down.shape != (INPUT_SIZE, HIDDEN_SIZE):
        raise ValueError(f"Unexpected down shape: {tuple(down.shape)}")

    gate = gate.to(device)
    up = up.to(device)
    down = down.to(device)

    torch.cuda.synchronize()
    start = perf_counter()

    gate_out = F.linear(x, gate)
    up_out = F.linear(x, up)
    hidden = F.silu(gate_out) * up_out
    output = F.linear(hidden, down)

    torch.cuda.synchronize()
    compute_ms = (perf_counter() - start) * 1000.0

    print(f"Expert: ({layer}, {expert})")
    print(f"Quantized cache bytes: {expert_blob.size_bytes:,}")
    print(f"Dequantization: {dequant_ms:.3f} ms")
    print(f"Compute: {compute_ms:.3f} ms")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Output norm: {torch.linalg.vector_norm(output).item():.6f}")
    print(f"Output mean: {output.mean().item():.6f}")
    print(f"Output std: {output.std().item():.6f}")
    print(f"CUDA allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MiB")
    print(f"CUDA reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MiB")
    print(cache.snapshot())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Qwen3.6 FP8 expert")
    parser.add_argument("root", type=Path, help="Directory containing Safetensors shards")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--ram-gb", type=float, default=6.0)
    parser.add_argument("--vram-gb", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    run_one(
        args.root,
        args.layer,
        args.expert,
        ram_gb=args.ram_gb,
        vram_gb=args.vram_gb,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
