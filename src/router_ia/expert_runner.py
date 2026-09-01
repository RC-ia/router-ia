from __future__ import annotations

"""Run one packed Qwen3.6 expert as an isolated experiment.

The cache keeps quantized GGUF bytes. This runner is the first bridge to
actual math:

    GGUF -> ExpertCache -> ggml dequantization -> FP32 matrices -> CUDA GEMM

It intentionally runs one expert only. It is not the full model executor.
Dequantization is delegated to exported ggml row-dequantization functions.
"""

import argparse
import ctypes
import os
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .expert_cache import ExpertCache


# Current GGML enum values: IQ3_XXS=18 and IQ4_XS=23.
TYPE_IQ3_XXS = 18
TYPE_IQ4_XS = 23
EXPERT_INPUT = 2048


class GGMLDequantizer:
    """Small ctypes wrapper around ggml's exported dequantizers."""

    def __init__(self, dll_path: Path | None = None) -> None:
        self.dll_path = dll_path or self._find_dll()
        if self.dll_path is None:
            raise FileNotFoundError(
                "ggml.dll was not found. Pass --ggml-dll PATH."
            )
        self.dll_path = self.dll_path.resolve()

        if os.name == "nt":
            try:
                os.add_dll_directory(str(self.dll_path.parent))
            except (AttributeError, FileNotFoundError, OSError):
                pass

        self.lib = ctypes.CDLL(str(self.dll_path))
        self.functions: dict[int, Callable[..., None]] = {}
        self._bind(TYPE_IQ3_XXS, "dequantize_row_iq3_xxs")
        self._bind(TYPE_IQ4_XS, "dequantize_row_iq4_xs")

    @staticmethod
    def _find_dll() -> Path | None:
        candidates: list[Path] = []
        for root in (Path.cwd(), Path(__file__).resolve().parent):
            candidates.extend(
                [root / "ggml.dll", root / "ggml-base.dll", root / "llama.dll"]
            )
        return next((p for p in candidates if p.is_file()), None)

    def _bind(self, type_id: int, symbol: str) -> None:
        try:
            function = getattr(self.lib, symbol)
        except AttributeError as exc:
            raise RuntimeError(
                f"{symbol} is not exported by {self.dll_path}."
            ) from exc

        function.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
        ]
        function.restype = None
        self.functions[type_id] = function

    def dequantize(self, raw: bytes, type_id: int, elements: int) -> np.ndarray:
        try:
            function = self.functions[type_id]
        except KeyError as exc:
            raise ValueError(f"Unsupported GGML type id: {type_id}") from exc

        output = np.empty(elements, dtype=np.float32)
        source = ctypes.create_string_buffer(raw)
        function(
            ctypes.cast(source, ctypes.c_void_p),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int64(elements),
        )
        return output


def dequantized_matrix(
    dequantizer: GGMLDequantizer,
    raw: bytes,
    dtype: str,
    packed_shape: tuple[int, ...],
) -> torch.Tensor:
    """Dequantize one expert slice to its logical 2D matrix."""
    if len(packed_shape) != 3 or packed_shape[2] != 256:
        raise ValueError(f"Unexpected expert tensor shape: {packed_shape}")

    shape = (packed_shape[0], packed_shape[1])
    elements = shape[0] * shape[1]
    values = dequantizer.dequantize(raw, int(dtype), elements)
    return torch.from_numpy(values.reshape(shape))


def run_one_expert(
    model: Path,
    layer: int,
    expert: int,
    *,
    device: str,
    ggml_dll: Path | None,
    ram_gb: float,
    vram_gb: float,
    seed: int,
) -> None:
    cache = ExpertCache(
        model,
        ram_limit_bytes=int(ram_gb * 1024**3),
        vram_limit_bytes=int(vram_gb * 1024**3),
        device=device,
    )

    parts = cache.index.get(layer, expert)
    cpu_blob = cache.get_cpu(layer, expert)

    # Exercise RAM -> VRAM residency before doing any math.
    if device.startswith("cuda"):
        cache.get_vram(layer, expert)

    dequantizer = GGMLDequantizer(ggml_dll)

    start = perf_counter()
    gate = dequantized_matrix(
        dequantizer, cpu_blob.slices["gate"], parts["gate"].dtype, parts["gate"].shape
    )
    up = dequantized_matrix(
        dequantizer, cpu_blob.slices["up"], parts["up"].dtype, parts["up"].shape
    )
    down = dequantized_matrix(
        dequantizer, cpu_blob.slices["down"], parts["down"].dtype, parts["down"].shape
    )
    dequant_ms = (perf_counter() - start) * 1000.0

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.randn(EXPERT_INPUT, generator=generator, dtype=torch.float32)

    gate = gate.to(device)
    up = up.to(device)
    down = down.to(device)
    x = x.to(device)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = perf_counter()

    # Logical expert projections for shapes [2048, 512], [2048, 512], [512, 2048].
    gate_out = x @ gate
    up_out = x @ up
    hidden = F.silu(gate_out) * up_out
    output = hidden @ down

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    compute_ms = (perf_counter() - start) * 1000.0

    print(f"GGML DLL: {dequantizer.dll_path}")
    print(f"Expert: ({layer}, {expert})")
    print(f"Quantized bytes: {cpu_blob.size}")
    print(f"Dequantization: {dequant_ms:.2f} ms")
    print(f"Compute: {compute_ms:.2f} ms")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Output norm: {torch.linalg.vector_norm(output).item():.6f}")
    print(f"Output mean: {output.mean().item():.6f}")
    print(f"Output std: {output.std().item():.6f}")
    if device.startswith("cuda"):
        print(f"CUDA allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MiB")
    print(f"Cache: {cache.snapshot()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Qwen3.6 GGUF expert")
    parser.add_argument("model", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--ggml-dll", type=Path, default=None)
    parser.add_argument("--ram-gb", type=float, default=6.0)
    parser.add_argument("--vram-gb", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this Python environment.")

    run_one_expert(
        args.model,
        args.layer,
        args.expert,
        device=args.device,
        ggml_dll=args.ggml_dll,
        ram_gb=args.ram_gb,
        vram_gb=args.vram_gb,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
