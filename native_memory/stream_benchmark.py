from __future__ import annotations

"""Benchmark single/multi CUDA stream transfer throughput.

Run from the repository root:
    python native_memory/stream_benchmark.py

Compares 1/2/4/8 streams for:
- H2D-only transfers
- D2H-only transfers
- mixed H2D+D2H traffic

The native backend assigns each VRAM slot to a stream by slot % stream_count.
All timings include the final native synchronization, so the measured interval
covers actual device work rather than only API enqueue time.
"""

import ctypes
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DLL = ROOT / "native_memory" / "build-ninja" / "router_ia_native_memory.dll"
os.environ.setdefault("ROUTER_IA_NATIVE_MEMORY_LIB", str(DLL))

from router_ia.qwen36_native_memory import NativeMemory


SLOT_BYTES = 8 * 1024 * 1024
VRAM_SLOTS = 8
RAM_SLOTS = 8
PAYLOAD_BYTES = 4 * 1024 * 1024

STREAM_VARIANTS = (1, 2, 4, 8)
WARMUP_ROUNDS = 3
MEASURE_ROUNDS = 12


def make_payload(size: int, seed: int) -> bytes:
    return bytes(((i * 131 + seed * 17 + 29) & 0xFF) for i in range(size))


def stage_all(mem: NativeMemory, buffers: list[ctypes.Array[ctypes.c_char]]) -> None:
    for slot in range(VRAM_SLOTS):
        payload = make_payload(PAYLOAD_BYTES, slot + 7)
        buf = ctypes.create_string_buffer(payload)
        buffers.append(buf)
        mem.stage_host(slot, ctypes.addressof(buf), PAYLOAD_BYTES)


def throughput_gbps(total_bytes: int, seconds: float) -> float:
    if seconds <= 0:
        return float("inf")
    return total_bytes / seconds / 1e9


def run_case(streams: int, mode: str) -> dict[str, float | int | str]:
    buffers: list[ctypes.Array[ctypes.c_char]] = []

    with NativeMemory(
        vram_slot_bytes=SLOT_BYTES,
        vram_slots=VRAM_SLOTS,
        ram_slot_bytes=SLOT_BYTES,
        ram_slots=RAM_SLOTS,
        streams=streams,
    ) as mem:
        stage_all(mem, buffers)

        # Prime the GPU and transfer paths.
        for _ in range(WARMUP_ROUNDS):
            if mode in ("h2d", "mixed"):
                for slot in range(VRAM_SLOTS):
                    mem.h2d_async(slot, slot, PAYLOAD_BYTES)
            if mode in ("d2h", "mixed"):
                for slot in range(VRAM_SLOTS):
                    mem.d2h_async(slot, slot, PAYLOAD_BYTES)
            mem.sync()

        start = time.perf_counter()

        for _ in range(MEASURE_ROUNDS):
            if mode == "h2d":
                for slot in range(VRAM_SLOTS):
                    mem.h2d_async(slot, slot, PAYLOAD_BYTES)

            elif mode == "d2h":
                for slot in range(VRAM_SLOTS):
                    mem.d2h_async(slot, slot, PAYLOAD_BYTES)

            elif mode == "mixed":
                # Submit both directions before one global sync. Different
                # slot-indexed streams can overlap where the GPU permits it.
                for slot in range(VRAM_SLOTS):
                    if slot % 2 == 0:
                        mem.h2d_async(slot, slot, PAYLOAD_BYTES)
                    else:
                        mem.d2h_async(slot, slot, PAYLOAD_BYTES)

            else:
                raise ValueError(f"unknown mode: {mode}")

        mem.sync()
        elapsed = time.perf_counter() - start
        stats = mem.stats()

    if mode == "mixed":
        total_bytes = MEASURE_ROUNDS * VRAM_SLOTS * PAYLOAD_BYTES
    else:
        total_bytes = MEASURE_ROUNDS * VRAM_SLOTS * PAYLOAD_BYTES

    return {
        "streams": streams,
        "mode": mode,
        "seconds": elapsed,
        "gbps": throughput_gbps(total_bytes, elapsed),
        "h2d_bytes": stats["bytes_h2d"],
        "d2h_bytes": stats["bytes_d2h"],
    }


def print_result(result: dict[str, float | int | str]) -> None:
    print(
        f"  streams={result['streams']} | "
        f"{result['mode']:>5} | "
        f"{result['seconds']:.4f} s | "
        f"{result['gbps']:.3f} GB/s"
    )


def main() -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não está disponível.")

    print("=" * 72)
    print("NATIVE MEMORY MULTI-STREAM BENCHMARK")
    print("=" * 72)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM slots: {VRAM_SLOTS}")
    print(f"RAM slots : {RAM_SLOTS}")
    print(f"Payload   : {PAYLOAD_BYTES / 1024 / 1024:.1f} MiB")
    print(f"Warmup    : {WARMUP_ROUNDS} rounds")
    print(f"Measure   : {MEASURE_ROUNDS} rounds")
    print(f"Variants  : {', '.join(map(str, STREAM_VARIANTS))} streams")

    all_results: dict[str, list[dict[str, float | int | str]]] = {
        "h2d": [],
        "d2h": [],
        "mixed": [],
    }

    for mode in ("h2d", "d2h", "mixed"):
        print("\n" + "-" * 72)
        print(f"{mode.upper()} — resultados")
        print("-" * 72)

        for streams in STREAM_VARIANTS:
            result = run_case(streams, mode)
            all_results[mode].append(result)
            print_result(result)

    print("\n" + "=" * 72)
    print("RESUMO")
    print("=" * 72)

    for mode, results in all_results.items():
        best = max(results, key=lambda r: float(r["gbps"]))
        base = results[0]
        speedup = float(best["gbps"]) / float(base["gbps"])
        print(
            f"{mode:>5}: melhor={best['streams']} streams | "
            f"{float(best['gbps']):.3f} GB/s | "
            f"speedup vs 1 stream={speedup:.2f}x"
        )

    print("\n[integridade] Estatísticas registradas durante o benchmark.")
    print("[integridade] Para corrupção de dados, rode também slot_swap_stress.py.")
    print("\nBENCHMARK = PASS")


if __name__ == "__main__":
    main()
