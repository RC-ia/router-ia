from __future__ import annotations

"""Stress test for repeated RAM/VRAM slot swapping and reuse.

Run from the repository root:
    python native_memory/slot_swap_stress.py

The test simulates model-block residency changes:
- fills RAM slots with distinct payloads;
- loads them into VRAM slots;
- verifies all resident payloads;
- evicts a subset back to RAM;
- overwrites/reuses VRAM slots with new payloads;
- reloads evicted payloads;
- verifies every slot after each phase.
"""

import ctypes
import os
import sys
from pathlib import Path

# Make src/router_ia importable when run directly from the repository.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# The current Ninja build places the DLL here. An environment override still wins.
DLL = ROOT / "native_memory" / "build-ninja" / "router_ia_native_memory.dll"
os.environ.setdefault("ROUTER_IA_NATIVE_MEMORY_LIB", str(DLL))

from router_ia.qwen36_native_memory import NativeMemory


SLOT_BYTES = 8 * 1024 * 1024
VRAM_SLOTS = 8
RAM_SLOTS = 8
PAYLOAD_BYTES = 4 * 1024 * 1024
ROUNDS = 8


def make_payload(size: int, seed: int) -> bytes:
    # Distinct deterministic payload per logical block/version.
    return bytes(((i * 131 + seed * 17 + 29) & 0xFF) for i in range(size))


def write_ram(mem: NativeMemory, slot: int, payload: bytes) -> None:
    # ctypes buffer gives us a stable host pointer for stage_host().
    buf = ctypes.create_string_buffer(payload)
    mem._test_buffers.append(buf)
    mem.stage_host(slot, ctypes.addressof(buf), len(payload))


def read_ram(mem: NativeMemory, slot: int, size: int) -> bytes:
    return ctypes.string_at(mem.ram_ptr(slot), size)


def verify_ram(mem: NativeMemory, slot: int, expected: bytes) -> None:
    actual = read_ram(mem, slot, len(expected))
    if actual != expected:
        raise AssertionError(f"RAM slot {slot} data mismatch")


def main() -> None:
    if not torch_cuda_available():
        raise RuntimeError("CUDA não está disponível.")

    print("=" * 60)
    print("NATIVE MEMORY SLOT-SWAP STRESS TEST")
    print("=" * 60)
    print(f"GPU: {get_gpu_name()}")
    print(f"VRAM slots: {VRAM_SLOTS}")
    print(f"RAM slots : {RAM_SLOTS}")
    print(f"Slot size : {SLOT_BYTES / 1024 / 1024:.1f} MiB")
    print(f"Payload   : {PAYLOAD_BYTES / 1024 / 1024:.1f} MiB")
    print(f"Rounds    : {ROUNDS}")

    with NativeMemory(
        vram_slot_bytes=SLOT_BYTES,
        vram_slots=VRAM_SLOTS,
        ram_slot_bytes=SLOT_BYTES,
        ram_slots=RAM_SLOTS,
    ) as mem:
        # Keep ctypes source buffers alive for the duration of the test.
        mem._test_buffers = []

        print("\n[1] Verificando slots...")
        vram_ptrs = [mem.vram_ptr(i) for i in range(VRAM_SLOTS)]
        ram_ptrs = [mem.ram_ptr(i) for i in range(RAM_SLOTS)]
        assert len(set(vram_ptrs)) == VRAM_SLOTS
        assert len(set(ram_ptrs)) == RAM_SLOTS
        print("  PASS")

        print("\n[2] Fase inicial: 8 blocos distintos RAM -> VRAM...")
        payloads = {}
        for i in range(VRAM_SLOTS):
            payload = make_payload(PAYLOAD_BYTES, seed=i + 1)
            payloads[i] = payload
            write_ram(mem, i, payload)
            mem.h2d_async(i, i, PAYLOAD_BYTES)
        mem.sync()
        print("  PASS")

        print("\n[3] Verificando residência inicial...")
        for i in range(VRAM_SLOTS):
            mem.d2h_async(i, i, PAYLOAD_BYTES)
        mem.sync()
        for i in range(VRAM_SLOTS):
            verify_ram(mem, i, payloads[i])
            print(f"  slot {i}: PASS")

        print("\n[4] Evict: VRAM -> RAM e reutilização parcial...")
        # Evict even slots and reuse those VRAM slots for new logical blocks.
        for i in range(0, VRAM_SLOTS, 2):
            mem.d2h_async(i, i, PAYLOAD_BYTES)
        mem.sync()

        for i in range(0, VRAM_SLOTS, 2):
            verify_ram(mem, i, payloads[i])

        for i in range(0, VRAM_SLOTS, 2):
            new_payload = make_payload(PAYLOAD_BYTES, seed=100 + i)
            payloads[i + VRAM_SLOTS] = new_payload
            write_ram(mem, i, new_payload)
            mem.h2d_async(i, i, PAYLOAD_BYTES)
        mem.sync()
        print("  PASS")

        print("\n[5] Recarregando blocos antigos nos slots RAM/VRAM...")
        # Save currently resident replacement blocks back to RAM first.
        for i in range(0, VRAM_SLOTS, 2):
            mem.d2h_async(i, i, PAYLOAD_BYTES)
        mem.sync()
        for i in range(0, VRAM_SLOTS, 2):
            verify_ram(mem, i, payloads[i + VRAM_SLOTS])

        # Rebuild the original evicted payloads in the same RAM slots and load them again.
        for i in range(0, VRAM_SLOTS, 2):
            write_ram(mem, i, payloads[i])
            mem.h2d_async(i, i, PAYLOAD_BYTES)
        mem.sync()
        print("  PASS")

        print("\n[6] Rodando trocas repetidas...")
        for round_id in range(ROUNDS):
            for i in range(VRAM_SLOTS):
                logical_id = round_id * VRAM_SLOTS + i
                payload = make_payload(PAYLOAD_BYTES, seed=1000 + logical_id)
                payloads[logical_id] = payload
                write_ram(mem, i, payload)
                mem.h2d_async(i, i, PAYLOAD_BYTES)
            mem.sync()

            # Every round, evict all slots and verify the complete set.
            for i in range(VRAM_SLOTS):
                mem.d2h_async(i, i, PAYLOAD_BYTES)
            mem.sync()

            for i in range(VRAM_SLOTS):
                logical_id = round_id * VRAM_SLOTS + i
                verify_ram(mem, i, payloads[logical_id])

            if round_id == 0 or round_id == ROUNDS - 1 or round_id % 2 == 1:
                print(f"  round {round_id + 1}/{ROUNDS}: PASS")

        print("\n[7] Estatísticas finais:")
        stats = mem.stats()
        print(stats)

        expected_h2d = (VRAM_SLOTS * (1 + ROUNDS + 1))
        expected_d2h = (VRAM_SLOTS * (2 + ROUNDS))
        expected_bytes = PAYLOAD_BYTES

        print("\n[8] Verificação de contadores...")
        print(f"  H2D calls: {stats['h2d_calls']} (>= {expected_h2d})")
        print(f"  D2H calls: {stats['d2h_calls']} (>= {expected_d2h})")
        print(f"  H2D MiB  : {stats['bytes_h2d'] / 1024 / 1024:.1f}")
        print(f"  D2H MiB  : {stats['bytes_d2h'] / 1024 / 1024:.1f}")

        if stats["bytes_h2d"] < expected_h2d * expected_bytes:
            raise AssertionError("H2D byte counter abaixo do esperado")
        if stats["bytes_d2h"] < expected_d2h * expected_bytes:
            raise AssertionError("D2H byte counter abaixo do esperado")

        print("  PASS")

    print("\n" + "=" * 60)
    print("SLOT-SWAP STRESS TEST = PASS")
    print("=" * 60)


def torch_cuda_available() -> bool:
    import torch
    return bool(torch.cuda.is_available())


def get_gpu_name() -> str:
    import torch
    return torch.cuda.get_device_name(0)


if __name__ == "__main__":
    main()
