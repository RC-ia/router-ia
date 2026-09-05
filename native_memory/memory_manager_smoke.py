from __future__ import annotations

"""Smoke test for logical block residency management.

Run from repository root:
    python native_memory/memory_manager_smoke.py
"""

import ctypes
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DLL = ROOT / "native_memory" / "build-ninja" / "router_ia_native_memory.dll"
os.environ.setdefault("ROUTER_IA_NATIVE_MEMORY_LIB", str(DLL))

from router_ia.memory_manager import MemoryManager


SLOT_BYTES = 8 * 1024 * 1024
PAYLOAD_BYTES = 4 * 1024 * 1024
VRAM_SLOTS = 4
RAM_SLOTS = 12
BLOCKS = 12


def payload(block_id: int) -> bytes:
    return bytes(((i * 29 + block_id * 47 + 13) & 0xFF) for i in range(PAYLOAD_BYTES))


def main() -> None:
    print("=" * 68)
    print("NATIVE LOGICAL MEMORY MANAGER SMOKE TEST")
    print("=" * 68)
    print(f"VRAM slots: {VRAM_SLOTS}")
    print(f"RAM slots : {RAM_SLOTS}")
    print(f"Blocks    : {BLOCKS}")
    print(f"Payload   : {PAYLOAD_BYTES / 1024 / 1024:.1f} MiB")

    buffers: list[ctypes.Array[ctypes.c_char]] = []

    with MemoryManager(
        vram_slot_bytes=SLOT_BYTES,
        vram_slots=VRAM_SLOTS,
        ram_slot_bytes=SLOT_BYTES,
        ram_slots=RAM_SLOTS,
        streams=2,
    ) as mm:
        print("\n[1] Registrando blocos lógicos...")
        for block_id in range(BLOCKS):
            buf = ctypes.create_string_buffer(payload(block_id))
            buffers.append(buf)
            mm.register_block(block_id, buf, PAYLOAD_BYTES)
            assert mm.is_registered(block_id)
        print("  PASS")

        print("\n[2] Carregando working set inicial...")
        locations = {}
        for block_id in range(VRAM_SLOTS):
            slot = mm.acquire(block_id, PAYLOAD_BYTES)
            locations[block_id] = slot
            assert mm.is_resident(block_id)
        print("  PASS")
        print("  residentes:", locations)

        print("\n[3] Testando cache hit...")
        slot_before = locations[0]
        slot_after = mm.acquire(0, PAYLOAD_BYTES)
        assert slot_before == slot_after
        print(f"  block 0: VRAM slot {slot_after} | PASS")

        print("\n[4] Testando pin/unpin...")
        mm.pin(0)
        mm.pin(1)
        assert mm.is_resident(0)
        assert mm.is_resident(1)
        mm.unpin(0)
        mm.unpin(1)
        print("  PASS")

        print("\n[5] Forçando misses e eviction LRU...")
        for block_id in range(VRAM_SLOTS, BLOCKS):
            slot = mm.acquire(block_id, PAYLOAD_BYTES)
            locations[block_id] = slot
            assert mm.is_resident(block_id)
        print("  PASS")

        print("\n[6] Verificando que alguns blocos foram expulsos...")
        resident_count = sum(mm.is_resident(i) for i in range(BLOCKS))
        print(f"  residentes atuais: {resident_count}/{BLOCKS}")
        assert resident_count == VRAM_SLOTS
        print("  PASS")

        print("\n[7] Testando retorno de bloco antigo...")
        slot = mm.acquire(0, PAYLOAD_BYTES)
        assert mm.is_resident(0)
        print(f"  block 0 recarregado em VRAM slot {slot}: PASS")

        print("\n[8] Estatísticas:")
        stats = mm.stats()
        print(stats)
        assert stats["cache_hits"] >= 1
        assert stats["cache_misses"] >= BLOCKS - VRAM_SLOTS
        assert stats["evictions"] >= 1
        assert stats["bytes_h2d"] > 0
        assert stats["bytes_d2h"] > 0
        print("  PASS")

        print("\n[9] Unregister de bloco não residente/residente...")
        # Ensure block 2 can be safely removed after it is made non-resident.
        mm.evict(2)
        mm.unregister_block(2)
        assert not mm.is_registered(2)
        print("  PASS")

    print("\n" + "=" * 68)
    print("MEMORY MANAGER = PASS")
    print("=" * 68)


if __name__ == "__main__":
    main()
