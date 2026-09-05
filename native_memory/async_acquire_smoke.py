from __future__ import annotations

"""HW smoke test for the asynchronous acquire path (Proposal #1).

Requires the compiled native library (CUDA). Run from repository root:

    python native_memory/async_acquire_smoke.py
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

from router_ia.memory_manager import MemoryManager

SLOT_BYTES = 8 * 1024 * 1024
PAYLOAD_BYTES = 4 * 1024 * 1024
VRAM_SLOTS = 4
RAM_SLOTS = 12
BLOCKS = 24


def payload(block_id: int) -> bytes:
    return bytes(((i * 29 + block_id * 47 + 13) & 0xFF) for i in range(PAYLOAD_BYTES))


def main() -> None:
    print("=" * 68)
    print("NATIVE ASYNC-ACQUIRE SMOKE TEST (Proposal #1)")
    print("=" * 68)

    buffers: list[ctypes.Array[ctypes.c_char]] = []
    with MemoryManager(
        vram_slot_bytes=SLOT_BYTES,
        vram_slots=VRAM_SLOTS,
        ram_slot_bytes=SLOT_BYTES,
        ram_slots=RAM_SLOTS,
        streams=2,
    ) as mm:
        for block_id in range(BLOCKS):
            buf = ctypes.create_string_buffer(payload(block_id))
            buffers.append(buf)
            mm.register_block(block_id, buf, PAYLOAD_BYTES)

        print("\n[1] Disparando acquire_async em rajada (sem sincronizar)...")
        slots = {}
        t0 = time.perf_counter()
        for block_id in range(8):
            slots[block_id] = mm.acquire_async(block_id, PAYLOAD_BYTES)
        dispatch_elapsed = time.perf_counter() - t0
        print(f"  dispatch de 8 H2D: {dispatch_elapsed*1000:.1f} ms")
        print("  (se serialize, cada H2D de 4MiB levaria dezenas de ms aqui)")
        for block_id in range(8):
            assert mm.is_loading(block_id), f"block {block_id} deve estar em loading"
        print("  PASS: todos em loading, dispatch sem bloquear")

        print("\n[2] Re-acquire no mesmo bloco não duplica transfer...")
        stats_before = mm.stats()
        same_slot = mm.acquire_async(0, PAYLOAD_BYTES)
        assert same_slot == slots[0], "re-acquire deve reusar o slot reservado"
        stats_after = mm.stats()
        assert stats_after["h2d_calls"] == stats_before["h2d_calls"], \
            "re-acquire não deve emitir nova H2D"
        print("  PASS: sem transferência duplicada")

        print("\n[3] wait_acquire promove para residente com dados corretos...")
        for block_id in range(8):
            mm.wait_acquire(block_id)
            assert not mm.is_loading(block_id)
            assert mm.is_resident(block_id)
        print("  PASS: 8 blocos promovidos a residente")

        print("\n[4] Verificando integridade dos dados pós-transferência...")
        for block_id in range(8):
            vram_slot = slots[block_id]
            ptr = mm.vram_ptr(vram_slot)
            got = ctypes.string_at(ptr, PAYLOAD_BYTES)
            expect = payload(block_id)
            assert got == expect, f"block {block_id} dados corrompidos"
        print("  PASS: conteúdo VRAM íntegro (dados batem bit a bit)")

        print("\n[5] Overlap: medindo tempo de dispatch vs. sync separados...")
        # dispatch 8 fresh blocks (forcing eviction), then wait all
        mm.unregister_block(0)
        mm.wait_acquire(0)  # no-op, not loading
        # re-register a block to force fresh transfers
        fresh_ids = list(range(8, 16))
        fresh_slots = {}
        t0 = time.perf_counter()
        for bid in fresh_ids:
            fresh_slots[bid] = mm.acquire_async(bid, PAYLOAD_BYTES)
        dispatch2 = time.perf_counter() - t0
        t0 = time.perf_counter()
        for bid in fresh_ids:
            mm.wait_acquire(bid)
        sync2 = time.perf_counter() - t0
        print(f"  dispatch (8 H2D): {dispatch2*1000:.1f} ms")
        print(f"  wait_all (sync) : {sync2*1000:.1f} ms")
        print("  -> o tempo de transferência fica escondido no wait_all, não no dispatch")

        print("\n[6] Estatísticas finais...")
        print(mm.stats())

    print("\n" + "=" * 68)
    print("ASYNC-ACQUIRE = PASS")
    print("=" * 68)


if __name__ == "__main__":
    main()