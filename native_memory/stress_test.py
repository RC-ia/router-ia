# native_memory/stress_test.py

from __future__ import annotations

import ctypes
import os
import sys
import torch

# Aponte para a DLL gerada pelo CMake/Ninja.
# Ajuste somente se o caminho for diferente.
os.environ["ROUTER_IA_NATIVE_MEMORY_LIB"] = (
    r"D:\router\router-ia\native_memory\build-ninja\router_ia_native_memory.dll"
)

from router_ia.qwen36_native_memory import NativeMemory


SLOT_BYTES = 8 * 1024 * 1024
VRAM_SLOTS = 8
RAM_SLOTS = 8

# 4 MiB por teste para manter margem.
TEST_BYTES = 4 * 1024 * 1024


def make_pattern(size: int, seed: int) -> torch.Tensor:
    # Padrão determinístico diferente para cada slot.
    x = torch.arange(size, dtype=torch.uint8)
    return (x + seed) & 0xFF


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não está disponível.")

    print("=" * 60)
    print("NATIVE MEMORY STRESS TEST")
    print("=" * 60)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM slots: {VRAM_SLOTS}")
    print(f"RAM slots : {RAM_SLOTS}")
    print(f"Slot size : {SLOT_BYTES / 1024 / 1024:.1f} MiB")
    print(f"Teste     : {TEST_BYTES / 1024 / 1024:.1f} MiB por slot")

    with NativeMemory(
        vram_slot_bytes=SLOT_BYTES,
        vram_slots=VRAM_SLOTS,
        ram_slot_bytes=SLOT_BYTES,
        ram_slots=RAM_SLOTS,
    ) as mem:

        print("\n[1] Verificando ponteiros...")

        vram_ptrs = [mem.vram_ptr(i) for i in range(VRAM_SLOTS)]
        ram_ptrs = [mem.ram_ptr(i) for i in range(RAM_SLOTS)]

        for i, ptr in enumerate(vram_ptrs):
            print(f"  VRAM[{i}] = 0x{ptr:x}")

        for i, ptr in enumerate(ram_ptrs):
            print(f"  RAM [{i}] = 0x{ptr:x}")

        if len(set(vram_ptrs)) != VRAM_SLOTS:
            raise AssertionError("VRAM slots não possuem ponteiros distintos.")

        if len(set(ram_ptrs)) != RAM_SLOTS:
            raise AssertionError("RAM slots não possuem ponteiros distintos.")

        print("  PASS")

        print("\n[2] Preenchendo RAM com padrões diferentes...")

        fontes = []

        for i in range(VRAM_SLOTS):
            src = make_pattern(TEST_BYTES, seed=i * 17 + 3)
            fontes.append(src)

            mem.stage_host(
                i,
                src,
                TEST_BYTES,
            )

        print("  PASS")

        print("\n[3] Enviando TODOS os slots RAM -> VRAM...")

        for i in range(VRAM_SLOTS):
            mem.h2d_async(
                i,
                i,
                TEST_BYTES,
            )

        mem.sync()

        print("  PASS")

        print("\n[4] Limpando RAM destino...")

        # Copiamos a saída de cada VRAM para outro slot RAM.
        # Assim não destruímos os slots fonte.
        for i in range(VRAM_SLOTS):
            mem.d2h_async(
                i,
                i,
                TEST_BYTES,
            )

        mem.sync()

        print("  PASS")

        print("\n[5] Validando conteúdo de TODOS os slots...")

        for i in range(VRAM_SLOTS):
            ptr = mem.ram_ptr(i)

            raw = ctypes.string_at(ptr, TEST_BYTES)
            expected = fontes[i].numpy().tobytes()

            if raw != expected:
                raise AssertionError(
                    f"Mismatch no slot {i}: "
                    f"conteúdo RAM != conteúdo original"
                )

            print(f"  slot {i}: PASS")

        print("\n[6] Testando sobrescrita/reutilização...")

        # Reutiliza os mesmos slots com outros padrões.
        fontes2 = []

        for i in range(VRAM_SLOTS):
            src = make_pattern(
                TEST_BYTES,
                seed=200 + i * 31,
            )
            fontes2.append(src)

            mem.stage_host(
                i,
                src,
                TEST_BYTES,
            )

            mem.h2d_async(
                i,
                i,
                TEST_BYTES,
            )

        mem.sync()

        for i in range(VRAM_SLOTS):
            mem.d2h_async(
                i,
                i,
                TEST_BYTES,
            )

        mem.sync()

        for i in range(VRAM_SLOTS):
            raw = ctypes.string_at(
                mem.ram_ptr(i),
                TEST_BYTES,
            )

            expected = fontes2[i].numpy().tobytes()

            if raw != expected:
                raise AssertionError(
                    f"Mismatch após reutilização no slot {i}"
                )

            print(f"  slot {i}: PASS")

        print("\n[7] Estatísticas...")

        print(mem.stats())

        print("\n" + "=" * 60)
        print("STRESS TEST = PASS")
        print("=" * 60)


if __name__ == "__main__":
    main()
