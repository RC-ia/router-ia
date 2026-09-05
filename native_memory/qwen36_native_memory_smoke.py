from __future__ import annotations

"""Smoke test for the native fixed-slot CUDA memory backend.

Usage from the repository root after building:
    python native_memory/qwen36_native_memory_smoke.py
"""

import ctypes

try:
    import torch
except ImportError as exc:
    raise SystemExit("PyTorch is required for the smoke test") from exc

from router_ia.qwen36_native_memory import NativeMemory


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; the native memory smoke test needs a CUDA device")

    # Prototype sizing is intentionally modest so it can run on the target
    # 4 GiB GPU. One slot is large enough for a representative packed Q4 blob.
    slot_bytes = 8 * 1024 * 1024
    with NativeMemory(
        vram_slot_bytes=slot_bytes,
        vram_slots=8,
        ram_slot_bytes=slot_bytes,
        ram_slots=8,
    ) as mem:
        src = torch.arange(1024 * 1024, dtype=torch.uint8, pin_memory=True)
        src_ptr = int(src.data_ptr())
        mem.stage_host(0, src_ptr, src.numel())
        mem.h2d_async(0, 0, src.numel())
        mem.sync()

        # Read a tiny value back through a normal CUDA tensor that aliases the
        # native VRAM slot. The pointer never required another allocation.
        ptr = mem.vram_ptr(0)
        raw = (ctypes.c_ubyte * src.numel()).from_address(mem.ram_ptr(1))
        del raw

        mem.d2h_async(0, 1, src.numel())
        mem.sync()
        result = ctypes.string_at(mem.ram_ptr(1), src.numel())
        expected = src.cpu().numpy().tobytes()
        if result != expected:
            raise AssertionError("native H2D/D2H round-trip mismatch")

        print(f"native_memory=PASS|vram_slot_ptr=0x{ptr:x}|stats={mem.stats()}")


if __name__ == "__main__":
    main()
