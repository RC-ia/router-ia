from __future__ import annotations

"""Model-like residency benchmark for the native memory backend.

Run from the repository root:
    python native_memory/model_residency_benchmark.py

This simulates a model larger than VRAM:
- 32 logical model blocks live in a CPU-side backing store;
- only 8 physical VRAM slots are available;
- requests generate hits and misses;
- misses evict an existing VRAM block to a RAM staging slot and load the
  requested block into that VRAM slot;
- transfers are submitted asynchronously and synchronized once per batch;
- payloads are checked after every batch so a fast run cannot hide corruption.

The benchmark compares several access patterns and stream counts and reports
request latency, hit rate, transfer volume, and effective H2D/D2H throughput.
"""

import ctypes
import os
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DLL = ROOT / "native_memory" / "build-ninja" / "router_ia_native_memory.dll"
os.environ.setdefault("ROUTER_IA_NATIVE_MEMORY_LIB", str(DLL))

from router_ia.qwen36_native_memory import NativeMemory


SLOT_BYTES = 8 * 1024 * 1024
PAYLOAD_BYTES = 4 * 1024 * 1024
VRAM_SLOTS = 8
RAM_SLOTS = 8
LOGICAL_BLOCKS = 32
BATCH_SIZE = 8
ROUNDS = 64
STREAM_VARIANTS = (1, 2, 4, 8)


def make_payload(size: int, block_id: int, version: int = 0) -> bytes:
    # Deterministic pseudo-weight blob. Different blocks and reload versions
    # remain distinguishable without relying on random state.
    a = (block_id * 17 + version * 7 + 29) & 0xFF
    b = (block_id * 131 + version * 19 + 11) & 0xFF
    out = bytearray(size)
    for i in range(size):
        out[i] = (i * 29 + a + ((i >> 8) * b)) & 0xFF
    return bytes(out)


def build_trace(name: str) -> list[int]:
    trace: list[int] = []

    if name == "sequential":
        for r in range(ROUNDS):
            base = (r * BATCH_SIZE) % LOGICAL_BLOCKS
            trace.extend((base + i) % LOGICAL_BLOCKS for i in range(BATCH_SIZE))

    elif name == "local":
        hot = list(range(8))
        cold = list(range(8, 32))
        for r in range(ROUNDS):
            if r % 4 == 3:
                start = ((r // 4) * BATCH_SIZE) % len(cold)
                group = [cold[(start + i) % len(cold)] for i in range(BATCH_SIZE)]
            else:
                group = [hot[(r * 3 + i) % len(hot)] for i in range(BATCH_SIZE)]
            trace.extend(group)

    elif name == "random":
        # Deterministic pseudo-random sequence so runs are reproducible.
        state = 0x12345678
        for _ in range(ROUNDS * BATCH_SIZE):
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            trace.append(state % LOGICAL_BLOCKS)

    else:
        raise ValueError(f"unknown trace: {name}")

    return trace


def read_ram(mem: NativeMemory, slot: int, size: int) -> bytes:
    return ctypes.string_at(mem.ram_ptr(slot), size)


def stage_bytes(
    mem: NativeMemory,
    buffers: list[ctypes.Array[ctypes.c_char]],
    ram_slot: int,
    payload: bytes,
) -> None:
    buf = ctypes.create_string_buffer(payload)
    buffers.append(buf)
    mem.stage_host(ram_slot, ctypes.addressof(buf), len(payload))


def run_case(streams: int, pattern: str) -> dict[str, float | int | str]:
    trace = build_trace(pattern)
    buffers: list[ctypes.Array[ctypes.c_char]] = []

    # CPU-side backing store: represents blocks that are not currently
    # resident in VRAM. RAM slots are only transfer staging buffers.
    backing = {
        block_id: make_payload(PAYLOAD_BYTES, block_id)
        for block_id in range(LOGICAL_BLOCKS)
    }

    # physical_slot -> logical_block
    resident: dict[int, int] = {}
    # logical_block -> physical_slot
    location: dict[int, int] = {}
    lru: deque[int] = deque()

    requests = 0
    hits = 0
    misses = 0
    start = time.perf_counter()

    with NativeMemory(
        vram_slot_bytes=SLOT_BYTES,
        vram_slots=VRAM_SLOTS,
        ram_slot_bytes=SLOT_BYTES,
        ram_slots=RAM_SLOTS,
        streams=streams,
    ) as mem:
        # Fill the initial working set. This is cold-start traffic and is
        # intentionally included in the benchmark.
        for slot in range(VRAM_SLOTS):
            block_id = slot
            stage_bytes(mem, buffers, slot, backing[block_id])
            mem.h2d_async(slot, slot, PAYLOAD_BYTES)
            resident[slot] = block_id
            location[block_id] = slot
            lru.append(slot)
        mem.sync()

        for batch_start in range(0, len(trace), BATCH_SIZE):
            batch = trace[batch_start : batch_start + BATCH_SIZE]

            # Do not issue two loads to the same physical slot in one batch.
            # We first identify the unique logical blocks requested by the
            # simulated router.
            batch_unique: list[int] = []
            seen: set[int] = set()
            for block_id in batch:
                if block_id not in seen:
                    batch_unique.append(block_id)
                    seen.add(block_id)

            pending_evictions: list[tuple[int, int]] = []
            pending_loads: list[tuple[int, int]] = []
            used_slots: set[int] = set()

            for block_id in batch_unique:
                requests += 1

                if block_id in location:
                    hits += 1
                    slot = location[block_id]
                    used_slots.add(slot)
                    try:
                        lru.remove(slot)
                    except ValueError:
                        pass
                    lru.append(slot)
                    continue

                misses += 1

                # Prefer a free VRAM slot; otherwise evict the LRU slot that
                # is not already selected in this batch.
                free = next((s for s in range(VRAM_SLOTS) if s not in resident), None)
                if free is not None:
                    slot = free
                else:
                    candidates = [s for s in lru if s not in used_slots]
                    if not candidates:
                        # This can happen when a batch requests more unique
                        # blocks than there are free physical slots.
                        candidates = list(lru)
                    slot = candidates[0]
                    old_block = resident[slot]

                    # RAM slot == physical slot gives each VRAM slot a stable
                    # staging location and keeps the ABI simple.
                    pending_evictions.append((slot, old_block))

                    del location[old_block]
                    del resident[slot]
                    try:
                        lru.remove(slot)
                    except ValueError:
                        pass

                pending_loads.append((slot, block_id))
                used_slots.add(slot)

            # Evictions must complete before their RAM staging slots are reused
            # for incoming loads. There is no native event API yet, so this is
            # the correctness barrier for the simulated router scheduler.
            for slot, _old_block in pending_evictions:
                mem.d2h_async(slot, slot, PAYLOAD_BYTES)
            if pending_evictions:
                mem.sync()

                for slot, old_block in pending_evictions:
                    evicted = read_ram(mem, slot, PAYLOAD_BYTES)
                    expected_old = backing[old_block]
                    if evicted != expected_old:
                        raise AssertionError(
                            f"eviction corruption: slot={slot}, block={old_block}"
                        )
                    backing[old_block] = evicted

            # Load missing blocks.
            for slot, block_id in pending_loads:
                stage_bytes(mem, buffers, slot, backing[block_id])
                mem.h2d_async(slot, slot, PAYLOAD_BYTES)

            mem.sync()

            for slot, block_id in pending_loads:
                resident[slot] = block_id
                location[block_id] = slot
                try:
                    lru.remove(slot)
                except ValueError:
                    pass
                lru.append(slot)

            # Verify every newly loaded block by round-tripping it through its
            # staging RAM slot. This is intentionally strict rather than fast.
            for slot, block_id in pending_loads:
                mem.d2h_async(slot, slot, PAYLOAD_BYTES)
            if pending_loads:
                mem.sync()
                for slot, block_id in pending_loads:
                    got = read_ram(mem, slot, PAYLOAD_BYTES)
                    if got != backing[block_id]:
                        raise AssertionError(
                            f"load corruption: slot={slot}, block={block_id}"
                        )

    elapsed = time.perf_counter() - start
    stats = mem.stats() if False else None

    # NativeMemory has already been destroyed. Reconstruct transfer volume from
    # the counters we know were issued in this benchmark.
    with NativeMemory(
        vram_slot_bytes=SLOT_BYTES,
        vram_slots=1,
        ram_slot_bytes=SLOT_BYTES,
        ram_slots=1,
        streams=1,
    ) as counter_probe:
        stats = counter_probe.stats()

    # Recompute traffic from the trace accounting. We report request-level
    # traffic rather than relying on the destroyed native handle.
    # Initial fill: 8 H2D. Every miss: 1 D2H eviction + 1 H2D load.
    # Every miss is also round-tripped once for integrity (1 D2H).
    miss_count = misses
    h2d_calls = VRAM_SLOTS + miss_count
    d2h_calls = miss_count * 2
    h2d_bytes = h2d_calls * PAYLOAD_BYTES
    d2h_bytes = d2h_calls * PAYLOAD_BYTES

    return {
        "streams": streams,
        "pattern": pattern,
        "requests": requests,
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / requests if requests else 0.0,
        "seconds": elapsed,
        "requests_per_s": requests / elapsed if elapsed else float("inf"),
        "h2d_gbps": h2d_bytes / elapsed / 1e9 if elapsed else float("inf"),
        "d2h_gbps": d2h_bytes / elapsed / 1e9 if elapsed else float("inf"),
        "total_gbps": (h2d_bytes + d2h_bytes) / elapsed / 1e9 if elapsed else float("inf"),
        "h2d_calls": h2d_calls,
        "d2h_calls": d2h_calls,
    }


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não está disponível.")

    print("=" * 72)
    print("NATIVE MEMORY MODEL-RESIDENCY BENCHMARK")
    print("=" * 72)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Logical blocks : {LOGICAL_BLOCKS}")
    print(f"VRAM slots     : {VRAM_SLOTS}")
    print(f"Payload/block  : {PAYLOAD_BYTES / 1024 / 1024:.1f} MiB")
    print(f"Batch          : {BATCH_SIZE}")
    print(f"Requests       : {ROUNDS * BATCH_SIZE}")
    print(f"Streams        : {', '.join(map(str, STREAM_VARIANTS))}")

    results: list[dict[str, float | int | str]] = []

    for pattern in ("sequential", "local", "random"):
        print("\n" + "-" * 72)
        print(f"PATTERN: {pattern}")
        print("-" * 72)
        for streams in STREAM_VARIANTS:
            result = run_case(streams, pattern)
            results.append(result)
            print(
                f"  streams={streams} | "
                f"hit={float(result['hit_rate']) * 100:5.1f}% | "
                f"miss={int(result['misses']):4d} | "
                f"time={float(result['seconds']):.4f}s | "
                f"req/s={float(result['requests_per_s']):.1f} | "
                f"H2D={float(result['h2d_gbps']):.3f} GB/s | "
                f"D2H={float(result['d2h_gbps']):.3f} GB/s"
            )

    print("\n" + "=" * 72)
    print("RESUMO POR PADRÃO (MELHOR REQ/s)")
    print("=" * 72)

    for pattern in ("sequential", "local", "random"):
        group = [r for r in results if r["pattern"] == pattern]
        best = max(group, key=lambda r: float(r["requests_per_s"]))
        print(
            f"{pattern:>10}: streams={best['streams']} | "
            f"req/s={float(best['requests_per_s']):.1f} | "
            f"hit={float(best['hit_rate']) * 100:.1f}% | "
            f"total={float(best['total_gbps']):.3f} GB/s"
        )

    print("\n[integridade] Todas as cargas/evictions foram verificadas por round-trip.")
    print("BENCHMARK = PASS")


if __name__ == "__main__":
    main()
