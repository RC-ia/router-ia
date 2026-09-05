"""Logical validation of the async-acquire state machine (Proposal #1).

This does NOT require a GPU or the compiled native library. It mirrors the
state transitions implemented in ``native_memory/router_memory.cu`` in pure
Python and asserts the invariants the C code must uphold:

  1. ``acquire_async`` returns immediately and puts the block in ``loading``.
  2. A second ``acquire_async`` on the same block does NOT issue a duplicate
     transfer (reuses the reserved slot).
  3. Eviction never reuses a slot whose owner is ``loading``.
  4. ``wait_acquire`` promotes ``loading`` -> ``resident``.
  5. ``unregister_block`` synchronizes (waits) before freeing a loading block.
  6. The legacy synchronous ``acquire`` waits for a coincident async load and
     does not duplicate it.

Run this anywhere::

    python experimental/async_memory/simulate_async_acquire.py
"""

from __future__ import annotations

INVALID = -(2**31)  # uint32 -1 sentinel, kept as a large negative int


class SimMemoryManager:
    def __init__(self, vram_slots: int, ram_slots: int) -> None:
        self.vram_owner: list[int] = [-1] * vram_slots
        self.ram_used: list[bool] = [False] * ram_slots
        self.lru: list[int] = []
        # block_id -> dict
        self.blocks: dict[int, dict] = {}
        self.transfers: int = 0  # counts issued H2D copies
        self.cache_hits = 0
        self.cache_misses = 0
        self.evictions = 0

    # --- internal helpers mirroring the C ---
    def _load(self, owner: int) -> bool:
        return any(b.get("loading_slot") is not None for b in self.blocks.values()) and any(
            b.get("loading_slot") == owner for b in self.blocks.values()
        )

    def _slot_is_loading(self, slot: int) -> bool:
        return any(b.get("loading_slot") == slot for b in self.blocks.values())

    def _find_free(self) -> int | None:
        for i in range(len(self.vram_owner)):
            if self.vram_owner[i] < 0 and not self._slot_is_loading(i):
                return i
        return None

    def _lru_remove(self, slot: int) -> None:
        if slot in self.lru:
            self.lru.remove(slot)

    def _lru_touch(self, slot: int) -> None:
        self._lru_remove(slot)
        self.lru.append(slot)

    def _evict(self, slot: int) -> bool:
        owner = self.vram_owner[slot]
        if owner < 0:
            return True
        block = self.blocks[owner]
        if block["pin"] != 0:
            return False
        # D2H would happen here; we just release
        self.vram_owner[slot] = -1
        block["vram_slot"] = None
        self._lru_remove(slot)
        self.evictions += 1
        return True

    # --- public API mirror ---
    def register(self, block_id: int, bytes_: int) -> None:
        assert block_id not in self.blocks
        ram_slot = next(i for i, used in enumerate(self.ram_used) if not used)
        self.ram_used[ram_slot] = True
        self.blocks[block_id] = {
            "ram_slot": ram_slot,
            "vram_slot": None,
            "loading_slot": None,
            "bytes": bytes_,
            "pin": 0,
        }

    def acquire_async(self, block_id: int) -> int:
        block = self.blocks[block_id]
        if block["vram_slot"] is not None:
            self.cache_hits += 1
            self._lru_touch(block["vram_slot"])
            return block["vram_slot"]
        if block["loading_slot"] is not None:
            return block["loading_slot"]  # no duplicate transfer
        self.cache_misses += 1
        slot = self._find_free()
        if slot is None:
            slot = next(
                (c for c in self.lru if self.vram_owner[c] >= 0 and not self._slot_is_loading(c)),
                None,
            )
            if slot is None:
                raise RuntimeError("no evictable slot")
            assert self._evict(slot)
        # issue transfer, do NOT synchronize
        self.transfers += 1
        block["loading_slot"] = slot
        return slot

    def is_loading(self, block_id: int) -> bool:
        return self.blocks[block_id]["loading_slot"] is not None

    def wait_acquire(self, block_id: int) -> None:
        block = self.blocks[block_id]
        if block["loading_slot"] is None:
            assert block["vram_slot"] is not None
            return
        slot = block["loading_slot"]
        block["vram_slot"] = slot
        self.vram_owner[slot] = block_id
        block["loading_slot"] = None
        self._lru_touch(slot)

    def acquire(self, block_id: int) -> int:  # legacy synchronous mirror
        block = self.blocks[block_id]
        if block["loading_slot"] is not None:
            # wait for coincident async load, then return it
            self.wait_acquire(block_id)
            return block["vram_slot"]
        if block["vram_slot"] is not None:
            self.cache_hits += 1
            self._lru_touch(block["vram_slot"])
            return block["vram_slot"]
        return self.acquire_async(block_id)

    def unregister(self, block_id: int) -> None:
        block = self.blocks[block_id]
        if block["loading_slot"] is not None:
            self.wait_acquire(block_id)  # sync before freeing RAM slot
        if block["vram_slot"] is not None:
            self._evict(block["vram_slot"])
        self.ram_used[block["ram_slot"]] = False
        del self.blocks[block_id]


def main() -> None:
    mm = SimMemoryManager(vram_slots=2, ram_slots=8)

    # Invariant 1: async returns immediately, block goes loading.
    mm.register(10, 100)
    mm.register(11, 100)
    mm.register(12, 100)
    s10 = mm.acquire_async(10)
    assert mm.is_loading(10), "block 10 should be loading after acquire_async"
    assert mm.transfers == 1
    print("PASS  invariant 1: acquire_async returns immediately, block loading")

    # Invariant 2: second async on same block reuses slot, no duplicate transfer.
    s10_again = mm.acquire_async(10)
    assert s10_again == s10, "second acquire_async must reuse reserved slot"
    assert mm.transfers == 1, "no duplicate H2D must be issued"
    print("PASS  invariant 2: no duplicate transfer on re-acquire")

    # Invariant 3: eviction skips loading slots.
    s11 = mm.acquire_async(11)
    # vram now has slots 0 (loading 10) and 1 (loading 11). Exhaust free slots,
    # then try to acquire 12 -> must NOT evict a loading slot. With 2 slots
    # both loading, 12 has no evictable slot -> RuntimeError.
    try:
        mm.acquire_async(12)
        raise AssertionError("should have no evictable slot while both loading")
    except RuntimeError:
        pass
    print("PASS  invariant 3: eviction refuses to reuse a loading slot")

    # Invariant 4: wait_acquire promotes loading -> resident.
    mm.wait_acquire(10)
    assert not mm.is_loading(10)
    assert mm.blocks[10]["vram_slot"] == s10
    assert mm.vram_owner[s10] == 10
    print("PASS  invariant 4: wait_acquire promotes to resident")

    # Invariant 5: unregister syncs before freeing a loading block.
    mm.unregister(11)  # 11 is still loading (never waited)
    assert 11 not in mm.blocks, "unregister must remove the block"
    print("PASS  invariant 5: unregister syncs then frees loading block")

    # Invariant 6: legacy acquire waits for coincident async load, no dup.
    mm2 = SimMemoryManager(vram_slots=4, ram_slots=8)
    mm2.register(20, 100)
    s20 = mm2.acquire_async(20)
    transfers_before = mm2.transfers
    s20_legacy = mm2.acquire(20)  # legacy path sees loading -> waits
    assert s20_legacy == s20, "legacy acquire returns the same resident slot"
    assert mm2.transfers == transfers_before, "legacy acquire must not duplicate transfer"
    assert not mm2.is_loading(20), "legacy acquire must promote to resident"
    print("PASS  invariant 6: legacy acquire coalesces with async, no dup")

    print("\nALL ASYNC-ACQUIRE STATE-MACHINE INVARIANTS PASSED")


if __name__ == "__main__":
    main()