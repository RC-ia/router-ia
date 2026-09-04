from __future__ import annotations

"""Physical-RAM guard v3.

Do not confuse physical RAM available right now with the router's cache
capacity. The cache budget is derived from installed physical RAM; current
availability is used only as an emergency pressure signal.

This module intentionally ignores pagefile capacity. Cold Q4 experts are
spilled to the existing SSD backing tier when physical RAM approaches the
reserve floor.
"""

import ctypes
import os
import sys
from pathlib import Path

from . import qwen36_cached_loop as cached
from . import qwen36_expert_q4_hierarchy_fixed as q4

PHYSICAL_RAM_RESERVE_GB = max(
    float(os.getenv("QWEN36_PHYSICAL_RAM_RESERVE_GB", "1.5")), 0.25
)
PHYSICAL_RAM_RESERVE_BYTES = int(PHYSICAL_RAM_RESERVE_GB * 1024**3)
EMERGENCY_HEADROOM_GB = max(
    float(os.getenv("QWEN36_PHYSICAL_RAM_EMERGENCY_HEADROOM_GB", "0.50")),
    0.10,
)
EMERGENCY_HEADROOM_BYTES = int(EMERGENCY_HEADROOM_GB * 1024**3)


def _memory_status() -> tuple[int, int]:
    """Return (total physical RAM, available physical RAM), never pagefile."""
    if sys.platform.startswith("win"):
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys), int(status.ullAvailPhys)
        return 0, 0

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total = int(os.sysconf("SC_PHYS_PAGES")) * page_size
        available = int(os.sysconf("SC_AVPHYS_PAGES")) * page_size
        return total, max(available, 0)
    except (AttributeError, OSError, ValueError):
        return 0, 0


def _generic_ram_bytes() -> int:
    total = 0
    for store in getattr(cached, "_stores", {}).values():
        try:
            total += int(store.ram_cache.bytes_used)
        except AttributeError:
            pass
    return total


def _q4_ram_bytes() -> int:
    total = 0
    for cache in getattr(q4.official, "_EXPERT_CACHES", {}).values():
        try:
            total += int(cache.q4_bytes_used)
        except AttributeError:
            pass
    return total


def _configured_q4_budget() -> int:
    total, _ = _memory_status()
    if total <= 0:
        return int(q4.Q4_RAM_BUDGET_BYTES)
    return min(
        int(q4.Q4_RAM_BUDGET_BYTES),
        max(total - PHYSICAL_RAM_RESERVE_BYTES - int(q4.GENERIC_RESIDENT_BYTES), 0),
    )


def _emergency_spill(cache, root: Path) -> None:
    """Move cold Q4 entries to SSD until physical headroom is recovered."""
    _, available = _memory_status()
    if available <= 0:
        return
    floor = PHYSICAL_RAM_RESERVE_BYTES
    recovery = floor + EMERGENCY_HEADROOM_BYTES
    while available < floor:
        victim = None
        with cache.lock:
            for layer in range(cache.layers):
                bank = cache.q4_entries.get(layer)
                if bank:
                    expert_id, entry = bank.popitem(last=False)
                    cache._erase(layer, expert_id, "q4")
                    victim = (layer, expert_id, entry)
                    break
        if victim is None:
            return
        layer, expert_id, entry = victim
        written = q4._save_ssd(root, layer, expert_id, entry)
        with cache.lock:
            cache._q4_ssd_writes += 1
            cache._q4_ssd_bytes_written += written
            cache._q4_ram_evictions_to_ssd += 1
            cache.q4_drops += 1
            cache.q4_ram_evictions += 1
        _, available = _memory_status()
        if available >= recovery:
            return


def _guarded_q4_ram_insert(cache, root: Path, layer: int, expert_id: int, entry) -> None:
    old_budget = cache._q4_ram_budget
    try:
        cache._q4_ram_budget = _configured_q4_budget()
        q4._ram_insert(cache, root, layer, expert_id, entry)
    finally:
        cache._q4_ram_budget = old_budget
    _emergency_spill(cache, root)


_ORIGINAL_RAM_PUT = cached._PriorityTensorCache.put


def _guarded_ram_put(self, name, tensor):
    if self.name != "ram":
        return _ORIGINAL_RAM_PUT(self, name, tensor)
    total, _ = _memory_status()
    old_budget = self.max_bytes
    try:
        if total > 0:
            self.max_bytes = min(
                old_budget,
                max(total - PHYSICAL_RAM_RESERVE_BYTES - _q4_ram_bytes(), 0),
            )
        return _ORIGINAL_RAM_PUT(self, name, tensor)
    finally:
        self.max_bytes = old_budget


def print_memory_status(root: Path, label: str = "status") -> None:
    total, available = _memory_status()
    if total <= 0:
        print("  physical_ram_guard_v3: unavailable")
        return
    print(
        f"  physical_ram_guard_v3 {label}: "
        f"total={total / 1024**3:.2f} GiB | "
        f"available={available / 1024**3:.2f} GiB | "
        f"q4={_q4_ram_bytes() / 1024**3:.2f} GiB | "
        f"generic={_generic_ram_bytes() / 1024**3:.2f} GiB | "
        f"reserve={PHYSICAL_RAM_RESERVE_GB:.2f} GiB | "
        "pagefile=not-counted"
    )


q4._ram_insert = _guarded_q4_ram_insert
cached._PriorityTensorCache.put = _guarded_ram_put

print(
    "physical_ram_guard_v3=enabled|"
    f"total-based-q4-budget={_configured_q4_budget() / 1024**3:.2f}GiB|"
    f"reserve={PHYSICAL_RAM_RESERVE_GB:.2f}GiB|"
    f"emergency-headroom={EMERGENCY_HEADROOM_GB:.2f}GiB|"
    "pagefile=not-counted|spill=Q4-SSD"
)
