from __future__ import annotations

"""Physical-RAM-only budgeting for the router.

This guard does not pretend to disable the Windows pagefile. It simply never
uses pagefile/commit capacity as cache budget: generic RAM cache and routed
Q4 RAM share a ceiling derived from currently available physical RAM, while
cold Q4 experts spill to the router's SSD backing tier.
"""

import ctypes
import os
import sys
from pathlib import Path

from . import qwen36_cached_loop as cached
from . import qwen36_expert_q4_hierarchy_fixed as q4_hierarchy

RESERVE_GB = max(float(os.getenv("QWEN36_PHYSICAL_RAM_RESERVE_GB", "1.5")), 0.25)
RESERVE_BYTES = int(RESERVE_GB * 1024**3)

_ORIGINAL_GENERIC_PUT = cached._PriorityTensorCache.put
_ORIGINAL_Q4_RAM_INSERT = q4_hierarchy._ram_insert


def physical_available_bytes() -> int:
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
            return int(status.ullAvailPhys)
        return 0

    try:
        return max(
            int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE")),
            0,
        )
    except (AttributeError, OSError, ValueError):
        return 0


def _generic_ram_bytes() -> int:
    return sum(
        int(store.ram_cache.bytes_used)
        for store in getattr(cached, "_stores", {}).values()
        if hasattr(store, "ram_cache")
    )


def _q4_ram_bytes() -> int:
    return sum(
        int(cache.q4_bytes_used)
        for cache in getattr(q4_hierarchy.official, "_EXPERT_CACHES", {}).values()
        if hasattr(cache, "q4_bytes_used")
    )


def _dynamic_budget(configured: int, other: int) -> int:
    available = physical_available_bytes()
    if available <= 0:
        return max(int(configured), 0)
    return min(int(configured), max(available - RESERVE_BYTES - int(other), 0))


def _guard_generic_put(self, name, tensor):
    if self.name != "ram":
        return _ORIGINAL_GENERIC_PUT(self, name, tensor)
    old = self.max_bytes
    try:
        self.max_bytes = _dynamic_budget(old, _q4_ram_bytes())
        return _ORIGINAL_GENERIC_PUT(self, name, tensor)
    finally:
        self.max_bytes = old


def _guard_q4_insert(cache, root: Path, layer: int, expert_id: int, entry):
    old = cache._q4_ram_budget
    try:
        cache._q4_ram_budget = _dynamic_budget(old, _generic_ram_bytes())
        return _ORIGINAL_Q4_RAM_INSERT(cache, root, layer, expert_id, entry)
    finally:
        cache._q4_ram_budget = old


def memory_snapshot() -> tuple[int, int, int, int]:
    available = physical_available_bytes()
    generic = _generic_ram_bytes()
    q4 = _q4_ram_bytes()
    return available, generic, q4, generic + q4


def print_status(label: str = "runtime") -> None:
    available, generic, q4, managed = memory_snapshot()
    if available <= 0:
        print("  physical_ram_guard: unavailable")
        return
    print(
        f"  physical_ram_guard {label}: "
        f"available={available / 1024**3:.2f} GiB | "
        f"managed={managed / 1024**3:.2f} GiB | "
        f"generic={generic / 1024**3:.2f} GiB | "
        f"q4={q4 / 1024**3:.2f} GiB | "
        f"reserve={RESERVE_GB:.2f} GiB | swap=not-budgeted"
    )


cached._PriorityTensorCache.put = _guard_generic_put
q4_hierarchy._ram_insert = _guard_q4_insert
q4_hierarchy.print_status = print_status

print(
    "physical_ram_guard=enabled|"
    f"reserve={RESERVE_GB:.2f}GiB|source=physical-available|"
    "swap=not-budgeted|spill=Q4-SSD"
)
