from __future__ import annotations

"""Physical-RAM-only governor for the Qwen3.6 hierarchical router.

The operating system may still have a pagefile/swap, but the router never
counts it as usable cache capacity. Host caches are throttled against actual
physical-memory availability and spill through the existing Q4 SSD tier before
RAM pressure becomes the next bottleneck.

This does not attempt to disable the OS pagefile globally. It makes the router
pagefile-blind: its own cache budgets are based only on physical RAM and it
spills cold Q4 experts to the router's SSD tier instead of deliberately filling
physical RAM to the point where the OS must page.
"""

import ctypes
import os
import sys
from pathlib import Path

from . import qwen36_cached_loop as cached
from . import qwen36_expert_q4_hierarchy as q4_hierarchy

PHYSICAL_RAM_RESERVE_GB = max(
    float(os.getenv("QWEN36_PHYSICAL_RAM_RESERVE_GB", "1.5")), 0.25
)
PHYSICAL_RAM_RESERVE_BYTES = int(PHYSICAL_RAM_RESERVE_GB * 1024**3)

_ORIGINAL_RAM_PUT = cached._PriorityTensorCache.put
_ORIGINAL_Q4_RAM_INSERT = q4_hierarchy._ram_insert


def _physical_available_bytes() -> int:
    """Return available physical RAM only; swap/pagefile is deliberately ignored."""
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
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return max(int(pages) * int(page_size), 0)
    except (AttributeError, OSError, ValueError):
        return 0


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
    for cache in getattr(q4_hierarchy.official, "_EXPERT_CACHES", {}).values():
        try:
            total += int(cache.q4_bytes_used)
        except AttributeError:
            pass
    return total


def _host_cache_target(configured: int, current: int, other: int) -> int:
    available = _physical_available_bytes()
    if available <= 0:
        return max(current, 0)
    headroom = max(available - PHYSICAL_RAM_RESERVE_BYTES - other, 0)
    return min(int(configured), int(headroom))


def _guarded_ram_put(self, name, tensor):
    if self.name != "ram":
        return _ORIGINAL_RAM_PUT(self, name, tensor)

    old_budget = self.max_bytes
    try:
        current = int(self.bytes_used)
        other = _q4_ram_bytes()
        self.max_bytes = _host_cache_target(old_budget, current, other)
        return _ORIGINAL_RAM_PUT(self, name, tensor)
    finally:
        self.max_bytes = old_budget


def _guarded_q4_ram_insert(cache, root: Path, layer: int, expert_id: int, entry):
    old_budget = cache._q4_ram_budget
    try:
        current = int(cache.q4_bytes_used)
        other = _generic_ram_bytes()
        cache._q4_ram_budget = _host_cache_target(old_budget, current, other)
        return _ORIGINAL_Q4_RAM_INSERT(cache, root, layer, expert_id, entry)
    finally:
        cache._q4_ram_budget = old_budget


def _memory_snapshot() -> tuple[int, int, int, int, int]:
    available = _physical_available_bytes()
    generic = _generic_ram_bytes()
    q4 = _q4_ram_bytes()
    managed = generic + q4
    safe = max(available - PHYSICAL_RAM_RESERVE_BYTES, 0)
    return available, safe, generic, q4, managed


def _print_memory_status(root: Path, label: str) -> None:
    available, safe, generic, q4, managed = _memory_snapshot()
    if available <= 0:
        print("  physical_ram_guard: unavailable")
        return
    print(
        f"  physical_ram_guard {label}: "
        f"available={available / 1024**3:.2f} GiB | "
        f"safe-cache={safe / 1024**3:.2f} GiB | "
        f"managed={managed / 1024**3:.2f} GiB | "
        f"generic={generic / 1024**3:.2f} GiB | "
        f"q4={q4 / 1024**3:.2f} GiB | "
        f"reserve={PHYSICAL_RAM_RESERVE_GB:.2f} GiB | swap=ignored"
    )


cached._PriorityTensorCache.put = _guarded_ram_put
q4_hierarchy._ram_insert = _guarded_q4_ram_insert

print(
    "physical_ram_guard=enabled|"
    f"reserve={PHYSICAL_RAM_RESERVE_GB:.2f}GiB|"
    "source=physical-available|swap=ignored|spill=Q4-SSD"
)
