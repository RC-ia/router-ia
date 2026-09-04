from __future__ import annotations

"""Physical-RAM governor for the Qwen3.6 router.

The router does not use pagefile size as cache capacity. Cache budgets are
bounded independently; current physical availability is only an emergency
signal that causes cold Q4 experts to spill to the router's SSD backing.
"""

import ctypes
import os
import sys
from pathlib import Path

from . import qwen36_cached_loop as cached
from . import qwen36_expert_q4_hierarchy_fixed as q4
from . import qwen36_chat_batch as chat

PHYSICAL_RAM_RESERVE_GB = max(float(os.getenv("QWEN36_PHYSICAL_RAM_RESERVE_GB", "1.5")), 0.25)
PHYSICAL_RAM_RESERVE_BYTES = int(PHYSICAL_RAM_RESERVE_GB * 1024**3)
EMERGENCY_TRIGGER_GB = max(float(os.getenv("QWEN36_PHYSICAL_RAM_EMERGENCY_TRIGGER_GB", "2.0")), PHYSICAL_RAM_RESERVE_GB)
EMERGENCY_TRIGGER_BYTES = int(EMERGENCY_TRIGGER_GB * 1024**3)
EMERGENCY_RECOVERY_GB = max(float(os.getenv("QWEN36_PHYSICAL_RAM_EMERGENCY_RECOVERY_GB", "2.5")), EMERGENCY_TRIGGER_GB + 0.10)
EMERGENCY_RECOVERY_BYTES = int(EMERGENCY_RECOVERY_GB * 1024**3)

_ORIGINAL_Q4_RAM_INSERT = q4._ram_insert
_ORIGINAL_RAM_PUT = cached._PriorityTensorCache.put
_ORIGINAL_PRINT_CACHE = chat.print_cache


def _memory_status() -> tuple[int, int]:
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
        return (
            int(os.sysconf("SC_PHYS_PAGES")) * page_size,
            int(os.sysconf("SC_AVPHYS_PAGES")) * page_size,
        )
    except (AttributeError, OSError, ValueError):
        return 0, 0


def _generic_ram_bytes() -> int:
    return sum(int(store.ram_cache.bytes_used) for store in getattr(cached, "_stores", {}).values() if hasattr(store, "ram_cache"))


def _q4_ram_bytes() -> int:
    return sum(int(cache.q4_bytes_used) for cache in getattr(q4.official, "_EXPERT_CACHES", {}).values() if hasattr(cache, "q4_bytes_used"))


def _emergency_spill(cache, root: Path) -> None:
    _total, available = _memory_status()
    if available <= 0 or available >= EMERGENCY_TRIGGER_BYTES:
        return
    while available < EMERGENCY_RECOVERY_BYTES:
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
        _total, available = _memory_status()


def _guarded_q4_ram_insert(cache, root: Path, layer: int, expert_id: int, entry) -> None:
    old_budget = cache._q4_ram_budget
    try:
        # Use the configured 1.5 GiB Q4 RAM tier. Do not collapse the budget
        # just because Windows reports a transiently low available-RAM value.
        cache._q4_ram_budget = int(q4.Q4_RAM_BUDGET_BYTES)
        _ORIGINAL_Q4_RAM_INSERT(cache, root, layer, expert_id, entry)
    finally:
        cache._q4_ram_budget = old_budget
    _emergency_spill(cache, root)


def _guarded_ram_put(self, name, tensor):
    # Keep the generic cache ceiling unchanged; Q4 has its own physical-RAM
    # emergency mechanism and SSD backing.
    return _ORIGINAL_RAM_PUT(self, name, tensor)


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
        f"q4-budget={q4.Q4_RAM_BUDGET_BYTES / 1024**3:.2f} GiB | "
        f"trigger={EMERGENCY_TRIGGER_GB:.2f} GiB | "
        f"recovery={EMERGENCY_RECOVERY_GB:.2f} GiB | "
        f"reserve={PHYSICAL_RAM_RESERVE_GB:.2f} GiB | pagefile=not-counted"
    )


def _print_cache(root: Path, label: str) -> None:
    _ORIGINAL_PRINT_CACHE(root, label)
    print_memory_status(root, label)


q4._ram_insert = _guarded_q4_ram_insert
cached._PriorityTensorCache.put = _guarded_ram_put
chat.print_cache = _print_cache

print(
    "physical_ram_guard_v3=enabled|"
    f"q4-budget={q4.Q4_RAM_BUDGET_BYTES / 1024**3:.2f}GiB|"
    f"trigger={EMERGENCY_TRIGGER_GB:.2f}GiB|"
    f"recovery={EMERGENCY_RECOVERY_GB:.2f}GiB|"
    f"reserve={PHYSICAL_RAM_RESERVE_GB:.2f}GiB|"
    "pagefile=not-counted|spill=Q4-SSD"
)
