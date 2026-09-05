from __future__ import annotations

"""High-level ctypes wrapper for the native logical memory manager."""

import ctypes
from typing import Any

from .qwen36_native_memory import NativeMemory, NativeMemoryError


class RouterMemoryManagerStats(ctypes.Structure):
    _fields_ = [
        ("h2d_calls", ctypes.c_uint64),
        ("d2h_calls", ctypes.c_uint64),
        ("bytes_h2d", ctypes.c_uint64),
        ("bytes_d2h", ctypes.c_uint64),
        ("sync_calls", ctypes.c_uint64),
        ("cache_hits", ctypes.c_uint64),
        ("cache_misses", ctypes.c_uint64),
        ("evictions", ctypes.c_uint64),
    ]


class MemoryManager:
    """Logical block residency manager backed by native CUDA memory."""

    def __init__(
        self,
        vram_slot_bytes: int,
        vram_slots: int,
        ram_slot_bytes: int,
        ram_slots: int,
        streams: int = 2,
    ) -> None:
        self._memory = NativeMemory(
            vram_slot_bytes=vram_slot_bytes,
            vram_slots=vram_slots,
            ram_slot_bytes=ram_slot_bytes,
            ram_slots=ram_slots,
            streams=streams,
        )
        lib = self._memory.lib
        lib.router_mm_register_block.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint64,
        ]
        lib.router_mm_register_block.restype = ctypes.c_int
        lib.router_mm_unregister_block.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        lib.router_mm_unregister_block.restype = ctypes.c_int
        lib.router_mm_is_registered.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        lib.router_mm_is_registered.restype = ctypes.c_int
        lib.router_mm_is_resident.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        lib.router_mm_is_resident.restype = ctypes.c_int
        lib.router_mm_acquire.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint32),
        ]
        lib.router_mm_acquire.restype = ctypes.c_int
        lib.router_mm_touch.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        lib.router_mm_touch.restype = ctypes.c_int
        lib.router_mm_pin.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        lib.router_mm_pin.restype = ctypes.c_int
        lib.router_mm_unpin.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        lib.router_mm_unpin.restype = ctypes.c_int
        lib.router_mm_evict.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        lib.router_mm_evict.restype = ctypes.c_int
        lib.router_mm_stats.argtypes = [ctypes.c_void_p]
        lib.router_mm_stats.restype = RouterMemoryManagerStats

        # Proposal #1: asynchronous acquire API, added alongside the legacy
        # synchronous acquire. The legacy path is left untouched.
        lib.router_mm_acquire_async.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint32),
        ]
        lib.router_mm_acquire_async.restype = ctypes.c_int
        lib.router_mm_wait_acquire.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        lib.router_mm_wait_acquire.restype = ctypes.c_int
        lib.router_mm_is_loading.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        lib.router_mm_is_loading.restype = ctypes.c_int

    def __enter__(self) -> "MemoryManager":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    @property
    def memory(self) -> NativeMemory:
        return self._memory

    def close(self) -> None:
        self._memory.close()

    def _ptr(self) -> ctypes.c_void_p:
        return ctypes.c_void_p(self._memory.handle)

    @staticmethod
    def _host_ptr(src: Any) -> int:
        if hasattr(src, "data_ptr"):
            return int(src.data_ptr())
        if isinstance(src, int):
            return src
        return ctypes.addressof(ctypes.c_char.from_buffer(src))

    def register_block(self, block_id: int, src: Any, bytes: int) -> None:
        ptr = self._host_ptr(src)
        if not self._memory.lib.router_mm_register_block(
            self._ptr(), int(block_id), ptr, int(bytes)
        ):
            raise NativeMemoryError(f"failed to register block {block_id}")

    def unregister_block(self, block_id: int) -> None:
        if not self._memory.lib.router_mm_unregister_block(self._ptr(), int(block_id)):
            raise NativeMemoryError(f"failed to unregister block {block_id}")

    def is_registered(self, block_id: int) -> bool:
        return bool(self._memory.lib.router_mm_is_registered(self._ptr(), int(block_id)))

    def is_resident(self, block_id: int) -> bool:
        return bool(self._memory.lib.router_mm_is_resident(self._ptr(), int(block_id)))

    def acquire(self, block_id: int, bytes: int) -> int:
        out = ctypes.c_uint32()
        if not self._memory.lib.router_mm_acquire(
            self._ptr(), int(block_id), int(bytes), ctypes.byref(out)
        ):
            raise NativeMemoryError(f"failed to acquire block {block_id}")
        return int(out.value)

    def touch(self, block_id: int) -> None:
        if not self._memory.lib.router_mm_touch(self._ptr(), int(block_id)):
            raise NativeMemoryError(f"failed to touch block {block_id}")

    def pin(self, block_id: int) -> None:
        if not self._memory.lib.router_mm_pin(self._ptr(), int(block_id)):
            raise NativeMemoryError(f"failed to pin block {block_id}")

    def unpin(self, block_id: int) -> None:
        if not self._memory.lib.router_mm_unpin(self._ptr(), int(block_id)):
            raise NativeMemoryError(f"failed to unpin block {block_id}")

    def evict(self, block_id: int) -> None:
        if not self._memory.lib.router_mm_evict(self._ptr(), int(block_id)):
            raise NativeMemoryError(f"failed to evict block {block_id}")

    def acquire_async(self, block_id: int, bytes: int) -> int:
        """Issue the block's H2D transfer and return immediately (Proposal #1).

        The returned slot is reserved but NOT yet safe to read. Call
        :meth:`wait_acquire` before using the VRAM pointer. Several
        ``acquire_async`` calls may be dispatched back-to-back so the
        transfers overlap with computation on already-resident blocks.
        """
        out = ctypes.c_uint32()
        if not self._memory.lib.router_mm_acquire_async(
            self._ptr(), int(block_id), int(bytes), ctypes.byref(out)
        ):
            raise NativeMemoryError(f"failed to acquire block {block_id} (async)")
        return int(out.value)

    def is_loading(self, block_id: int) -> bool:
        return bool(self._memory.lib.router_mm_is_loading(self._ptr(), int(block_id)))

    def wait_acquire(self, block_id: int) -> None:
        """Synchronize an in-flight async acquire and promote it to resident."""
        if not self._memory.lib.router_mm_wait_acquire(self._ptr(), int(block_id)):
            raise NativeMemoryError(f"failed to wait for block {block_id}")

    def stats(self) -> dict[str, int]:
        s = self._memory.lib.router_mm_stats(self._ptr())
        return {
            "h2d_calls": int(s.h2d_calls),
            "d2h_calls": int(s.d2h_calls),
            "bytes_h2d": int(s.bytes_h2d),
            "bytes_d2h": int(s.bytes_d2h),
            "sync_calls": int(s.sync_calls),
            "cache_hits": int(s.cache_hits),
            "cache_misses": int(s.cache_misses),
            "evictions": int(s.evictions),
        }
