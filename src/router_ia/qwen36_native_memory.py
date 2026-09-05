from __future__ import annotations

"""Optional ctypes wrapper for the fixed-slot CUDA memory prototype.

The native runtime deliberately owns only storage and transfer primitives.
The Python router remains responsible for routing and expert metadata.
"""

import ctypes
import os
from pathlib import Path
from typing import Any


class RouterMemStats(ctypes.Structure):
    _fields_ = [
        ("h2d_calls", ctypes.c_uint64),
        ("d2h_calls", ctypes.c_uint64),
        ("bytes_h2d", ctypes.c_uint64),
        ("bytes_d2h", ctypes.c_uint64),
        ("sync_calls", ctypes.c_uint64),
    ]


class NativeMemoryError(RuntimeError):
    pass


def _candidate_paths() -> list[Path]:
    root = Path(__file__).resolve().parents[2]
    env = os.getenv("ROUTER_IA_NATIVE_MEMORY_LIB")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env).expanduser().resolve())
    native = root / "native_memory"
    for build_dir in (native / "build", native / "build-ninja"):
        for name in (
            "router_ia_native_memory.dll",
            "librouter_ia_native_memory.so",
            "librouter_ia_native_memory.dylib",
        ):
            candidates.append(build_dir / name)
            candidates.append(build_dir / "Release" / name)
            candidates.append(build_dir / "Debug" / name)
    return candidates


def _prepare_windows_dll_search_path() -> list[Path]:
    """Register CUDA/MSVC runtime directories so ctypes can load dependencies.

    Windows 10/11 restricts DLL dependency lookup for libraries loaded by
    ctypes. The native library links against the CUDA runtime (cudart), so
    adding CUDA's bin directory here makes the loader independent from the
    shell's PATH configuration.
    """
    if os.name != "nt":
        return []

    roots: list[Path] = []

    cuda_path = os.getenv("CUDA_PATH")
    if cuda_path:
        roots.append(Path(cuda_path))

    # Common NVIDIA CUDA Toolkit installation locations. Keep this fallback
    # version-agnostic so it also works after a toolkit upgrade.
    program_files = os.getenv("ProgramFiles", r"C:\\Program Files")
    cuda_root = Path(program_files) / "NVIDIA GPU Computing Toolkit" / "CUDA"
    if cuda_root.is_dir():
        for version_dir in sorted(cuda_root.glob("v*"), reverse=True):
            roots.append(version_dir)

    dll_dirs: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        bin_dir = root / "bin"
        if bin_dir.is_dir():
            resolved = bin_dir.resolve()
            if resolved not in seen:
                seen.add(resolved)
                try:
                    os.add_dll_directory(str(resolved))
                except (AttributeError, FileNotFoundError, OSError):
                    # Older Python/Windows combinations may not expose
                    # add_dll_directory. PATH fallback below still applies.
                    pass
                dll_dirs.append(resolved)

    # The CUDA DLL may also be discoverable through the current environment.
    # Add the directories to PATH as a compatibility fallback.
    if dll_dirs:
        current_path = os.environ.get("PATH", "")
        additions = os.pathsep.join(str(p) for p in dll_dirs if str(p) not in current_path)
        if additions:
            os.environ["PATH"] = additions + (os.pathsep + current_path if current_path else "")

    return dll_dirs


def _load_library() -> ctypes.CDLL:
    _prepare_windows_dll_search_path()
    for path in _candidate_paths():
        if path.is_file():
            try:
                return ctypes.CDLL(str(path))
            except OSError as exc:
                raise NativeMemoryError(
                    f"Native CUDA memory library exists but could not be loaded: {path}\n"
                    f"Windows dependency/loader error: {exc}\n"
                    "Check that the CUDA Toolkit runtime (cudart) matching the build "
                    "is installed and that the Visual C++ runtime is available."
                ) from exc
    raise NativeMemoryError(
        "Native CUDA memory library not found. Build native_memory with CMake "
        "or set ROUTER_IA_NATIVE_MEMORY_LIB."
    )


class NativeMemory:
    def __init__(
        self,
        vram_slot_bytes: int,
        vram_slots: int,
        ram_slot_bytes: int,
        ram_slots: int,
        streams: int = 1,
    ) -> None:
        if streams < 1:
            raise ValueError("streams must be >= 1")

        self.lib = _load_library()
        self._configure_abi()

        self.handle = self.lib.router_mem_create_ex(
            int(vram_slot_bytes),
            int(vram_slots),
            int(ram_slot_bytes),
            int(ram_slots),
            int(streams),
        )
        if not self.handle:
            raise NativeMemoryError(
                "router_mem_create_ex failed; check CUDA availability/memory"
            )

        self.vram_slot_bytes = int(vram_slot_bytes)
        self.vram_slots = int(vram_slots)
        self.ram_slot_bytes = int(ram_slot_bytes)
        self.ram_slots = int(ram_slots)
        self.streams = int(streams)

    def _configure_abi(self) -> None:
        lib = self.lib
        lib.router_mem_create.argtypes = [
            ctypes.c_uint64, ctypes.c_uint32,
            ctypes.c_uint64, ctypes.c_uint32,
        ]
        lib.router_mem_create.restype = ctypes.c_void_p

        lib.router_mem_create_ex.argtypes = [
            ctypes.c_uint64, ctypes.c_uint32,
            ctypes.c_uint64, ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        lib.router_mem_create_ex.restype = ctypes.c_void_p

        lib.router_mem_destroy.argtypes = [ctypes.c_void_p]
        lib.router_mem_destroy.restype = None

        lib.router_mem_vram_ptr.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        lib.router_mem_vram_ptr.restype = ctypes.c_void_p

        lib.router_mem_ram_ptr.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        lib.router_mem_ram_ptr.restype = ctypes.c_void_p

        lib.router_mem_stage_host.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32,
            ctypes.c_void_p, ctypes.c_uint64,
        ]
        lib.router_mem_stage_host.restype = ctypes.c_int

        lib.router_mem_h2d_async.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint64,
        ]
        lib.router_mem_h2d_async.restype = ctypes.c_int

        lib.router_mem_d2h_async.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint64,
        ]
        lib.router_mem_d2h_async.restype = ctypes.c_int

        lib.router_mem_sync.argtypes = [ctypes.c_void_p]
        lib.router_mem_sync.restype = ctypes.c_int

        lib.router_mem_stats.argtypes = [ctypes.c_void_p]
        lib.router_mem_stats.restype = RouterMemStats

        lib.router_mem_zero_vram.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64,
        ]
        lib.router_mem_zero_vram.restype = ctypes.c_int

        lib.router_mem_streams.argtypes = [ctypes.c_void_p]
        lib.router_mem_streams.restype = ctypes.c_uint32

    def __enter__(self) -> "NativeMemory":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        if self.handle:
            self.lib.router_mem_destroy(self.handle)
            self.handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def ram_ptr(self, slot: int) -> int:
        ptr = self.lib.router_mem_ram_ptr(self.handle, int(slot))
        if not ptr:
            raise NativeMemoryError(f"invalid RAM slot {slot}")
        return int(ptr)

    def vram_ptr(self, slot: int) -> int:
        ptr = self.lib.router_mem_vram_ptr(self.handle, int(slot))
        if not ptr:
            raise NativeMemoryError(f"invalid VRAM slot {slot}")
        return int(ptr)

    def stage_host(self, ram_slot: int, src: Any, bytes: int) -> None:
        if hasattr(src, "data_ptr"):
            src_ptr = int(src.data_ptr())
        elif isinstance(src, int):
            src_ptr = src
        else:
            src_ptr = ctypes.addressof(ctypes.c_char.from_buffer(src))
        if not self.lib.router_mem_stage_host(
            self.handle, int(ram_slot), src_ptr, int(bytes)
        ):
            raise NativeMemoryError("router_mem_stage_host failed")

    def h2d_async(self, ram_slot: int, vram_slot: int, bytes: int) -> None:
        if not self.lib.router_mem_h2d_async(
            self.handle, int(ram_slot), int(vram_slot), int(bytes)
        ):
            raise NativeMemoryError("router_mem_h2d_async failed")

    def d2h_async(self, vram_slot: int, ram_slot: int, bytes: int) -> None:
        if not self.lib.router_mem_d2h_async(
            self.handle, int(vram_slot), int(ram_slot), int(bytes)
        ):
            raise NativeMemoryError("router_mem_d2h_async failed")

    def zero_vram(self, vram_slot: int, bytes: int | None = None) -> None:
        count = self.vram_slot_bytes if bytes is None else int(bytes)
        if not self.lib.router_mem_zero_vram(
            self.handle, int(vram_slot), count
        ):
            raise NativeMemoryError("router_mem_zero_vram failed")

    def sync(self) -> None:
        if not self.lib.router_mem_sync(self.handle):
            raise NativeMemoryError("cuda stream synchronize failed")

    def stats(self) -> dict[str, int]:
        s = self.lib.router_mem_stats(self.handle)
        return {
            "h2d_calls": int(s.h2d_calls),
            "d2h_calls": int(s.d2h_calls),
            "bytes_h2d": int(s.bytes_h2d),
            "bytes_d2h": int(s.bytes_d2h),
            "sync_calls": int(s.sync_calls),
        }
