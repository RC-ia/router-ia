# Native CUDA memory prototype

This is the first native-memory prototype for `router-ia`.

It does **not** replace the Python router yet. It provides a fixed-slot CUDA
VRAM pool plus a pinned host-RAM pool and one non-blocking CUDA stream.

## Build

For the complete Windows build plus CUDA smoke test, run this from the
workspace parent (`D:\router`):

```bat
python router-ia\native_memory\build_and_smoke.py
```

The helper discovers the x64 MSVC environment with Visual Studio's developer
scripts, then uses CMake/Ninja when available. If CMake cannot initialize the
CUDA toolset or the configure/build fails, it automatically compiles the
translation unit directly with `nvcc` and the same MSVC environment.

From the repository root, with CMake and the CUDA toolkit installed:

```bash
cmake -S native_memory -B native_memory/build
cmake --build native_memory/build --config Release
```

On Windows the DLL is normally produced under:

```text
native_memory/build/Release/router_ia_native_memory.dll
```

The Python wrapper searches the common build locations automatically. To use
another location, set `ROUTER_IA_NATIVE_MEMORY_LIB` to the full library path.

## Smoke test

From the repository root:

```bash
python native_memory/qwen36_native_memory_smoke.py
```

Expected result:

```text
native_memory=PASS|...
```

## Current architecture

```text
Python router
    |
    v
NativeMemory
    +-- fixed VRAM slots (cudaMalloc)
    +-- fixed pinned RAM slots (cudaHostAlloc)
    +-- one non-blocking CUDA stream
    +-- async H2D / D2H
```

The next integration step is to let the Q4 hierarchy assign
`(layer, expert) -> native VRAM slot`, while keeping SSD/Q4 metadata in Python.
