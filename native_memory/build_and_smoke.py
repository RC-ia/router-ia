from __future__ import annotations

"""Windows-friendly builder for the native CUDA memory prototype.

CMake is still used when Ninja is available. On Visual Studio installations
without the CUDA MSBuild integration (CUDA*.props/targets), fall back to a
standalone nvcc build so the prototype does not require that integration.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native_memory"
SRC = ROOT / "src"
BUILD = NATIVE / "build"


def _find(name: str, extra: list[Path]) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for candidate in extra:
        if candidate.is_file():
            return str(candidate)
    return None


def _run(cmd: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


cmake = _find(
    "cmake",
    [
        Path(r"C:\Program Files\CMake\bin\cmake.exe"),
        Path(r"C:\Program Files (x86)\CMake\bin\cmake.exe"),
    ],
)
nvcc = _find(
    "nvcc",
    [
        Path(os.environ.get("CUDA_PATH", "")) / "bin" / "nvcc.exe",
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe"),
    ],
)
ninja = _find("ninja", [Path(sys.prefix) / "Scripts" / "ninja.exe"])

if nvcc is None:
    raise SystemExit("nvcc not found. Install the CUDA Toolkit and expose nvcc.exe.")

cuda_root = Path(nvcc).resolve().parents[1]
env = os.environ.copy()
env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
env["CUDACXX"] = str(Path(nvcc).resolve())
env["CUDAToolkit_ROOT"] = str(cuda_root)

print(f"native_memory_build=nvcc:{nvcc}")
print(f"native_memory_build=cuda_root:{cuda_root}")

# Prefer Ninja/CMake when it is actually available: it does not need the
# Visual Studio CUDA MSBuild integration. Otherwise compile the single CUDA
# translation unit directly with nvcc. This is enough for this prototype and
# avoids requiring CUDA 12.4 BuildCustomizations inside VS Build Tools.
if ninja is not None and cmake is not None:
    build = NATIVE / "build-ninja"
    print(f"native_memory_build=generator:Ninja|ninja:{ninja}")
    _run([cmake, "-S", str(NATIVE), "-B", str(build), "-G", "Ninja"], env)
    _run([cmake, "--build", str(build), "--config", "Release"], env)
else:
    if cmake is not None:
        print("native_memory_build=generator:nvcc-direct|reason=VS-CUDA-toolset-missing")
    else:
        print("native_memory_build=generator:nvcc-direct|reason=cmake-not-found")

    # Build a DLL directly. CUDA 12.4 can invoke the installed MSVC host
    # compiler; -allow-unsupported-compiler avoids nvcc rejecting newer VS
    # minor releases even though the generated code is still ordinary CUDA.
    output_dir = BUILD / "Release"
    output_dir.mkdir(parents=True, exist_ok=True)
    dll = output_dir / "router_ia_native_memory.dll"
    _run(
        [
            nvcc,
            "-shared",
            "-O3",
            "-std=c++17",
            "-allow-unsupported-compiler",
            "-Xcompiler=/MD",
            f"-o={dll}",
            str(NATIVE / "router_memory.cu"),
        ],
        env,
    )

_run([sys.executable, str(NATIVE / "qwen36_native_memory_smoke.py")], env)
