from __future__ import annotations

"""Windows-friendly helper for building and smoke-testing the CUDA memory prototype."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native_memory"
SRC = ROOT / "src"


def _find(name: str, extra: list[Path]) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for candidate in extra:
        if candidate.is_file():
            return str(candidate)
    return None


def _run(cmd: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)

cmake = _find("cmake", [Path(r"C:\Program Files\CMake\bin\cmake.exe"), Path(r"C:\Program Files (x86)\CMake\bin\cmake.exe")])
nvcc = _find("nvcc", [Path(os.environ.get("CUDA_PATH", "")) / "bin" / "nvcc.exe", Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe")])
ninja = _find("ninja", [Path(sys.prefix) / "Scripts" / "ninja.exe"])
if cmake is None:
    raise SystemExit("CMake not found. Install CMake and add it to PATH.")
if nvcc is None:
    raise SystemExit("nvcc not found. Install the CUDA Toolkit and expose nvcc.exe.")
cuda_root = Path(nvcc).resolve().parents[1]
env = os.environ.copy()
env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
env["CUDACXX"] = str(Path(nvcc).resolve())
env["CUDAToolkit_ROOT"] = str(cuda_root)
print(f"native_memory_build=cmake:{cmake}")
print(f"native_memory_build=nvcc:{nvcc}")
print(f"native_memory_build=cuda_root:{cuda_root}")
if ninja is not None:
    build = NATIVE / "build-ninja"
    print(f"native_memory_build=generator:Ninja|ninja:{ninja}")
    _run([cmake, "-S", str(NATIVE), "-B", str(build), "-G", "Ninja"], env)
else:
    build = NATIVE / "build"
    print("native_memory_build=generator:Visual Studio 17 2022|toolset:cuda=12.4")
    _run([cmake, "-S", str(NATIVE), "-B", str(build), "-G", "Visual Studio 17 2022", "-A", "x64", "-T", "cuda=12.4"], env)
_run([cmake, "--build", str(build), "--config", "Release"], env)
_run([sys.executable, str(NATIVE / "qwen36_native_memory_smoke.py")], env)
