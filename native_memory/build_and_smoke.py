from __future__ import annotations

"""Windows-friendly helper: configure/build the CUDA memory prototype, then run its smoke test."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "native_memory" / "build"
SRC = ROOT / "src"


def _find_executable(name: str, windows_candidates: tuple[Path, ...] = ()) -> str:
    found = shutil.which(name)
    if found:
        return found
    for candidate in windows_candidates:
        if candidate.is_file():
            return str(candidate)
    raise SystemExit(
        f"{name} não foi encontrado. Instale-o e coloque-o no PATH, "
        f"ou use o caminho completo. Procurado também em: "
        + ", ".join(str(p) for p in windows_candidates)
    )


cmake = _find_executable(
    "cmake",
    (
        Path(os.environ.get("ProgramFiles", r"C:\\Program Files")) / "CMake" / "bin" / "cmake.exe",
        Path(os.environ.get("ProgramFiles(x86)", r"C:\\Program Files (x86)")) / "CMake" / "bin" / "cmake.exe",
    ),
)

nvcc = shutil.which("nvcc")
if not nvcc:
    cuda_root = os.environ.get("CUDA_PATH")
    if cuda_root:
        candidate = Path(cuda_root) / "bin" / "nvcc.exe"
        if candidate.is_file():
            nvcc = str(candidate)
if not nvcc:
    raise SystemExit(
        "nvcc não foi encontrado. O protótipo precisa do CUDA Toolkit "
        "com nvcc disponível no PATH ou em CUDA_PATH."
    )

print(f"native_memory_build=cmake:{cmake}")
print(f"native_memory_build=nvcc:{nvcc}")

env = os.environ.copy()
env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")

def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, env=env)
    except FileNotFoundError as exc:
        raise SystemExit(f"Executável não encontrado durante o build: {cmd[0]}") from exc


run([cmake, "-S", str(ROOT / "native_memory"), "-B", str(BUILD)])
run([cmake, "--build", str(BUILD), "--config", "Release"])
run([sys.executable, str(ROOT / "native_memory" / "qwen36_native_memory_smoke.py")])
