from __future__ import annotations

"""Windows-friendly builder for the native CUDA memory prototype."""

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


def _vs_installation() -> Path | None:
    vswhere = _find(
        "vswhere",
        [Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")],
    )
    if vswhere is None:
        return None
    try:
        result = subprocess.run(
            [
                vswhere,
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            check=True,
            capture_output=True,
            text=True,
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return Path(lines[-1]).resolve() if lines else None


def _msvc_environment(env: dict[str, str]) -> dict[str, str]:
    """Load x64 MSVC variables using the most direct batch file available."""
    if shutil.which("cl.exe"):
        print("native_memory_build=msvc:cl-found-in-PATH")
        return env

    installation = _vs_installation()
    candidates: list[Path] = []
    if installation is not None:
        # Prefer vcvars64.bat: it is narrower and does not depend on optional
        # VS workload components beyond the MSVC build tools themselves.
        candidates.append(installation / "VC" / "Auxiliary" / "Build" / "vcvars64.bat")
        candidates.append(installation / "Common7" / "Tools" / "VsDevCmd.bat")

    candidates.extend(
        [
            Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"),
            Path(r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"),
            Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"),
            Path(r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"),
        ]
    )

    # Deduplicate while preserving preference order.
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path).lower()
        if path.is_file() and key not in seen:
            seen.add(key)
            unique.append(path)

    last_error: Exception | None = None
    for batch in unique:
        command = f'call "{batch}" amd64 && set'
        try:
            result = subprocess.run(
                ["cmd.exe", "/d", "/s", "/c", command],
                check=True,
                capture_output=True,
                text=True,
                errors="replace",
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            last_error = exc
            continue

        merged = dict(env)
        for line in result.stdout.splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key:
                merged[key] = value

        cl = shutil.which("cl.exe", path=merged.get("PATH"))
        if cl is not None:
            print(f"native_memory_build=msvc:{cl}|env={batch}")
            return merged

    detail = f" Last error: {last_error}" if last_error else ""
    raise SystemExit(
        "MSVC cl.exe could not be initialized. Install Visual Studio Build Tools "
        "with Desktop development with C++ / MSVC x64 tools." + detail
    )


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

if ninja is not None and cmake is not None:
    build = NATIVE / "build-ninja"
    print(f"native_memory_build=generator:Ninja|ninja:{ninja}")
    env = _msvc_environment(env)
    _run([cmake, "-S", str(NATIVE), "-B", str(build), "-G", "Ninja"], env)
    _run([cmake, "--build", str(build), "--config", "Release"], env)
else:
    if cmake is not None:
        print("native_memory_build=generator:nvcc-direct|reason=VS-CUDA-toolset-missing")
    else:
        print("native_memory_build=generator:nvcc-direct|reason=cmake-not-found")

    env = _msvc_environment(env)
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
