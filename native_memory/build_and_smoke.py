from __future__ import annotations

"""Windows-friendly builder for the native CUDA memory prototype.

CMake is still used when Ninja is available. On Visual Studio installations
without the CUDA MSBuild integration (CUDA*.props/targets), fall back to a
standalone nvcc build. The standalone path initializes the MSVC developer
environment automatically so nvcc can find cl.exe, INCLUDE and LIB.
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
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip().splitlines()
    return Path(value[-1]).resolve() if value else None


def _load_msvc_environment(env: dict[str, str]) -> dict[str, str]:
    """Load the x64 MSVC developer environment into a child-process env."""
    if shutil.which("cl.exe"):
        print("native_memory_build=msvc:cl-found-in-PATH")
        return env

    candidates: list[Path] = []
    installation = _vs_installation()
    if installation is not None:
        candidates.extend(
            [
                installation / "Common7" / "Tools" / "VsDevCmd.bat",
                installation / "VC" / "Auxiliary" / "Build" / "vcvars64.bat",
            ]
        )
    candidates.extend(
        [
            Path(r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"),
            Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"),
        ]
    )
    devcmd = next((path for path in candidates if path.is_file()), None)
    if devcmd is None:
        raise SystemExit(
            "MSVC cl.exe not found. Install Visual Studio Build Tools with "
            "Desktop development with C++ / MSVC C++ x64/x86 build tools."
        )

    command = f'call "{devcmd}" -arch=x64 -host_arch=x64 && set'
    try:
        result = subprocess.run(
            ["cmd.exe", "/d", "/s", "/c", command],
            check=True,
            capture_output=True,
            text=True,
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Failed to initialize MSVC environment via {devcmd}: {exc}") from exc

    merged = dict(env)
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key:
            merged[key] = value

    cl = shutil.which("cl.exe", path=merged.get("PATH"))
    if cl is None:
        raise SystemExit(
            f"MSVC developer environment loaded from {devcmd}, but cl.exe is still not visible."
        )
    print(f"native_memory_build=msvc:{cl}")
    return merged


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

    # nvcc uses cl.exe as its Windows host compiler. Initialize the x64 MSVC
    # environment first instead of assuming the caller opened a VS Developer
    # Command Prompt.
    env = _load_msvc_environment(env)

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
