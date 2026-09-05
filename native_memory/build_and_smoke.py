from __future__ import annotations

"""Build and smoke-test the native CUDA memory prototype on Windows.

The standalone nvcc path is intentional. CUDA can be installed without the
Visual Studio CUDA MSBuild integration, while nvcc still needs the MSVC
compiler, headers, libraries, and Windows SDK selected by a VS batch file.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native_memory"
SRC = ROOT / "src"
BUILD = NATIVE / "build"


def _find(name: str, extra: Iterable[Path]) -> str | None:
    for candidate in extra:
        if candidate and candidate.is_file():
            return str(candidate.resolve())
    found = shutil.which(name)
    if found:
        return found
    return None


def _run(cmd: Sequence[str], env: dict[str, str]) -> None:
    print("+", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(list(cmd), check=True, env=env)


def _canonicalize_env(source: dict[str, str]) -> dict[str, str]:
    """Remove duplicate case-insensitive environment keys on Windows."""

    values: dict[str, tuple[str, str]] = {}
    for key, value in source.items():
        folded = key.casefold()
        previous = values.get(folded)
        # Prefer the conventional all-uppercase spelling when both PATH and
        # Path (or an equivalent pair) came from a launcher.
        if previous is None or key == key.upper():
            values[folded] = (key, value)
    return {key: value for key, value in values.values()}


def _set_env(env: dict[str, str], name: str, value: str) -> None:
    folded = name.casefold()
    for existing in list(env):
        if existing.casefold() == folded:
            del env[existing]
    env[name] = value


def _get_env(env: dict[str, str], name: str, default: str = "") -> str:
    folded = name.casefold()
    for key, value in env.items():
        if key.casefold() == folded:
            return value
    return default


def _prepend_path(env: dict[str, str], paths: Iterable[Path]) -> None:
    current = _get_env(env, "PATH")
    entries = [str(path) for path in paths if path]
    entries.extend(part for part in current.split(os.pathsep) if part)
    seen: set[str] = set()
    unique: list[str] = []
    for entry in entries:
        key = os.path.normcase(os.path.normpath(entry))
        if key not in seen:
            seen.add(key)
            unique.append(entry)
    _set_env(env, "PATH", os.pathsep.join(unique))


def _tool_version(executable: str, args: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            [executable, *args],
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable:{exc}"
    output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return " | ".join(lines[:6]) if lines else f"unavailable:exit={result.returncode}"


def _vswhere_path() -> str | None:
    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    return _find(
        "vswhere.exe",
        [
            Path(program_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe",
            Path(program_files) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe",
            Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"),
        ],
    )


def _vs_installations() -> tuple[list[Path], str | None]:
    """Discover VS roots with vswhere, then add filesystem fallbacks."""

    vswhere = _vswhere_path()
    installations: list[Path] = []

    def add(path: Path) -> None:
        path = path.resolve()
        if path.is_dir() and path not in installations:
            installations.append(path)

    if vswhere is not None:
        queries = [
            [
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            ["-latest", "-products", "*", "-property", "installationPath"],
        ]
        for query in queries:
            try:
                result = subprocess.run(
                    [vswhere, *query],
                    check=False,
                    capture_output=True,
                    text=True,
                    errors="replace",
                )
            except (OSError, subprocess.SubprocessError):
                continue
            if result.returncode != 0:
                continue
            for line in result.stdout.splitlines():
                value = line.strip().strip('"')
                if value:
                    add(Path(value))
            if installations:
                break

    program_roots: list[Path] = []
    for variable, fallback in (
        ("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        ("ProgramFiles", r"C:\Program Files"),
    ):
        root = Path(os.environ.get(variable, fallback)) / "Microsoft Visual Studio"
        if root not in program_roots:
            program_roots.append(root)

    for root in program_roots:
        for year in ("2022", "2019"):
            year_root = root / year
            if not year_root.is_dir():
                continue
            try:
                products = list(year_root.iterdir())
            except OSError:
                continue
            for product in products:
                if product.is_dir():
                    add(product)

    return installations, vswhere


def _cl_details(cl: Path, env: dict[str, str]) -> tuple[str, str, str]:
    try:
        result = subprocess.run(
            [str(cl)],
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            env=env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable:{exc}", "unknown", str(exc)

    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    # cl.exe localizes this banner. The stable prefix also survives code-page
    # replacement characters, so diagnostics do not depend on UI language.
    version_match = re.search(
        r"Vers[^0-9\r\n]{0,20}([0-9]+(?:\.[0-9]+){2,3})",
        output,
        re.IGNORECASE,
    )
    arch_match = re.search(
        r"\b(?:for|para)\s+(x64|x86|arm64|arm)\b",
        output,
        re.IGNORECASE,
    )
    version = version_match.group(1) if version_match else "unknown"
    arch = arch_match.group(1).lower() if arch_match else "unknown"
    return version, arch, output.strip()


def _batch_command(batch: Path, args: Sequence[str], env: dict[str, str]) -> str:
    """Build a cmd.exe command line without corrupting /c's inner quotes.

    On Windows, passing ["cmd.exe", "/c", 'call "C:\\Program Files\\..."']
    through subprocess makes the CRT escape the inner quotes as backslash-quote.
    cmd.exe does not use backslash as its quote escape. One complete command
    line preserves the quotes around a batch path containing spaces.
    """

    comspec = _get_env(env, "ComSpec") or _get_env(env, "COMSPEC") or "cmd.exe"
    if Path(comspec).is_file():
        comspec = str(Path(comspec).resolve())
    arg_text = " ".join(args)
    command = f'call "{batch}"{(" " + arg_text) if arg_text else ""} && set'
    return f'"{comspec}" /d /s /c {command}'


def _display_batch_output(stdout: str, stderr: str, limit: int = 4000) -> str:
    output = "\n".join(part for part in (stdout, stderr) if part).strip()
    if not output:
        return "<no batch output>"
    if len(output) > limit:
        return "..." + output[-limit:]
    return output


def _msvc_environment(
    env: dict[str, str],
    *,
    nvcc_path: str,
    nvcc_version: str,
) -> dict[str, str]:
    """Load and validate the x64 MSVC environment in a child-process env."""

    base_env = _canonicalize_env(env)
    installations, vswhere = _vs_installations()
    candidates: list[tuple[Path, tuple[str, ...]]] = []

    def add_candidate(batch: Path, args: Sequence[str]) -> None:
        if not batch.is_file():
            return
        key = (str(batch).casefold(), tuple(args))
        if any((str(existing).casefold(), existing_args) == key for existing, existing_args in candidates):
            return
        candidates.append((batch, tuple(args)))

    for installation in installations:
        # vcvars64.bat selects x64 itself and takes no architecture argument.
        add_candidate(installation / "VC" / "Auxiliary" / "Build" / "vcvars64.bat", ())
        # vcvarsall's x64 form selects both host and target x64.
        add_candidate(installation / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat", ("x64",))
        # VsDevCmd uses named architecture switches, not the vcvarsall token.
        add_candidate(
            installation / "Common7" / "Tools" / "VsDevCmd.bat",
            ("-arch=x64", "-host_arch=x64"),
        )

    # Cover standard Build Tools paths even if vswhere is unavailable.
    for installation in (
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools"),
    ):
        add_candidate(installation / "VC" / "Auxiliary" / "Build" / "vcvars64.bat", ())
        add_candidate(installation / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat", ("x64",))
        add_candidate(
            installation / "Common7" / "Tools" / "VsDevCmd.bat",
            ("-arch=x64", "-host_arch=x64"),
        )

    attempts: list[str] = []
    for batch, batch_args in candidates:
        command_line = _batch_command(batch, batch_args, base_env)
        try:
            result = subprocess.run(
                command_line,
                check=False,
                capture_output=True,
                text=True,
                errors="replace",
                env=base_env,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            attempts.append(
                f"script={batch}|args={' '.join(batch_args) or '<none>'}|"
                "returncode=<spawn-failed>|cl=<not found>|cl_version=<unknown>|"
                f"cl_arch=<unknown>|error={exc}"
            )
            continue

        if result.returncode != 0:
            attempts.append(
                f"script={batch}|args={' '.join(batch_args) or '<none>'}|"
                f"returncode={result.returncode}|cl=<not found>|cl_version=<unknown>|"
                "cl_arch=<unknown>|batch_output:\n"
                f"{_display_batch_output(result.stdout, result.stderr)}"
            )
            continue

        merged = _canonicalize_env(base_env)
        for line in result.stdout.splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key:
                _set_env(merged, key, value)

        cl_text = shutil.which("cl.exe", path=_get_env(merged, "PATH"))
        target_arch = _get_env(merged, "VSCMD_ARG_TGT_ARCH", "unknown").lower()
        host_arch = _get_env(merged, "VSCMD_ARG_HOST_ARCH", "unknown").lower()
        missing = [
            name
            for name in ("PATH", "INCLUDE", "LIB", "LIBPATH")
            if not _get_env(merged, name)
        ]

        if cl_text is None:
            attempts.append(
                f"script={batch}|args={' '.join(batch_args) or '<none>'}|"
                "returncode=0|cl=<not found>|cl_version=<unknown>|cl_arch=<unknown>|"
                "batch_output:\n"
                f"{_display_batch_output(result.stdout, result.stderr)}"
            )
            continue

        cl = Path(cl_text).resolve()
        cl_version, cl_arch, cl_output = _cl_details(cl, merged)
        invalid: list[str] = []
        if missing:
            invalid.append("missing=" + ",".join(missing))
        if target_arch not in ("unknown", "x64"):
            invalid.append(f"target_arch={target_arch}")
        if host_arch not in ("unknown", "x64"):
            invalid.append(f"host_arch={host_arch}")
        if cl_arch != "x64":
            invalid.append(f"cl_arch={cl_arch}")

        if invalid:
            attempts.append(
                f"script={batch}|args={' '.join(batch_args) or '<none>'}|cl={cl}|"
                f"cl_version={cl_version}|{'|'.join(invalid)}|cl_output:\n"
                f"{cl_output[-2000:]}"
            )
            continue

        _set_env(merged, "CUDAHOSTCXX", str(cl))
        print(
            f"native_memory_build=msvc:{cl}|env={batch}|version={cl_version}|"
            f"arch=x64|host_arch={host_arch}"
        )
        return merged

    installation_text = ", ".join(str(path) for path in installations) or "<none>"
    attempt_text = "\n\n".join(attempts) if attempts else "<no usable vcvars script found>"
    raise SystemExit(
        "MSVC cl.exe could not be initialized for x64.\n"
        f"vswhere={vswhere or '<not found>'}\n"
        f"visual_studio_installations={installation_text}\n"
        f"nvcc={nvcc_path}\n"
        f"nvcc_version={nvcc_version}\n"
        "architecture=x64\n"
        "MSVC initialization attempts:\n"
        f"{attempt_text}"
    )


def _ninja_candidates(installations: Iterable[Path], cmake: str | None) -> list[Path]:
    candidates = [Path(sys.prefix) / "Scripts" / "ninja.exe"]
    if cmake:
        candidates.append(Path(cmake).resolve().parent / "ninja.exe")
    for installation in installations:
        candidates.append(
            installation
            / "Common7"
            / "IDE"
            / "CommonExtensions"
            / "Microsoft"
            / "CMake"
            / "Ninja"
            / "ninja.exe"
        )
    return candidates


def _find_built_library(build_roots: Iterable[Path]) -> Path | None:
    names = (
        "router_ia_native_memory.dll",
        "librouter_ia_native_memory.so",
        "librouter_ia_native_memory.dylib",
    )
    for root in build_roots:
        for name in names:
            for candidate in (root / name, root / "Release" / name, root / "Debug" / name):
                if candidate.is_file():
                    return candidate.resolve()
    return None


def _build_direct(nvcc: str, cl: Path, env: dict[str, str]) -> Path:
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
            "-cudart=shared",
            "-ccbin",
            str(cl.parent),
            "-Xcompiler=/MD",
            "-o",
            str(dll),
            str(NATIVE / "router_memory.cu"),
        ],
        env,
    )
    if not dll.is_file():
        raise SystemExit(f"nvcc completed without producing the expected DLL: {dll}")
    _set_env(env, "ROUTER_IA_NATIVE_MEMORY_LIB", str(dll.resolve()))
    print(f"native_memory_build=library:{dll.resolve()}")
    return dll.resolve()


def main() -> None:
    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    cmake = _find(
        "cmake.exe",
        [
            Path(program_files) / "CMake" / "bin" / "cmake.exe",
            Path(program_files_x86) / "CMake" / "bin" / "cmake.exe",
            Path(r"C:\Program Files\CMake\bin\cmake.exe"),
            Path(r"C:\Program Files (x86)\CMake\bin\cmake.exe"),
        ],
    )
    cuda_path = os.environ.get("CUDA_PATH")
    nvcc_candidates: list[Path] = []
    if cuda_path:
        nvcc_candidates.append(Path(cuda_path) / "bin" / "nvcc.exe")
    nvcc_candidates.append(
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe")
    )
    nvcc = _find("nvcc.exe", nvcc_candidates)
    if nvcc is None:
        raise SystemExit("nvcc not found. Install the CUDA Toolkit and expose nvcc.exe.")

    nvcc_path = str(Path(nvcc).resolve())
    cuda_root = Path(nvcc_path).parents[1]
    nvcc_version = _tool_version(nvcc_path, ("--version",))
    env = _canonicalize_env(os.environ.copy())
    old_pythonpath = _get_env(env, "PYTHONPATH")
    pythonpath = str(SRC) + (os.pathsep + old_pythonpath if old_pythonpath else "")
    _set_env(env, "PYTHONPATH", pythonpath)
    _set_env(env, "CUDACXX", nvcc_path)
    _set_env(env, "CUDAToolkit_ROOT", str(cuda_root))
    _set_env(env, "CUDA_PATH", str(cuda_root))
    _prepend_path(env, (cuda_root / "bin",))

    installations, _ = _vs_installations()
    ninja = _find("ninja.exe", _ninja_candidates(installations, cmake))

    print(f"native_memory_build=nvcc:{nvcc_path}")
    print(f"native_memory_build=nvcc_version:{nvcc_version}")
    print(f"native_memory_build=cuda_root:{cuda_root}")
    print("native_memory_build=architecture:x64")

    if cmake is None:
        print("native_memory_build=generator:nvcc-direct|reason=cmake-not-found")
    elif ninja is None:
        print("native_memory_build=generator:nvcc-direct|reason=VS-CUDA-toolset-missing")

    env = _msvc_environment(
        env,
        nvcc_path=nvcc_path,
        nvcc_version=nvcc_version,
    )
    cl_text = shutil.which("cl.exe", path=_get_env(env, "PATH"))
    if cl_text is None:
        raise SystemExit("MSVC environment validation succeeded but cl.exe disappeared from PATH")
    cl = Path(cl_text).resolve()
    _prepend_path(env, (cuda_root / "bin",))
    if ninja is not None:
        # The executable may have been found by an explicit path rather than
        # the inherited PATH; make CMake and its build subprocess see it.
        _prepend_path(env, (Path(ninja).resolve().parent,))

    library: Path | None = None
    if cmake is not None and ninja is not None:
        cmake_build = NATIVE / "build-ninja"
        print(f"native_memory_build=generator:Ninja|ninja:{ninja}")
        try:
            _run(
                [
                    cmake,
                    "-S",
                    str(NATIVE),
                    "-B",
                    str(cmake_build),
                    "-G",
                    "Ninja",
                    f"-DCMAKE_MAKE_PROGRAM={ninja}",
                    "-DCMAKE_BUILD_TYPE=Release",
                    f"-DCMAKE_CUDA_COMPILER={nvcc_path}",
                    f"-DCMAKE_CUDA_HOST_COMPILER={cl}",
                    f"-DCUDAToolkit_ROOT={cuda_root}",
                ],
                env,
            )
            _run([cmake, "--build", str(cmake_build), "--config", "Release"], env)
            library = _find_built_library((cmake_build,))
            if library is None:
                raise RuntimeError(f"CMake completed without producing a native library in {cmake_build}")
            _set_env(env, "ROUTER_IA_NATIVE_MEMORY_LIB", str(library))
            print(f"native_memory_build=library:{library}")
        except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
            print(f"native_memory_build=cmake-failed:{type(exc).__name__}:{exc}")
            print("native_memory_build=generator:nvcc-direct|reason=cmake-failed")

    if library is None:
        library = _build_direct(nvcc_path, cl, env)

    _run([sys.executable, str(NATIVE / "qwen36_native_memory_smoke.py")], env)


if __name__ == "__main__":
    main()
