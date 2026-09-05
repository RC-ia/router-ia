from __future__ import annotations

"""Windows-friendly helper: configure/build the CUDA memory prototype, then run its smoke test."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "native_memory" / "build"
SRC = ROOT / "src"


env = os.environ.copy()
env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")

subprocess.run(
    ["cmake", "-S", str(ROOT / "native_memory"), "-B", str(BUILD)],
    check=True,
env=env,
)
subprocess.run(
    ["cmake", "--build", str(BUILD), "--config", "Release"],
    check=True,
env=env,
)
subprocess.run(
    [sys.executable, str(ROOT / "native_memory" / "qwen36_native_memory_smoke.py")],
    check=True,
env=env,
)
