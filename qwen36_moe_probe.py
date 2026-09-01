from __future__ import annotations

"""Compatibility launcher for the packaged Qwen3.6 MoE probe.

This file exists so ``python qwen36_moe_probe.py ...`` works from a source
checkout as well as ``python -m router_ia.qwen36_moe_probe ...``.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from router_ia.qwen36_moe_probe import main


if __name__ == "__main__":
    main()
