from __future__ import annotations

"""Standalone diagnostic entrypoint for detailed Qwen3.6 stage timings.

Usage:
    python -m router-ia.src.router_ia.qwen36_stage_profile_runner D:\\router\\ia --device cuda --prompt "Python é" --max-new-tokens 8

Unlike the normal runner, this launcher enables only the stage profiler. It
imports the canonical runner first so every normal optimization is installed,
then installs the diagnostic timing hooks, and finally calls the same chat
entrypoint.
"""

import sys

from . import qwen36_chat_batch_runner as canonical_runner
from . import qwen36_stage_profiler as _stage_profiler  # noqa: F401


if __name__ == "__main__":
    canonical_runner.chat.main()
