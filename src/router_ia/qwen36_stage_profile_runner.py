from __future__ import annotations

"""Standalone diagnostic entrypoint for detailed Qwen3.6 stage timings.

Usage:
    python -m router-ia.src.router_ia.qwen36_stage_profile_runner D:\\router\\ia --device cuda --prompt "Python é" --max-new-tokens 8

This launcher enables the stage profiler before importing the canonical runner,
so the profiler's import-time hooks are active for the real generation path.
The normal runner is not modified by this file.
"""

import os

# qwen36_stage_profiler reads this flag at import time.
os.environ["QWEN36_STAGE_PROFILE"] = "1"

from . import qwen36_chat_batch_runner as canonical_runner
from . import qwen36_stage_profiler as _stage_profiler  # noqa: F401,E402


if __name__ == "__main__":
    canonical_runner.chat.main()
