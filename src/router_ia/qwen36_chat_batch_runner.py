from __future__ import annotations

"""Canonical entrypoint for the optimized Qwen3.6 stateful chat runner.

The target chat module is imported first and only then patched. This avoids
runpy's duplicate-module warning when using ``python -m`` and ensures the
expert cache, adaptive expert policy, asynchronous lookahead, stateful
attention hooks, expert-tier routing, adaptive Q4 retention, VRAM governor,
prompt-scoped generation heat, and optional profiler patch the exact module
instance executed by ``main()``.
"""

import os

from . import qwen36_chat_batch as chat
from . import runtime_optimizations as _runtime_optimizations  # noqa: F401
from . import qwen36_official_optimizations as _official_optimizations  # noqa: F401
from . import qwen36_vram_governor as _vram_governor  # noqa: F401
from . import qwen36_adaptive_experts as _adaptive_experts  # noqa: F401
from . import qwen36_adaptive_experts_fix as _adaptive_experts_fix  # noqa: F401
from . import qwen36_expert_tier_policy as _expert_tier_policy  # noqa: F401
from . import qwen36_adaptive_q4_policy as _adaptive_q4_policy  # noqa: F401
from . import qwen36_async_scheduler as _async_scheduler  # noqa: F401
from . import qwen36_generation_heat as _generation_heat  # noqa: F401

if os.getenv("QWEN36_PROFILE", "0").strip().lower() in {"1", "true", "yes", "on"}:
    from . import qwen36_profiler as _profiler  # noqa: F401


if __name__ == "__main__":
    chat.main()
