from __future__ import annotations

"""Canonical entrypoint for the optimized Qwen3.6 stateful chat runner."""

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
from . import qwen36_async_current_route as _async_current_route  # noqa: F401
from . import qwen36_generation_heat as _generation_heat  # noqa: F401
from . import qwen36_adaptive_expert_ram as _adaptive_expert_ram  # noqa: F401
from . import qwen36_gpu_q4 as _gpu_q4  # noqa: F401
from . import qwen36_fp16_expert_cache as _fp16_expert_cache  # noqa: F401
from . import qwen36_expert_batch_plan_v2 as _expert_batch_plan_v2  # noqa: F401
from . import qwen36_expert_batch_plan_v3 as _expert_batch_plan_v3  # noqa: F401
from . import qwen36_expert_q4_hierarchy_fixed as _expert_q4_hierarchy  # noqa: F401
from . import qwen36_expert_q4_activate as _expert_q4_activate  # noqa: F401
from . import qwen36_physical_memory_guard_v3 as _physical_memory_guard  # noqa: F401
from . import qwen36_memory_policy_v2 as _memory_policy_v2  # noqa: F401

if os.getenv("QWEN36_PROFILE", "0").strip().lower() in {"1", "true", "yes", "on"}:
    from . import qwen36_profiler as _profiler  # noqa: F401


if __name__ == "__main__":
    chat.main()
