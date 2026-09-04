from __future__ import annotations

"""Activate the fixed Q4 expert hierarchy on the actual v2 hot path."""

from . import qwen36_async_scheduler as async_scheduler
from . import qwen36_expert_batch_plan_v2 as planner_v2
from . import qwen36_expert_q4_hierarchy_fixed as hierarchy
from . import qwen36_chat_batch as chat

planner_v2._plan_layer = hierarchy._plan_layer_q4

# Q4 hierarchy owns expert residency. The legacy async scheduler may continue
# to exist for diagnostics/adaptive prediction, but it must not materialize
# speculative experts because every prefetch is a real VRAM allocation and
# creates FP8->Q4 conversion churn on the host.
chat._router_q4_hierarchy_active = True
async_scheduler.ENABLED = False

print("expert_q4_hotpath=enabled|planner=v2-plan-layer|storage=VRAM-RAM-SSD")
print("expert_q4_memory_owner=hierarchy|async_materialization=disabled|adaptive_prediction=retained")
