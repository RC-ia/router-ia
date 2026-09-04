from __future__ import annotations

"""Activate the fixed Q4 expert hierarchy on the actual v2 hot path."""

from . import qwen36_expert_batch_plan_v2 as planner_v2
from . import qwen36_expert_q4_hierarchy_fixed as hierarchy

planner_v2._plan_layer = hierarchy._plan_layer_q4

print("expert_q4_hotpath=enabled|planner=v2-plan-layer|storage=VRAM-RAM-SSD")
