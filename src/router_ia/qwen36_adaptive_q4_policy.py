from __future__ import annotations

"""Adaptive FP8 eviction policy for Qwen3.6 expert cache.

An FP8 victim is copied to the Q4 RAM tier only when its adaptive score is
high enough to justify retaining it. Otherwise the victim is dropped and will
be fetched from the safetensor shard again if it becomes necessary.

This module only changes cache policy; model math is untouched.
"""

from collections import OrderedDict
import os

from . import qwen36_adaptive_experts as adaptive
from . import qwen36_expert_cache as expert_cache_module
from .qwen36_expert_cache import RoutedExpertCache

Q4_RETENTION_SCORE = float(os.getenv("QWEN36_EXPERT_Q4_RETENTION_SCORE", "4.0"))


def _adaptive_insert(self: RoutedExpertCache, layer: int, expert_id: int, entry) -> None:
    layer = int(layer)
    expert_id = int(expert_id)
    policy = None
    for root, candidate in adaptive._POLICIES.items():
        try:
            if adaptive.official._EXPERT_CACHES.get(root) is self:
                policy = candidate
                break
        except Exception:
            continue

    bank = self.fp8_entries.setdefault(layer, OrderedDict())
    old_ids = set(bank)

    if policy is not None and bank:
        ordered = sorted(
            bank.items(),
            key=lambda item: policy.score(layer, int(item[0])),
        )
        bank.clear()
        bank.update(ordered)

    if expert_id in bank:
        self._erase(layer, expert_id, "fp8")
        bank.pop(expert_id, None)
    bank[expert_id] = entry
    self._record(layer, expert_id, "fp8", self._fp8_size(entry))
    bank.move_to_end(expert_id)

    while len(bank) > self.fp8_slots:
        victim_id, victim = bank.popitem(last=False)
        self._erase(layer, victim_id, "fp8")

        retain_q4 = policy is not None and policy.score(layer, int(victim_id)) >= Q4_RETENTION_SCORE
        if self.q4_slots > 0 and retain_q4:
            cold_gpu = expert_cache_module._q4_quantize_entry_from_fp8(victim)
            cold = expert_cache_module._move_q4_to_cpu(cold_gpu)
            q4 = self.q4_entries.setdefault(layer, OrderedDict())
            old_q4 = q4.pop(victim_id, None)
            if old_q4 is not None:
                self._erase(layer, victim_id, "q4")
            q4[victim_id] = cold
            self._record(layer, victim_id, "q4", self._q4_size(cold))
            q4.move_to_end(victim_id)
            self.fp8_to_q4 += 1

            while len(q4) > self.q4_slots:
                dropped_id, _ = q4.popitem(last=False)
                self._erase(layer, dropped_id, "q4")
                self.q4_drops += 1
                self.q4_ram_evictions += 1
        else:
            self.evictions += 1

    if policy is not None:
        new_ids = set(bank)
        for victim in old_ids - new_ids:
            if victim != expert_id:
                policy.note_demotion(layer, int(victim))


RoutedExpertCache._insert_fp8_locked = _adaptive_insert

# Repeated Q4 reuse is enough to justify promotion. Comparing against lifetime
# FP8 hits made previously-hot experts effectively impossible to promote back.
def _should_promote(self, layer: int, expert_id: int) -> bool:
    with self.lock:
        item = self._entry(layer, expert_id)
        return item.q4_hits >= adaptive.PROMOTION_HITS


adaptive.AdaptiveExpertPolicy.should_promote = _should_promote
adaptive.Q4_RETENTION_SCORE = Q4_RETENTION_SCORE

print(f"adaptive_q4_policy=score-retention|threshold={Q4_RETENTION_SCORE:.2f}|cold=drop|warm=Q4")
