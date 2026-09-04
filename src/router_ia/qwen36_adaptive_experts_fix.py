from __future__ import annotations

"""Small runtime fix for the adaptive expert policy."""

import math

from . import qwen36_adaptive_experts as adaptive


def _score_locked(self, layer: int, expert_id: int) -> float:
    item = self._entry(layer, expert_id)
    age = max(self.tick - item.last_tick, 0)
    recency = 1.0 / (1.0 + age / float(adaptive.RECENCY_WINDOW))
    frequency = math.log1p(item.accesses)
    return 1.5 * frequency + 2.0 * recency


def _snapshot(self, cache=None):
    with self.lock:
        scored = [
            (self._score_locked(layer, expert_id), layer, expert_id, item)
            for (layer, expert_id), item in self.usage.items()
            if item.accesses
        ]
        scored.sort(reverse=True, key=lambda item: item[0])
        hot = sum(score >= adaptive.PREFETCH_SCORE for score, *_ in scored)
        return {
            "promotion_hits": adaptive.PROMOTION_HITS,
            "recency_window": adaptive.RECENCY_WINDOW,
            "tracked": len(scored),
            "hot": hot,
            "fp8_hits": self.fp8_hits,
            "q4_hits": self.q4_hits,
            "cold_misses": self.cold_misses,
            "promotions": self.promotions,
            "demotions": self.demotions,
            "prefetch_requests": self.prefetch_requests,
            "prefetch_selected": self.prefetch_selected,
            "prefetch_skipped": self.prefetch_skipped,
            "top": [
                {
                    "layer": layer,
                    "expert": expert_id,
                    "score": round(score, 3),
                    "accesses": item.accesses,
                    "fp8_hits": item.fp8_hits,
                    "q4_hits": item.q4_hits,
                    "tier": self._tier(cache, layer, expert_id),
                }
                for score, layer, expert_id, item in scored[: adaptive.TOP_LOG_ENTRIES]
            ],
        }


adaptive.AdaptiveExpertPolicy.snapshot = _snapshot
