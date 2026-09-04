from __future__ import annotations

"""Adaptive hot/warm expert policy for the official Qwen3.6 runner.

This layer changes only expert-cache policy. Model math and tensor values are
left untouched. FP8 entries with high recent usage are retained preferentially;
Q4 RAM entries are promoted back to FP8 only after repeated accesses.
"""

import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from . import qwen36_chat_batch as chat
from . import qwen36_official_optimizations as official
from .qwen36_expert_cache import RoutedExpertCache

PROMOTION_HITS = max(int(__import__("os").getenv("QWEN36_EXPERT_PROMOTION_HITS", "3")), 2)
RECENCY_WINDOW = max(int(__import__("os").getenv("QWEN36_EXPERT_RECENCY_WINDOW", "16")), 1)
PREFETCH_SCORE = float(__import__("os").getenv("QWEN36_EXPERT_PREFETCH_SCORE", "2.5"))
TOP_LOG_ENTRIES = max(int(__import__("os").getenv("QWEN36_EXPERT_TOP_LOG", "8")), 1)


@dataclass
class ExpertUsage:
    accesses: int = 0
    fp8_hits: int = 0
    q4_hits: int = 0
    misses: int = 0
    last_tick: int = 0
    promotions: int = 0
    demotions: int = 0


class AdaptiveExpertPolicy:
    def __init__(self) -> None:
        self.lock = Lock()
        self.tick = 0
        self.usage: dict[tuple[int, int], ExpertUsage] = {}
        self.promotions = 0
        self.demotions = 0
        self.prefetch_requests = 0
        self.prefetch_skipped = 0
        self.prefetch_selected = 0
        self.cold_misses = 0
        self.q4_hits = 0
        self.fp8_hits = 0

    def _entry(self, layer: int, expert_id: int) -> ExpertUsage:
        key = (int(layer), int(expert_id))
        item = self.usage.get(key)
        if item is None:
            item = ExpertUsage()
            self.usage[key] = item
        return item

    def record(self, layer: int, expert_id: int, tier: str) -> ExpertUsage:
        with self.lock:
            self.tick += 1
            item = self._entry(layer, expert_id)
            item.accesses += 1
            item.last_tick = self.tick
            if tier == "fp8":
                item.fp8_hits += 1
                self.fp8_hits += 1
            elif tier == "q4":
                item.q4_hits += 1
                self.q4_hits += 1
            else:
                item.misses += 1
                self.cold_misses += 1
            return item

    def score(self, layer: int, expert_id: int) -> float:
        with self.lock:
            item = self._entry(layer, expert_id)
            age = max(self.tick - item.last_tick, 0)
            recency = 1.0 / (1.0 + age / float(RECENCY_WINDOW))
            frequency = math.log1p(item.accesses)
            return 1.5 * frequency + 2.0 * recency

    def should_promote(self, layer: int, expert_id: int) -> bool:
        with self.lock:
            item = self._entry(layer, expert_id)
            return item.q4_hits >= PROMOTION_HITS and item.q4_hits >= item.fp8_hits

    def note_promotion(self, layer: int, expert_id: int) -> None:
        with self.lock:
            item = self._entry(layer, expert_id)
            item.promotions += 1
            self.promotions += 1

    def note_demotion(self, layer: int, expert_id: int) -> None:
        with self.lock:
            item = self._entry(layer, expert_id)
            item.demotions += 1
            self.demotions += 1

    def snapshot(self, cache: RoutedExpertCache | None = None) -> dict[str, Any]:
        with self.lock:
            scored = [
                (self.score(layer, expert_id), layer, expert_id, item)
                for (layer, expert_id), item in self.usage.items()
                if item.accesses
            ]
            scored.sort(reverse=True, key=lambda item: item[0])
            hot = sum(score >= PREFETCH_SCORE for score, *_ in scored)
            payload: dict[str, Any] = {
                "promotion_hits": PROMOTION_HITS,
                "recency_window": RECENCY_WINDOW,
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
                    for score, layer, expert_id, item in scored[:TOP_LOG_ENTRIES]
                ],
            }
            return payload

    @staticmethod
    def _tier(cache: RoutedExpertCache | None, layer: int, expert_id: int) -> str:
        if cache is None:
            return "unknown"
        try:
            if expert_id in cache.fp8_entries.get(layer, {}):
                return "FP8"
            if expert_id in cache.q4_entries.get(layer, {}):
                return "Q4"
        except Exception:
            return "unknown"
        return "none"


_POLICIES: dict[Path, AdaptiveExpertPolicy] = {}
_POLICY_LOCK = Lock()
_ORIGINAL_TRIPLET = chat._expert_projection_triplet
_ORIGINAL_WARM = chat._warm_expert_raw_cache
_ORIGINAL_INSERT = RoutedExpertCache._insert_fp8_locked
_ORIGINAL_CACHE_STATS = chat.cache_stats
_ORIGINAL_PRINT_CACHE = chat.print_cache


def _policy(root: Path) -> AdaptiveExpertPolicy:
    key = root.resolve()
    with _POLICY_LOCK:
        value = _POLICIES.get(key)
        if value is None:
            value = AdaptiveExpertPolicy()
            _POLICIES[key] = value
        return value


def _layer_from_prefix(layer_prefix: str) -> int | None:
    marker = ".layers."
    if marker not in layer_prefix:
        return None
    try:
        return int(layer_prefix.split(marker, 1)[1].split(".", 1)[0])
    except (ValueError, IndexError):
        return None


def _adaptive_insert(self: RoutedExpertCache, layer: int, expert_id: int, entry) -> None:
    policy = None
    for root, candidate in _POLICIES.items():
        try:
            if official._EXPERT_CACHES.get(root) is self:
                policy = candidate
                break
        except Exception:
            continue

    bank = self.fp8_entries.setdefault(int(layer), OrderedDict())
    old_ids = set(bank)
    if policy is not None and bank:
        ordered = sorted(
            bank.items(),
            key=lambda item: policy.score(int(layer), int(item[0])),
        )
        bank.clear()
        bank.update(ordered)

    _ORIGINAL_INSERT(self, int(layer), int(expert_id), entry)

    if policy is not None:
        new_ids = set(bank)
        for victim in old_ids - new_ids:
            if victim != int(expert_id):
                policy.note_demotion(int(layer), int(victim))


def _adaptive_triplet(root: Path, layer_prefix: str, expert_id: int, device: str):
    if device != "cuda":
        return _ORIGINAL_TRIPLET(root, layer_prefix, expert_id, device)

    layer = _layer_from_prefix(layer_prefix)
    if layer is None:
        return _ORIGINAL_TRIPLET(root, layer_prefix, expert_id, device)

    expert_cache = official._expert_cache(root)
    expert_id = int(expert_id)
    with expert_cache.lock:
        if expert_id in expert_cache.fp8_entries.get(layer, {}):
            tier = "fp8"
        elif expert_id in expert_cache.q4_entries.get(layer, {}):
            tier = "q4"
        else:
            tier = "miss"

    result = _ORIGINAL_TRIPLET(root, layer_prefix, expert_id, device)
    policy = _policy(root)
    usage = policy.record(layer, expert_id, tier)

    if tier == "q4" and policy.should_promote(layer, expert_id):
        with expert_cache.lock:
            already_hot = expert_id in expert_cache.fp8_entries.get(layer, {})
        if not already_hot:
            expert_cache.put_fp16(layer, expert_id, result)
            policy.note_promotion(layer, expert_id)

    return result


def _adaptive_warm(root: Path, layer_prefix: str, expert_ids: list[int]) -> None:
    """Only prefetch routed experts that already look hot; cold requests load normally."""
    if not expert_ids:
        return

    layer = _layer_from_prefix(layer_prefix)
    if layer is None:
        return _ORIGINAL_WARM(root, layer_prefix, expert_ids)

    policy = _policy(root)
    expert_cache = official._expert_cache(root)
    selected: list[int] = []

    for expert_id in dict.fromkeys(int(v) for v in expert_ids):
        policy.prefetch_requests += 1
        score = policy.score(layer, expert_id)
        with expert_cache.lock:
            in_fp8 = expert_id in expert_cache.fp8_entries.get(layer, {})
        if in_fp8:
            policy.prefetch_skipped += 1
            continue
        if score >= PREFETCH_SCORE:
            selected.append(expert_id)
            policy.prefetch_selected += 1
        else:
            policy.prefetch_skipped += 1

    if not selected:
        return

    # Reuse the existing asynchronous prefetch implementation for the selected
    # hot candidates; no model computation is changed here.
    return _ORIGINAL_WARM(root, layer_prefix, selected)


def _adaptive_cache_stats(root: Path) -> dict[str, int | float]:
    stats = dict(_ORIGINAL_CACHE_STATS(root))
    expert = official._EXPERT_CACHES.get(root.resolve())
    adaptive = _policy(root)
    snap = adaptive.snapshot(expert)
    stats.update(
        {
            "adaptive_tracked": int(snap["tracked"]),
            "adaptive_hot": int(snap["hot"]),
            "adaptive_promotions": int(snap["promotions"]),
            "adaptive_demotions": int(snap["demotions"]),
            "adaptive_q4_hits": int(snap["q4_hits"]),
            "adaptive_cold_misses": int(snap["cold_misses"]),
            "adaptive_prefetch_selected": int(snap["prefetch_selected"]),
            "adaptive_prefetch_skipped": int(snap["prefetch_skipped"]),
        }
    )
    return stats


def _adaptive_print_cache(root: Path, label: str) -> None:
    _ORIGINAL_PRINT_CACHE(root, label)
    expert = official._EXPERT_CACHES.get(root.resolve())
    snap = _policy(root).snapshot(expert)
    print(
        f"  adaptive experts: tracked={snap['tracked']} | hot={snap['hot']} | "
        f"fp8_hits={snap['fp8_hits']} | q4_hits={snap['q4_hits']} | "
        f"cold_misses={snap['cold_misses']} | promotions={snap['promotions']} | "
        f"demotions={snap['demotions']} | promotion_after={PROMOTION_HITS} Q4 hits"
    )
    print(
        f"  adaptive prefetch: selected={snap['prefetch_selected']} | "
        f"skipped={snap['prefetch_skipped']} | threshold={PREFETCH_SCORE:.2f}"
    )
    if snap["top"]:
        top_text = ", ".join(
            f"L{item['layer']:02d}/E{item['expert']}={item['score']:.2f}:{item['tier']}({item['accesses']}x)"
            for item in snap["top"]
        )
        print(f"  adaptive top: {top_text}")


# Apply the policy after the dedicated official cache has been installed.
RoutedExpertCache._insert_fp8_locked = _adaptive_insert
chat._expert_projection_triplet = _adaptive_triplet
chat._warm_expert_raw_cache = _adaptive_warm
chat.cache_stats = _adaptive_cache_stats
chat.print_cache = _adaptive_print_cache
