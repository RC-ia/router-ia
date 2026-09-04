from __future__ import annotations

"""Adaptive Q4 RAM bank for routed experts.

The generic RAM tensor cache stays independent. Evicted routed experts are
retained as Q4 host entries up to a dedicated byte budget, ordered globally by
current-generation heat. Under pressure, the lowest-value Q4 experts are
removed first.
"""

import os
from collections import OrderedDict
from pathlib import Path
from threading import Lock

from . import qwen36_adaptive_experts as adaptive
from . import qwen36_expert_cache as expert_cache_module
from . import qwen36_official_optimizations as official
from .qwen36_expert_cache import RoutedExpertCache

RAM_Q4_GB = max(float(os.getenv("QWEN36_EXPERT_RAM_Q4_GB", "5.0")), 0.25)
RAM_Q4_BUDGET_BYTES = int(RAM_Q4_GB * 1024**3)
RAM_Q4_MIN_SCORE = float(os.getenv("QWEN36_EXPERT_RAM_Q4_MIN_SCORE", "0.0"))

_BANKS: dict[Path, "AdaptiveExpertRAMBank"] = {}
_BANKS_LOCK = Lock()


class AdaptiveExpertRAMBank:
    def __init__(self, budget_bytes: int) -> None:
        self.budget_bytes = max(int(budget_bytes), 0)
        self.entries: OrderedDict[tuple[int, int], tuple] = OrderedDict()
        self.sizes: dict[tuple[int, int], int] = {}
        self.bytes_used = 0
        self.insertions = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.lock = Lock()

    def _remove_locked(self, key: tuple[int, int]) -> None:
        self.entries.pop(key, None)
        self.bytes_used -= self.sizes.pop(key, 0)

    def _score(self, key: tuple[int, int]) -> float:
        layer, expert = key
        return adaptive.score_for_eviction(layer, expert)

    def put(self, layer: int, expert_id: int, entry) -> bool:
        key = (int(layer), int(expert_id))
        size = RoutedExpertCache._q4_size(entry)
        if self.budget_bytes <= 0 or size > self.budget_bytes:
            return False
        with self.lock:
            if key in self.entries:
                self._remove_locked(key)
            while self.bytes_used + size > self.budget_bytes and self.entries:
                victim = min(self.entries, key=self._score)
                self._remove_locked(victim)
                self.evictions += 1
            if self.bytes_used + size > self.budget_bytes:
                return False
            self.entries[key] = entry
            self.sizes[key] = size
            self.bytes_used += size
            self.insertions += 1
            return True

    def clear(self) -> None:
        with self.lock:
            self.entries.clear()
            self.sizes.clear()
            self.bytes_used = 0

    def snapshot(self) -> dict[str, int | float]:
        with self.lock:
            return {
                "items": len(self.entries),
                "bytes": self.bytes_used,
                "budget_bytes": self.budget_bytes,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / max(self.hits + self.misses, 1) * 100.0,
                "insertions": self.insertions,
                "evictions": self.evictions,
            }


def _bank(root: Path) -> AdaptiveExpertRAMBank:
    key = root.resolve()
    with _BANKS_LOCK:
        bank = _BANKS.get(key)
        if bank is None:
            bank = AdaptiveExpertRAMBank(RAM_Q4_BUDGET_BYTES)
            _BANKS[key] = bank
        return bank


def _score_for_eviction(layer: int, expert_id: int) -> float:
    for root in official._EXPERT_CACHES:
        policy = adaptive._POLICIES.get(root)
        if policy is not None:
            return float(policy.score(int(layer), int(expert_id)))
    return 0.0


adaptive.score_for_eviction = _score_for_eviction

_ORIGINAL_INSERT = RoutedExpertCache._insert_fp8_locked


def _insert_with_ram(self: RoutedExpertCache, layer: int, expert_id: int, entry) -> None:
    layer = int(layer)
    expert_id = int(expert_id)
    policy = None
    root_for_cache: Path | None = None
    for root, cache in official._EXPERT_CACHES.items():
        if cache is self:
            root_for_cache = root
            policy = adaptive._POLICIES.get(root)
            break

    bank = self.fp8_entries.setdefault(layer, OrderedDict())
    old_ids = set(bank)
    if policy is not None and bank:
        ordered = sorted(bank.items(), key=lambda item: policy.score(layer, int(item[0])))
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
        score = policy.score(layer, victim_id) if policy is not None else 0.0
        retained = False
        if root_for_cache is not None and score >= RAM_Q4_MIN_SCORE:
            cold_gpu = expert_cache_module._q4_quantize_entry_from_fp8(victim)
            cold = expert_cache_module._move_q4_to_cpu(cold_gpu)
            retained = _bank(root_for_cache).put(layer, victim_id, cold)
            if retained:
                q4 = self.q4_entries.setdefault(layer, OrderedDict())
                old_q4 = q4.pop(victim_id, None)
                if old_q4 is not None:
                    self._erase(layer, victim_id, "q4")
                q4[victim_id] = cold
                self._record(layer, victim_id, "q4", self._q4_size(cold))
                self.fp8_to_q4 += 1
        if not retained:
            self.evictions += 1

    if policy is not None:
        new_ids = set(bank)
        for victim in old_ids - new_ids:
            if victim != expert_id:
                policy.note_demotion(layer, int(victim))


RoutedExpertCache._insert_fp8_locked = _insert_with_ram


def stats(root: Path) -> dict[str, int | float]:
    return _bank(root).snapshot()


def clear(root: Path) -> None:
    _bank(root).clear()


def print_stats(root: Path) -> None:
    snap = stats(root)
    print(
        f"  adaptive expert RAM Q4: "
        f"{snap['bytes'] / 1024**2:.1f}/{snap['budget_bytes'] / 1024**2:.1f} MiB | "
        f"items={snap['items']} | hits={snap['hits']} | misses={snap['misses']} | "
        f"hit_rate={snap['hit_rate']:.2f}% | evictions={snap['evictions']}"
    )


print(
    f"adaptive_expert_ram=global-Q4|budget={RAM_Q4_GB:.2f}GiB|"
    f"min_score={RAM_Q4_MIN_SCORE:.2f}|evict=lowest-heat"
)
