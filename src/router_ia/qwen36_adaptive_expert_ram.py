from __future__ import annotations

"""Shared adaptive Q4 RAM bank for routed experts.

The bank consumes only host RAM left unused by the generic tensor cache. Its
entries mirror RoutedExpertCache.q4_entries, so Q4 hits continue through the
existing expert decoder. The lowest generation-heat experts are removed first
when the shared host budget tightens.
"""

from collections import OrderedDict
from pathlib import Path
from threading import Lock

from . import qwen36_adaptive_experts as adaptive
from . import qwen36_cached_loop as cached
from . import qwen36_expert_cache as expert_cache_module
from . import qwen36_official_optimizations as official
from .qwen36_expert_cache import RoutedExpertCache

_BANKS: dict[Path, "ExpertRAMBank"] = {}
_BANKS_LOCK = Lock()


class ExpertRAMBank:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.items: OrderedDict[tuple[int, int], None] = OrderedDict()
        self.sizes: dict[tuple[int, int], int] = {}
        self.bytes_used = 0
        self.insertions = 0
        self.evictions = 0
        self.lock = Lock()

    @property
    def total_budget_bytes(self) -> int:
        return int(cached.CACHE_BUDGET_BYTES)

    def _generic_ram_bytes(self) -> int:
        return int(cached._store(self.root).ram_cache.snapshot()["bytes"])

    def _available_bytes_locked(self) -> int:
        return max(self.total_budget_bytes - self._generic_ram_bytes(), 0)

    def _score(self, key: tuple[int, int]) -> float:
        policy = adaptive._POLICIES.get(self.root)
        if policy is None:
            return 0.0
        return float(policy.score(key[0], key[1]))

    def _remove_locked(self, key: tuple[int, int]) -> None:
        self.items.pop(key, None)
        self.bytes_used -= self.sizes.pop(key, 0)
        expert = official._EXPERT_CACHES.get(self.root)
        if expert is None:
            return
        layer, expert_id = key
        q4 = expert.q4_entries.get(layer)
        if q4 is not None and expert_id in q4:
            q4.pop(expert_id, None)
            expert._erase(layer, expert_id, "q4")
            expert.q4_drops += 1
            expert.q4_ram_evictions += 1

    def trim(self) -> None:
        with self.lock:
            available = self._available_bytes_locked()
            while self.bytes_used > available and self.items:
                victim = min(self.items, key=self._score)
                self._remove_locked(victim)
                self.evictions += 1

    def add(self, layer: int, expert_id: int, entry) -> bool:
        key = (int(layer), int(expert_id))
        size = RoutedExpertCache._q4_size(entry)
        with self.lock:
            available = self._available_bytes_locked()
            if key in self.items:
                self._remove_locked(key)
            while self.bytes_used + size > available and self.items:
                victim = min(self.items, key=self._score)
                self._remove_locked(victim)
                self.evictions += 1
                available = self._available_bytes_locked()
            if size > available:
                return False
            self.items[key] = None
            self.sizes[key] = size
            self.bytes_used += size
            self.insertions += 1
            return True

    def snapshot(self) -> dict[str, int | float]:
        with self.lock:
            return {
                "items": len(self.items),
                "bytes": self.bytes_used,
                "budget_bytes": self.total_budget_bytes,
                "generic_ram_bytes": self._generic_ram_bytes(),
                "available_bytes": self._available_bytes_locked(),
                "insertions": self.insertions,
                "evictions": self.evictions,
            }

    def clear(self) -> None:
        with self.lock:
            expert = official._EXPERT_CACHES.get(self.root)
            if expert is not None:
                for layer, expert_id in list(self.items):
                    q4 = expert.q4_entries.get(layer)
                    if q4 is not None and expert_id in q4:
                        q4.pop(expert_id, None)
                        expert._erase(layer, expert_id, "q4")
                expert.q4_drops += len(self.items)
            self.items.clear()
            self.sizes.clear()
            self.bytes_used = 0


def _bank(root: Path) -> ExpertRAMBank:
    key = root.resolve()
    with _BANKS_LOCK:
        bank = _BANKS.get(key)
        if bank is None:
            bank = ExpertRAMBank(key)
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


def _insert_with_shared_ram(self: RoutedExpertCache, layer: int, expert_id: int, entry) -> None:
    layer = int(layer)
    expert_id = int(expert_id)
    root_for_cache: Path | None = None
    policy = None
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
        retained = False
        if root_for_cache is not None:
            cold_gpu = expert_cache_module._q4_quantize_entry_from_fp8(victim)
            cold = expert_cache_module._move_q4_to_cpu(cold_gpu)
            retained = _bank(root_for_cache).add(layer, victim_id, cold)
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


RoutedExpertCache._insert_fp8_locked = _insert_with_shared_ram


def stats(root: Path) -> dict[str, int | float]:
    return _bank(root).snapshot()


def clear(root: Path) -> None:
    _bank(root).clear()


def maintain(root: Path) -> None:
    _bank(root).trim()


def print_stats(root: Path) -> None:
    snap = stats(root)
    print(
        f"  adaptive expert RAM Q4: {snap['bytes'] / 1024**2:.1f}/"
        f"{snap['budget_bytes'] / 1024**2:.1f} MiB total-RAM | "
        f"generic={snap['generic_ram_bytes'] / 1024**2:.1f} MiB | "
        f"free={snap['available_bytes'] / 1024**2:.1f} MiB | items={snap['items']} | "
        f"evictions={snap['evictions']}"
    )


print("adaptive_expert_ram=shared-RAM-Q4|evict=lowest-heat|budget=remaining-host-cache")
