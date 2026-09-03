from __future__ import annotations

"""Persistent GPU cache for complete Qwen3.6 routed experts.

Each cache entry is one (layer, expert) pair containing its gate/up/down
FP16 projection matrices. The cache guarantees a small resident minimum per
layer, while the remaining slots form a shared LRU pool that can be consumed
by hot layers. This avoids the pathological global-LRU scan while also
avoiding the rigidity of an identical slot count for every layer.
"""

from collections import OrderedDict
from threading import Lock

import torch

from . import qwen36_dequant as dequant


MODEL_LAYERS = 40
EXPERTS_PER_LAYER = 256
TOP_K = 8
EXPERT_HIDDEN = 512
HIDDEN = 2048
FP16_EXPERT_BYTES_ESTIMATE = 3 * EXPERT_HIDDEN * HIDDEN * 2
MIN_SLOTS_PER_LAYER = 2


class RoutedExpertCache:
    """Adaptive per-layer minimum + shared global LRU for complete experts."""

    def __init__(self, budget_bytes: int, layers: int = MODEL_LAYERS) -> None:
        self.budget_bytes = max(int(budget_bytes), 0)
        self.layers = max(int(layers), 1)

        self.total_slots = (
            self.budget_bytes // FP16_EXPERT_BYTES_ESTIMATE
            if self.budget_bytes
            else 0
        )
        self.min_slots_per_layer = (
            min(MIN_SLOTS_PER_LAYER, self.total_slots // self.layers)
            if self.total_slots
            else 0
        )
        guaranteed_slots = self.min_slots_per_layer * self.layers
        self.shared_slots = max(self.total_slots - guaranteed_slots, 0)

        self.entries: dict[
            int,
            OrderedDict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        ] = {layer: OrderedDict() for layer in range(self.layers)}
        self.entry_bytes: dict[tuple[int, int], int] = {}

        # Entries above each layer's guaranteed minimum are tagged as shared
        # and participate in the global overflow LRU.
        self.shared_lru: OrderedDict[tuple[int, int], None] = OrderedDict()

        self.bytes_used = 0
        self.hits = 0
        self.misses = 0
        self.loads = 0
        self.evictions = 0
        self.shared_evictions = 0
        self.local_evictions = 0
        self.lock = Lock()

    @staticmethod
    def _entry_size(entry: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> int:
        return sum(int(t.numel()) * int(t.element_size()) for t in entry)

    def _layer_bytes(self, layer: int) -> int:
        bank = self.entries.get(layer)
        if not bank:
            return 0
        return sum(
            self.entry_bytes.get((int(layer), int(expert_id)), 0)
            for expert_id in bank.keys()
        )

    def _is_shared(self, layer: int, expert_id: int) -> bool:
        return (int(layer), int(expert_id)) in self.shared_lru

    def get(self, layer: int, expert_id: int):
        layer = int(layer)
        expert_id = int(expert_id)
        key = (layer, expert_id)
        with self.lock:
            bank = self.entries.setdefault(layer, OrderedDict())
            entry = bank.get(expert_id)
            if entry is None:
                self.misses += 1
                return None

            self.hits += 1
            bank.move_to_end(expert_id)
            if key in self.shared_lru:
                self.shared_lru.move_to_end(key)
            return entry

    def _drop(self, layer: int, expert_id: int) -> None:
        key = (int(layer), int(expert_id))
        bank = self.entries[layer]
        bank.pop(expert_id, None)
        self.bytes_used -= self.entry_bytes.pop(key, 0)
        self.shared_lru.pop(key, None)

    def _evict_one_shared(self) -> bool:
        if not self.shared_lru:
            return False
        victim_layer, victim_expert = next(iter(self.shared_lru))
        self._drop(victim_layer, victim_expert)
        self.evictions += 1
        self.shared_evictions += 1
        return True

    def put(self, layer: int, expert_id: int, entry) -> bool:
        layer = int(layer)
        expert_id = int(expert_id)
        size = self._entry_size(entry)
        key = (layer, expert_id)

        with self.lock:
            bank = self.entries.setdefault(layer, OrderedDict())
            old = bank.get(expert_id)
            if old is not None:
                self._drop(layer, expert_id)

            if self.budget_bytes <= 0 or size > self.budget_bytes:
                return False

            while len(self.entry_bytes) >= self.total_slots and self.total_slots:
                if self._evict_one_shared():
                    continue

                victim_layer = None
                victim_expert = None
                for candidate_layer, candidate_bank in self.entries.items():
                    if len(candidate_bank) <= self.min_slots_per_layer:
                        continue
                    for candidate_expert in candidate_bank.keys():
                        if self._is_shared(candidate_layer, candidate_expert):
                            victim_layer = candidate_layer
                            victim_expert = candidate_expert
                            break
                    if victim_layer is not None:
                        break

                if victim_layer is None:
                    return False

                self._drop(victim_layer, victim_expert)  # type: ignore[arg-type]
                self.evictions += 1
                self.local_evictions += 1

            # The first min_slots_per_layer entries in a layer are protected.
            # Any subsequent entries consume the shared adaptive pool.
            shared = len(bank) >= self.min_slots_per_layer
            bank[expert_id] = entry
            self.entry_bytes[key] = size
            self.bytes_used += size
            if shared:
                self.shared_lru[key] = None
            self.loads += 1
            return True

    def get_or_load(self, store, layer: int, expert_id: int, layer_prefix: str):
        hit = self.get(layer, expert_id)
        if hit is not None:
            return hit

        expert_prefix = f"{layer_prefix}mlp.experts.{expert_id}"
        names = ("gate_proj", "up_proj", "down_proj")
        raw_weights = []
        raw_scales = []
        for name in names:
            proj = expert_prefix + "." + name
            raw_weights.append(store.load(proj + ".weight", device="cpu"))
            raw_scales.append(store.load(proj + ".weight_scale_inv", device="cpu"))

        if all(weight.dtype == torch.float8_e4m3fn for weight in raw_weights):
            # Qwen3.6 uses different orientations for down_proj:
            # gate/up = [512, 2048], down = [2048, 512]. The batched
            # dequantizer requires equal-shaped matrices, so dequantize the
            # two matching projections together and down_proj separately.
            gate_up_weights = torch.stack(raw_weights[:2], dim=0).to(device="cuda")
            gate_up_scales = torch.stack(raw_scales[:2], dim=0).to(device="cuda")
            gate_up_batch = dequant.dequantize_fp8_blockwise_batch(
                gate_up_weights, gate_up_scales
            ).to(dtype=torch.float16)

            down_weight = raw_weights[2].to(device="cuda")
            down_scale = raw_scales[2].to(device="cuda")
            down_output = dequant.dequantize_fp8_blockwise(
                down_weight, down_scale
            ).to(dtype=torch.float16)

            entry = (
                gate_up_batch[0],
                gate_up_batch[1],
                down_output,
            )
            del gate_up_weights, gate_up_scales, gate_up_batch
            del down_weight, down_scale
        else:
            entry = tuple(
                weight.to(device="cuda", dtype=torch.float16)
                for weight in raw_weights
            )

        self.put(layer, expert_id, entry)
        return entry

    def snapshot(self) -> dict[str, int | float]:
        with self.lock:
            total = self.hits + self.misses
            items = len(self.entry_bytes)
            shared_items = len(self.shared_lru)
            protected_items = max(items - shared_items, 0)
            return {
                "items": items,
                "bytes": self.bytes_used,
                "budget_bytes": self.budget_bytes,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / total * 100.0 if total else 0.0,
                "loads": self.loads,
                "evictions": self.evictions,
                "shared_evictions": self.shared_evictions,
                "local_evictions": self.local_evictions,
                "layers": self.layers,
                "layers_populated": sum(bool(bank) for bank in self.entries.values()),
                "total_slots": self.total_slots,
                "min_slots_per_layer": self.min_slots_per_layer,
                "shared_slots": self.shared_slots,
                "shared_items": shared_items,
                "protected_items": protected_items,
            }

    def clear(self) -> None:
        with self.lock:
            for bank in self.entries.values():
                bank.clear()
            self.entry_bytes.clear()
            self.shared_lru.clear()
            self.bytes_used = 0
