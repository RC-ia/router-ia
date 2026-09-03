from __future__ import annotations

"""Persistent GPU cache for complete Qwen3.6 routed experts.

Each cache entry is one (layer, expert) pair containing its gate/up/down
FP16 projection matrices. Entries are evicted atomically using LRU, so a
partially resident expert cannot occur.
"""

from collections import OrderedDict
from threading import Lock

import torch

from . import qwen36_dequant as dequant


class RoutedExpertCache:
    def __init__(self, budget_bytes: int) -> None:
        self.budget_bytes = max(int(budget_bytes), 0)
        self.entries: OrderedDict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = OrderedDict()
        self.entry_bytes: dict[tuple[int, int], int] = {}
        self.bytes_used = 0
        self.hits = 0
        self.misses = 0
        self.loads = 0
        self.evictions = 0
        self.lock = Lock()

    @staticmethod
    def _entry_size(entry: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> int:
        return sum(int(t.numel()) * int(t.element_size()) for t in entry)

    def get(self, layer: int, expert_id: int):
        key = (int(layer), int(expert_id))
        with self.lock:
            entry = self.entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            self.hits += 1
            self.entries.move_to_end(key)
            return entry

    def put(self, layer: int, expert_id: int, entry):
        key = (int(layer), int(expert_id))
        size = self._entry_size(entry)
        with self.lock:
            old = self.entries.pop(key, None)
            if old is not None:
                self.bytes_used -= self.entry_bytes.pop(key, 0)

            if self.budget_bytes <= 0 or size > self.budget_bytes:
                return False

            while self.bytes_used + size > self.budget_bytes and self.entries:
                victim, _ = self.entries.popitem(last=False)
                self.bytes_used -= self.entry_bytes.pop(victim, 0)
                self.evictions += 1

            self.entries[key] = entry
            self.entry_bytes[key] = size
            self.bytes_used += size
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
            weight_batch = torch.stack(raw_weights, dim=0).to(device="cuda")
            scale_batch = torch.stack(raw_scales, dim=0).to(device="cuda")
            output_batch = dequant.dequantize_fp8_blockwise_batch(weight_batch, scale_batch)
            output_batch = output_batch.to(dtype=torch.float16)
            entry = tuple(output_batch.unbind(dim=0))
            del weight_batch, scale_batch, output_batch
        else:
            entry = tuple(weight.to(device="cuda", dtype=torch.float16) for weight in raw_weights)

        self.put(layer, expert_id, entry)
        return entry

    def snapshot(self) -> dict[str, int | float]:
        with self.lock:
            total = self.hits + self.misses
            return {
                "items": len(self.entries),
                "bytes": self.bytes_used,
                "budget_bytes": self.budget_bytes,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / total * 100.0 if total else 0.0,
                "loads": self.loads,
                "evictions": self.evictions,
            }

    def clear(self) -> None:
        with self.lock:
            self.entries.clear()
            self.entry_bytes.clear()
            self.bytes_used = 0
