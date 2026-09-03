from __future__ import annotations

"""Persistent GPU cache for complete Qwen3.6 routed experts.

Each cache entry is one (layer, expert) pair containing its gate/up/down
FP16 projection matrices. Entries are kept in a small LRU bank *per layer*.
This is important for autoregressive MoE decoding: a global LRU sees a
40-layer sequential scan and can evict the early layers before the next token
reaches them, producing zero reuse even when expert IDs repeat.
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


class RoutedExpertCache:
    """Per-layer bounded LRU cache for complete routed experts."""

    def __init__(self, budget_bytes: int, layers: int = MODEL_LAYERS) -> None:
        self.budget_bytes = max(int(budget_bytes), 0)
        self.layers = max(int(layers), 1)

        # Equal per-layer banks prevent the sequential 40-layer decode sweep
        # from turning a global LRU into a streaming buffer. With the default
        # ~1.2 GiB budget this yields five complete experts per layer.
        bytes_per_layer = self.budget_bytes // self.layers
        estimated_slots = bytes_per_layer // FP16_EXPERT_BYTES_ESTIMATE
        self.slots_per_layer = max(int(estimated_slots), 1) if self.budget_bytes else 0
        self.layer_budgets = {
            layer: self.slots_per_layer * FP16_EXPERT_BYTES_ESTIMATE
            for layer in range(self.layers)
        }

        self.entries: dict[
            int,
            OrderedDict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        ] = {layer: OrderedDict() for layer in range(self.layers)}
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
        layer = int(layer)
        expert_id = int(expert_id)
        with self.lock:
            bank = self.entries.setdefault(layer, OrderedDict())
            entry = bank.get(expert_id)
            if entry is None:
                self.misses += 1
                return None
            self.hits += 1
            bank.move_to_end(expert_id)
            return entry

    def put(self, layer: int, expert_id: int, entry):
        layer = int(layer)
        expert_id = int(expert_id)
        size = self._entry_size(entry)

        with self.lock:
            bank = self.entries.setdefault(layer, OrderedDict())
            key = (layer, expert_id)

            old = bank.pop(expert_id, None)
            if old is not None:
                self.bytes_used -= self.entry_bytes.pop(key, 0)

            layer_budget = self.layer_budgets.get(layer, 0)
            if layer_budget <= 0 or size > layer_budget:
                return False

            # Evict only from the same layer. Other layers keep their reusable
            # experts across the full decode sweep.
            while bank and (
                len(bank) >= self.slots_per_layer
                or self._layer_bytes(bank, layer) + size > layer_budget
            ):
                victim, _ = bank.popitem(last=False)
                self.bytes_used -= self.entry_bytes.pop((layer, victim), 0)
                self.evictions += 1

            bank[expert_id] = entry
            self.entry_bytes[key] = size
            self.bytes_used += size
            self.loads += 1
            return True

    def _layer_bytes(self, bank: OrderedDict, layer: int) -> int:
        return sum(
            self.entry_bytes.get((int(layer), int(expert_id)), 0)
            for expert_id in bank.keys()
        )

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
            del down_weight, down_scale, down_output
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
            items = sum(len(bank) for bank in self.entries.values())
            layers_populated = sum(bool(bank) for bank in self.entries.values())
            return {
                "items": items,
                "bytes": self.bytes_used,
                "budget_bytes": self.budget_bytes,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / total * 100.0 if total else 0.0,
                "loads": self.loads,
                "evictions": self.evictions,
                "layers": self.layers,
                "layers_populated": layers_populated,
                "slots_per_layer": self.slots_per_layer,
            }

    def clear(self) -> None:
        with self.lock:
            for bank in self.entries.values():
                bank.clear()
            self.entry_bytes.clear()
            self.bytes_used = 0
