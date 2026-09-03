from __future__ import annotations

"""Persistent two-tier GPU cache for complete Qwen3.6 routed experts.

Hot experts stay in FP16 for immediate GEMM use. Less-used experts stay in
block-scaled FP8 plus scales. A compressed hit is reconstructed only for the
current operation; the compact FP8 entry remains resident. FP16 eviction is
handled without a Q4 conversion stage.
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
BLOCK = 128
FP8_MAX = 448.0
FP16_EXPERT_BYTES_ESTIMATE = 3 * EXPERT_HIDDEN * HIDDEN * 2
HOT_SLOTS_PER_LAYER = 2
FP8_SLOTS_PER_LAYER = 4
TOTAL_SLOTS_PER_LAYER = HOT_SLOTS_PER_LAYER + FP8_SLOTS_PER_LAYER

FP8Matrix = tuple[torch.Tensor, torch.Tensor]
HotEntry = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
WarmEntry = tuple[FP8Matrix, FP8Matrix, FP8Matrix]


def _fp8_quantize_blockwise(weight: torch.Tensor) -> FP8Matrix:
    if weight.ndim != 2:
        raise ValueError(f"Expected 2-D matrix, got {tuple(weight.shape)}")
    rows, cols = map(int, weight.shape)
    padded_rows = (rows + BLOCK - 1) // BLOCK * BLOCK
    padded_cols = (cols + BLOCK - 1) // BLOCK * BLOCK
    padded = torch.zeros(
        (padded_rows, padded_cols), device=weight.device, dtype=torch.float32
    )
    padded[:rows, :cols] = weight.float()
    blocks = padded.reshape(
        padded_rows // BLOCK, BLOCK, padded_cols // BLOCK, BLOCK
    )
    max_abs = blocks.abs().amax(dim=(1, 3), keepdim=True)
    scale = torch.clamp(max_abs / FP8_MAX, min=torch.finfo(torch.float32).tiny)
    quantized = (blocks / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    quantized = quantized.reshape(padded_rows, padded_cols)[:rows, :cols]
    scales = scale.reshape(padded_rows // BLOCK, padded_cols // BLOCK)
    return quantized, scales


def _fp8_dequantize_matrix(matrix: FP8Matrix) -> torch.Tensor:
    weight, scales = matrix
    return dequant.dequantize_fp8_blockwise(weight, scales).to(dtype=torch.float16)


def _fp8_quantize_entry(entry: HotEntry) -> WarmEntry:
    return tuple(_fp8_quantize_blockwise(t) for t in entry)  # type: ignore[return-value]


def _fp8_dequantize_entry(entry: WarmEntry) -> HotEntry:
    return tuple(_fp8_dequantize_matrix(matrix) for matrix in entry)  # type: ignore[return-value]


class RoutedExpertCache:
    """Per-layer LRU cache with FP16 hot tier and FP8 compact tier."""

    def __init__(self, budget_bytes: int, layers: int = MODEL_LAYERS) -> None:
        self.budget_bytes = max(int(budget_bytes), 0)
        self.layers = max(int(layers), 1)
        self.slots_per_layer = min(TOTAL_SLOTS_PER_LAYER, self._budget_slots())
        self.total_slots = self.slots_per_layer * self.layers

        remaining = self.slots_per_layer
        self.hot_slots = min(HOT_SLOTS_PER_LAYER, remaining)
        remaining -= self.hot_slots
        self.fp8_slots = min(FP8_SLOTS_PER_LAYER, remaining)

        self.entries: dict[int, OrderedDict[int, HotEntry]] = {
            layer: OrderedDict() for layer in range(self.layers)
        }
        self.fp8_entries: dict[int, OrderedDict[int, WarmEntry]] = {
            layer: OrderedDict() for layer in range(self.layers)
        }
        self.entry_bytes: dict[tuple[int, int, str], int] = {}
        self.bytes_used = 0

        self.hits = 0
        self.misses = 0
        self.loads = 0
        self.evictions = 0
        self.hot_hits = 0
        self.fp8_hits = 0
        self.fp16_to_fp8 = 0
        self.fp8_drops = 0
        self.lock = Lock()

    def _budget_slots(self) -> int:
        if not self.budget_bytes:
            return 0
        # Ensure the original two-FP16-hot working set can exist on every
        # layer, then add the compact FP8 tier without exceeding the nominal
        # persistent-expert envelope.
        old_per_layer = HOT_SLOTS_PER_LAYER * FP16_EXPERT_BYTES_ESTIMATE
        layer_count = max(self.layers, 1)
        supported_layers = self.budget_bytes // old_per_layer
        if supported_layers < layer_count:
            return min(
                HOT_SLOTS_PER_LAYER,
                self.budget_bytes // FP16_EXPERT_BYTES_ESTIMATE // layer_count,
            )
        return TOTAL_SLOTS_PER_LAYER

    @staticmethod
    def _hot_size(entry: HotEntry) -> int:
        return sum(int(t.numel()) * int(t.element_size()) for t in entry)

    @staticmethod
    def _fp8_size(entry: WarmEntry) -> int:
        total = 0
        for weight, scales in entry:
            total += int(weight.numel()) * int(weight.element_size())
            total += int(scales.numel()) * int(scales.element_size())
        return total

    def _record_hot(self, layer: int, expert_id: int, entry: HotEntry) -> None:
        self.entry_bytes[(layer, expert_id, "fp16")] = self._hot_size(entry)

    def _record_fp8(self, layer: int, expert_id: int, entry: WarmEntry) -> None:
        self.entry_bytes[(layer, expert_id, "fp8")] = self._fp8_size(entry)

    def _erase_bytes(self, layer: int, expert_id: int, tier: str) -> None:
        self.bytes_used -= self.entry_bytes.pop((layer, expert_id, tier), 0)

    def get(self, layer: int, expert_id: int):
        layer = int(layer)
        expert_id = int(expert_id)
        with self.lock:
            hot = self.entries.setdefault(layer, OrderedDict())
            entry = hot.get(expert_id)
            if entry is not None:
                self.hits += 1
                self.hot_hits += 1
                hot.move_to_end(expert_id)
                return entry

            fp8_bank = self.fp8_entries.setdefault(layer, OrderedDict())
            compact = fp8_bank.get(expert_id)
            if compact is not None:
                self.hits += 1
                self.fp8_hits += 1
                fp8_bank.move_to_end(expert_id)
                # Important: do not promote/move the FP8 entry into the hot
                # tier. Reconstruct only a transient compute copy, leaving
                # the compact resident representation untouched.
                return _fp8_dequantize_entry(compact)

            self.misses += 1
            return None

    def _insert_hot_locked(self, layer: int, expert_id: int, entry: HotEntry) -> None:
        hot = self.entries.setdefault(layer, OrderedDict())
        old = hot.pop(expert_id, None)
        if old is not None:
            self._erase_bytes(layer, expert_id, "fp16")

        while hot and len(hot) >= self.hot_slots:
            victim_id, victim = hot.popitem(last=False)
            self._erase_bytes(layer, victim_id, "fp16")

            if self.fp8_slots > 0:
                # Compress exactly once, to the source model's FP8 family.
                # There is deliberately no FP8->Q4 compression on eviction.
                compact = _fp8_quantize_entry(victim)
                fp8_bank = self.fp8_entries[layer]
                previous = fp8_bank.pop(victim_id, None)
                if previous is not None:
                    self._erase_bytes(layer, victim_id, "fp8")
                fp8_bank[victim_id] = compact
                self._record_fp8(layer, victim_id, compact)
                self.bytes_used += self.entry_bytes[(layer, victim_id, "fp8")]
                self.fp16_to_fp8 += 1

                while len(fp8_bank) > self.fp8_slots:
                    cold_id, _ = fp8_bank.popitem(last=False)
                    self._erase_bytes(layer, cold_id, "fp8")
                    self.fp8_drops += 1
                    self.evictions += 1
            else:
                self.evictions += 1

        if expert_id in self.fp8_entries[layer]:
            self.fp8_entries[layer].pop(expert_id, None)
            self._erase_bytes(layer, expert_id, "fp8")

        hot[expert_id] = entry
        self._record_hot(layer, expert_id, entry)
        self.bytes_used += self.entry_bytes[(layer, expert_id, "fp16")]

    def put(self, layer: int, expert_id: int, entry: HotEntry):
        layer = int(layer)
        expert_id = int(expert_id)
        with self.lock:
            if self.hot_slots <= 0:
                return False
            self._insert_hot_locked(layer, expert_id, entry)
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
            entry = (gate_up_batch[0], gate_up_batch[1], down_output)
            del gate_up_weights, gate_up_scales, gate_up_batch
            del down_weight, down_scale
        else:
            entry = tuple(
                weight.to(device="cuda", dtype=torch.float16) for weight in raw_weights
            )

        self.put(layer, expert_id, entry)
        return entry

    def snapshot(self) -> dict[str, int | float]:
        with self.lock:
            total = self.hits + self.misses
            hot_items = sum(len(bank) for bank in self.entries.values())
            fp8_items = sum(len(bank) for bank in self.fp8_entries.values())
            return {
                "items": hot_items + fp8_items,
                "bytes": self.bytes_used,
                "budget_bytes": self.budget_bytes,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / total * 100.0 if total else 0.0,
                "loads": self.loads,
                "evictions": self.evictions,
                "layers": self.layers,
                "layers_populated": sum(
                    bool(self.entries[layer] or self.fp8_entries[layer])
                    for layer in range(self.layers)
                ),
                "slots_per_layer": self.slots_per_layer,
                "total_slots": self.total_slots,
                "hot_slots_per_layer": self.hot_slots,
                "warm_slots_per_layer": self.fp8_slots,
                "cold_slots_per_layer": 0,
                "hot_items": hot_items,
                "warm_items": fp8_items,
                "cold_items": 0,
                "hot_hits": self.hot_hits,
                "fp8_hits": self.fp8_hits,
                "q4_hits": 0,
                "fp16_to_fp8": self.fp16_to_fp8,
                "fp8_to_q4": 0,
                "q4_drops": self.fp8_drops,
                "shared_items": fp8_items,
                "protected_items": hot_items,
                "min_slots_per_layer": self.hot_slots,
                "shared_slots": self.fp8_slots,
            }

    def clear(self) -> None:
        with self.lock:
            for layer in range(self.layers):
                self.entries[layer].clear()
                self.fp8_entries[layer].clear()
            self.entry_bytes.clear()
            self.bytes_used = 0
