from __future__ import annotations

"""Persistent tiered GPU cache for complete Qwen3.6 routed experts.

Each layer keeps three LRU tiers:

* hot  = FP16, fastest to consume;
* warm = blockwise FP8 + scales, roughly half the FP16 footprint;
* cold = packed symmetric Q4 + per-matrix scales, roughly one quarter of
  FP16.

Experts are therefore compressed before they are actually dropped from the
GPU cache. A hit in a compressed tier is transparently reconstructed to FP16
before being returned to the existing MoE compute path.
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
BLOCK = 128
FP8_MAX = 448.0

# The old cache held 5 complete FP16 experts per layer. The tiered cache keeps
# roughly the same nominal memory envelope while tracking 10 experts/layer.
HOT_SLOTS_PER_LAYER = 2
WARM_SLOTS_PER_LAYER = 4
COLD_SLOTS_PER_LAYER = 4
TOTAL_SLOTS_PER_LAYER = (
    HOT_SLOTS_PER_LAYER + WARM_SLOTS_PER_LAYER + COLD_SLOTS_PER_LAYER
)

FP8_EXPERT_BYTES_ESTIMATE = 3 * EXPERT_HIDDEN * HIDDEN + 3 * 4 * (
    (EXPERT_HIDDEN + BLOCK - 1) // BLOCK
) * ((HIDDEN + BLOCK - 1) // BLOCK)
Q4_EXPERT_BYTES_ESTIMATE = (
    3 * ((EXPERT_HIDDEN * HIDDEN + 1) // 2) + 3 * 4
)

HotEntry = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
FP8Matrix = tuple[torch.Tensor, torch.Tensor]
WarmEntry = tuple[FP8Matrix, FP8Matrix, FP8Matrix]
Q4Matrix = tuple[torch.Tensor, torch.Tensor, tuple[int, int]]
ColdEntry = tuple[Q4Matrix, Q4Matrix, Q4Matrix]


def _fp8_quantize_blockwise(weight: torch.Tensor) -> FP8Matrix:
    """Compress an FP16 matrix as blockwise FP8 plus inverse scales."""
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
        padded_rows // BLOCK,
        BLOCK,
        padded_cols // BLOCK,
        BLOCK,
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


def _q4_quantize(weight: torch.Tensor) -> Q4Matrix:
    """Quantize one matrix to symmetric packed Q4 with one scale."""
    if weight.ndim != 2:
        raise ValueError(f"Expected 2-D matrix, got {tuple(weight.shape)}")

    rows, cols = map(int, weight.shape)
    x = weight.float()
    max_abs = x.abs().amax()
    scale = torch.clamp(max_abs / 7.0, min=torch.finfo(torch.float32).tiny)
    q = torch.round(x / scale).clamp(-7, 7).to(torch.int16) + 8

    flat = q.reshape(-1)
    if flat.numel() & 1:
        flat = torch.cat(
            [flat, torch.full((1,), 8, device=flat.device, dtype=flat.dtype)]
        )
    packed = flat[0::2].to(torch.uint8) | (flat[1::2].to(torch.uint8) << 4)
    return packed, scale.to(dtype=torch.float16), (rows, cols)


def _q4_dequantize_matrix(matrix: Q4Matrix) -> torch.Tensor:
    packed, scale, shape = matrix
    low = (packed & 0x0F).to(torch.int16) - 8
    high = ((packed >> 4) & 0x0F).to(torch.int16) - 8
    q = torch.stack((low, high), dim=1).reshape(-1)[: shape[0] * shape[1]]
    return (q.to(torch.float32) * scale.float()).reshape(shape).to(torch.float16)


class RoutedExpertCache:
    """Fixed per-layer three-tier LRU cache for complete routed experts."""

    def __init__(self, budget_bytes: int, layers: int = MODEL_LAYERS) -> None:
        self.budget_bytes = max(int(budget_bytes), 0)
        self.layers = max(int(layers), 1)
        self.slots_per_layer = min(TOTAL_SLOTS_PER_LAYER, self._budget_slots())
        self.total_slots = self.slots_per_layer * self.layers

        # Keep the same 2/4/4 split whenever the budget can support it. For a
        # smaller budget, degrade the lower tiers first rather than removing
        # hot capacity.
        if self.slots_per_layer >= TOTAL_SLOTS_PER_LAYER:
            self.hot_slots = HOT_SLOTS_PER_LAYER
            self.warm_slots = WARM_SLOTS_PER_LAYER
            self.cold_slots = COLD_SLOTS_PER_LAYER
        else:
            remaining = self.slots_per_layer
            self.hot_slots = min(HOT_SLOTS_PER_LAYER, remaining)
            remaining -= self.hot_slots
            self.warm_slots = min(WARM_SLOTS_PER_LAYER, remaining)
            remaining -= self.warm_slots
            self.cold_slots = min(COLD_SLOTS_PER_LAYER, remaining)

        self.entries: dict[int, OrderedDict[int, HotEntry]] = {
            layer: OrderedDict() for layer in range(self.layers)
        }
        self.warm_entries: dict[int, OrderedDict[int, WarmEntry]] = {
            layer: OrderedDict() for layer in range(self.layers)
        }
        self.cold_entries: dict[int, OrderedDict[int, ColdEntry]] = {
            layer: OrderedDict() for layer in range(self.layers)
        }
        self.bytes_used = 0
        self.hits = 0
        self.misses = 0
        self.loads = 0
        self.evictions = 0
        self.hot_hits = 0
        self.fp8_hits = 0
        self.q4_hits = 0
        self.fp16_to_fp8 = 0
        self.fp8_to_q4 = 0
        self.q4_drops = 0
        self.lock = Lock()

    def _budget_slots(self) -> int:
        if not self.budget_bytes:
            return 0
        # The 2/4/4 tier shape is intentionally sized against the same
        # per-layer budget as the previous five-FP16-expert cache.
        per_layer_old = 5 * FP16_EXPERT_BYTES_ESTIMATE
        return min(TOTAL_SLOTS_PER_LAYER, self.budget_bytes // per_layer_old * TOTAL_SLOTS_PER_LAYER // self.layers)

    @staticmethod
    def _hot_size(entry: HotEntry) -> int:
        return sum(int(t.numel()) * int(t.element_size()) for t in entry)

    @staticmethod
    def _warm_size(entry: WarmEntry) -> int:
        total = 0
        for weight, scales in entry:
            total += int(weight.numel()) * int(weight.element_size())
            total += int(scales.numel()) * int(scales.element_size())
        return total

    @staticmethod
    def _cold_size(entry: ColdEntry) -> int:
        total = 0
        for packed, scale, _shape in entry:
            total += int(packed.numel()) * int(packed.element_size())
            total += int(scale.numel()) * int(scale.element_size())
        return total

    def _remove_key(self, layer: int, expert_id: int) -> None:
        key = (layer, expert_id)
        self.entries[layer].pop(expert_id, None)
        self.warm_entries[layer].pop(expert_id, None)
        self.cold_entries[layer].pop(expert_id, None)

    def _insert_hot_locked(self, layer: int, expert_id: int, entry: HotEntry) -> None:
        hot = self.entries.setdefault(layer, OrderedDict())
        old = hot.pop(expert_id, None)
        if old is not None:
            self.bytes_used -= self._hot_size(old)

        while len(hot) >= self.hot_slots and hot:
            victim_id, victim = hot.popitem(last=False)
            self.bytes_used -= self._hot_size(victim)
            warm = _fp8_quantize_entry(victim)
            self.warm_entries[layer][victim_id] = warm
            self.bytes_used += self._warm_size(warm)
            self.fp16_to_fp8 += 1

            while len(self.warm_entries[layer]) > self.warm_slots:
                warm_id, warm_entry = self.warm_entries[layer].popitem(last=False)
                self.bytes_used -= self._warm_size(warm_entry)
                cold = _q4_quantize_entry(
                    tuple(_fp8_dequantize_matrix(matrix) for matrix in warm_entry)
                )
                self.cold_entries[layer][warm_id] = cold
                self.bytes_used += self._cold_size(cold)
                self.fp8_to_q4 += 1

                while len(self.cold_entries[layer]) > self.cold_slots:
                    _cold_id, cold_entry = self.cold_entries[layer].popitem(last=False)
                    self.bytes_used -= self._cold_size(cold_entry)
                    self.q4_drops += 1
                    self.evictions += 1

        hot[expert_id] = entry
        self.bytes_used += self._hot_size(entry)

    def _budget_allows(self, layer: int, extra_bytes: int) -> bool:
        layer_capacity = self.budget_bytes // self.layers if self.layers else 0
        if layer_capacity <= 0:
            return False
        return extra_bytes <= layer_capacity

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

            warm = self.warm_entries.setdefault(layer, OrderedDict())
            compressed = warm.pop(expert_id, None)
            if compressed is not None:
                self.hits += 1
                self.fp8_hits += 1
                self.bytes_used -= self._warm_size(compressed)
                entry = tuple(_fp8_dequantize_matrix(matrix) for matrix in compressed)
                self._insert_hot_locked(layer, expert_id, entry)
                return entry

            cold = self.cold_entries.setdefault(layer, OrderedDict())
            packed = cold.pop(expert_id, None)
            if packed is not None:
                self.hits += 1
                self.q4_hits += 1
                self.bytes_used -= self._cold_size(packed)
                entry = tuple(_q4_dequantize_matrix(matrix) for matrix in packed)
                self._insert_hot_locked(layer, expert_id, entry)
                return entry

            self.misses += 1
            return None

    def put(self, layer: int, expert_id: int, entry: HotEntry):
        layer = int(layer)
        expert_id = int(expert_id)
        size = self._hot_size(entry)
        with self.lock:
            if self.hot_slots <= 0 or not self._budget_allows(layer, size):
                return False
            self._remove_key(layer, expert_id)
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
                weight.to(device="cuda", dtype=torch.float16)
                for weight in raw_weights
            )

        self.put(layer, expert_id, entry)
        return entry

    def snapshot(self) -> dict[str, int | float]:
        with self.lock:
            items = sum(
                len(bank) + len(self.warm_entries[layer]) + len(self.cold_entries[layer])
                for layer, bank in self.entries.items()
            )
            total = self.hits + self.misses
            hot_items = sum(len(bank) for bank in self.entries.values())
            warm_items = sum(len(bank) for bank in self.warm_entries.values())
            cold_items = sum(len(bank) for bank in self.cold_entries.values())
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
                "layers_populated": sum(
                    bool(self.entries[layer] or self.warm_entries[layer] or self.cold_entries[layer])
                    for layer in range(self.layers)
                ),
                "slots_per_layer": self.slots_per_layer,
                "total_slots": self.total_slots,
                "hot_slots_per_layer": self.hot_slots,
                "warm_slots_per_layer": self.warm_slots,
                "cold_slots_per_layer": self.cold_slots,
                "hot_items": hot_items,
                "warm_items": warm_items,
                "cold_items": cold_items,
                "hot_hits": self.hot_hits,
                "fp8_hits": self.fp8_hits,
                "q4_hits": self.q4_hits,
                "fp16_to_fp8": self.fp16_to_fp8,
                "fp8_to_q4": self.fp8_to_q4,
                "q4_drops": self.q4_drops,
                # Compatibility fields for the runner diagnostics.
                "shared_items": warm_items + cold_items,
                "protected_items": hot_items,
                "min_slots_per_layer": self.hot_slots,
                "shared_slots": self.warm_slots + self.cold_slots,
            }

    def clear(self) -> None:
        with self.lock:
            for layer in range(self.layers):
                self.entries[layer].clear()
                self.warm_entries[layer].clear()
                self.cold_entries[layer].clear()
            self.bytes_used = 0


def _fp8_quantize_entry(entry: HotEntry) -> WarmEntry:
    return tuple(_fp8_quantize_blockwise(weight) for weight in entry)  # type: ignore[return-value]


def _q4_quantize_entry(entry: HotEntry) -> ColdEntry:
    return tuple(_q4_quantize(weight) for weight in entry)  # type: ignore[return-value]
