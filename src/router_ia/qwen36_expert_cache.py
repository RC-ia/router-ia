from __future__ import annotations

"""Persistent three-tier GPU cache for complete Qwen3.6 routed experts.

Hot experts stay in FP16. Less-used experts are compressed to block-scaled
FP8, then to packed symmetric Q4 before finally being dropped. The three tiers
are per-layer LRU banks so a sequential 40-layer decode does not cause
cross-layer cache thrashing.
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
Q4_SLOTS_PER_LAYER = 4
TOTAL_SLOTS_PER_LAYER = HOT_SLOTS_PER_LAYER + FP8_SLOTS_PER_LAYER + Q4_SLOTS_PER_LAYER

FP8Matrix = tuple[torch.Tensor, torch.Tensor]
HotEntry = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
WarmEntry = tuple[FP8Matrix, FP8Matrix, FP8Matrix]
Q4Matrix = tuple[torch.Tensor, torch.Tensor, tuple[int, int]]
ColdEntry = tuple[Q4Matrix, Q4Matrix, Q4Matrix]


def _fp8_quantize_blockwise(weight: torch.Tensor) -> FP8Matrix:
    if weight.ndim != 2:
        raise ValueError(f"Expected 2-D matrix, got {tuple(weight.shape)}")
    rows, cols = map(int, weight.shape)
    padded_rows = (rows + BLOCK - 1) // BLOCK * BLOCK
    padded_cols = (cols + BLOCK - 1) // BLOCK * BLOCK
    padded = torch.zeros((padded_rows, padded_cols), device=weight.device, dtype=torch.float32)
    padded[:rows, :cols] = weight.float()
    blocks = padded.reshape(padded_rows // BLOCK, BLOCK, padded_cols // BLOCK, BLOCK)
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


def _q4_quantize_matrix(weight: torch.Tensor) -> Q4Matrix:
    if weight.ndim != 2:
        raise ValueError(f"Expected 2-D matrix, got {tuple(weight.shape)}")
    rows, cols = map(int, weight.shape)
    x = weight.float()
    scale = torch.clamp(x.abs().amax() / 7.0, min=torch.finfo(torch.float32).tiny)
    q = torch.round(x / scale).clamp(-7, 7).to(torch.int16) + 8
    flat = q.reshape(-1)
    if flat.numel() & 1:
        flat = torch.cat([flat, torch.full((1,), 8, device=flat.device, dtype=flat.dtype)])
    packed = flat[0::2].to(torch.uint8) | (flat[1::2].to(torch.uint8) << 4)
    return packed, scale.to(dtype=torch.float16), (rows, cols)


def _q4_dequantize_matrix(matrix: Q4Matrix) -> torch.Tensor:
    packed, scale, shape = matrix
    low = (packed & 0x0F).to(torch.int16) - 8
    high = ((packed >> 4) & 0x0F).to(torch.int16) - 8
    q = torch.stack((low, high), dim=1).reshape(-1)[: shape[0] * shape[1]]
    return (q.float() * scale.float()).reshape(shape).to(torch.float16)


def _q4_quantize_entry_from_fp8(entry: WarmEntry) -> ColdEntry:
    return tuple(_q4_quantize_matrix(_fp8_dequantize_matrix(matrix)) for matrix in entry)  # type: ignore[return-value]


def _q4_quantize_entry(entry: HotEntry) -> ColdEntry:
    return tuple(_q4_quantize_matrix(t) for t in entry)  # type: ignore[return-value]


class RoutedExpertCache:
    """Per-layer LRU cache: 2 FP16 hot, 4 FP8 warm, 4 Q4 cold."""

    def __init__(self, budget_bytes: int, layers: int = MODEL_LAYERS) -> None:
        self.budget_bytes = max(int(budget_bytes), 0)
        self.layers = max(int(layers), 1)
        self.slots_per_layer = min(TOTAL_SLOTS_PER_LAYER, self._budget_slots())
        self.total_slots = self.slots_per_layer * self.layers

        remaining = self.slots_per_layer
        self.hot_slots = min(HOT_SLOTS_PER_LAYER, remaining)
        remaining -= self.hot_slots
        self.fp8_slots = min(FP8_SLOTS_PER_LAYER, remaining)
        remaining -= self.fp8_slots
        self.q4_slots = min(Q4_SLOTS_PER_LAYER, remaining)

        self.entries: dict[int, OrderedDict[int, HotEntry]] = {layer: OrderedDict() for layer in range(self.layers)}
        self.fp8_entries: dict[int, OrderedDict[int, WarmEntry]] = {layer: OrderedDict() for layer in range(self.layers)}
        self.q4_entries: dict[int, OrderedDict[int, ColdEntry]] = {layer: OrderedDict() for layer in range(self.layers)}
        self.entry_bytes: dict[tuple[int, int, str], int] = {}
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
        self.stream_prefetch_hits = 0
        self.stream_prefetch_misses = 0
        self.lock = Lock()

    def _budget_slots(self) -> int:
        if not self.budget_bytes:
            return 0
        # Target footprint per layer is approximately:
        # 2*6 MiB + 4*3 MiB + 4*1.5 MiB = 24 MiB.
        target_per_layer = (
            HOT_SLOTS_PER_LAYER * FP16_EXPERT_BYTES_ESTIMATE
            + FP8_SLOTS_PER_LAYER * (FP16_EXPERT_BYTES_ESTIMATE // 2)
            + Q4_SLOTS_PER_LAYER * (FP16_EXPERT_BYTES_ESTIMATE // 4)
        )
        supported_layers = self.budget_bytes // max(target_per_layer, 1)
        if supported_layers >= self.layers:
            return TOTAL_SLOTS_PER_LAYER
        fp16_capacity = self.budget_bytes // FP16_EXPERT_BYTES_ESTIMATE // max(self.layers, 1)
        return min(HOT_SLOTS_PER_LAYER, fp16_capacity)

    @staticmethod
    def _hot_size(entry: HotEntry) -> int:
        return sum(int(t.numel()) * int(t.element_size()) for t in entry)

    @staticmethod
    def _fp8_size(entry: WarmEntry) -> int:
        return sum(
            int(weight.numel()) * int(weight.element_size())
            + int(scales.numel()) * int(scales.element_size())
            for weight, scales in entry
        )

    @staticmethod
    def _q4_size(entry: ColdEntry) -> int:
        return sum(
            int(packed.numel()) * int(packed.element_size())
            + int(scale.numel()) * int(scale.element_size())
            for packed, scale, _shape in entry
        )

    def _record(self, layer: int, expert_id: int, tier: str, size: int) -> None:
        self.entry_bytes[(layer, expert_id, tier)] = size
        self.bytes_used += size

    def _erase_bytes(self, layer: int, expert_id: int, tier: str) -> None:
        self.bytes_used -= self.entry_bytes.pop((layer, expert_id, tier), 0)

    @staticmethod
    def _stream_key(proj: str, kind: str) -> str:
        return f"{proj}.{kind}.__expert_prefetch__"

    def _raw_projection_for_gpu(self, store, proj: str):
        if hasattr(store, "vram_cache"):
            weight_key = self._stream_key(proj, "weight")
            scale_key = self._stream_key(proj, "scale")
            streamed_weight = store.vram_cache.get_stream(weight_key)
            streamed_scale = store.vram_cache.get_stream(scale_key)
            if streamed_weight is not None and streamed_scale is not None:
                self.stream_prefetch_hits += 1
                return streamed_weight, streamed_scale
            self.stream_prefetch_misses += 1

        weight = store.load(proj + ".weight", device="cpu")
        scale = store.load(proj + ".weight_scale_inv", device="cpu")
        if weight.dtype == torch.float8_e4m3fn and hasattr(store, "vram_cache"):
            gpu_weight = weight.to(device="cuda")
            gpu_scale = scale.to(device="cuda")
            store.vram_cache.put_stream(self._stream_key(proj, "weight"), gpu_weight)
            store.vram_cache.put_stream(self._stream_key(proj, "scale"), gpu_scale)
            return gpu_weight, gpu_scale
        return weight, scale

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
                return tuple(_fp8_dequantize_matrix(m) for m in compact)

            q4_bank = self.q4_entries.setdefault(layer, OrderedDict())
            packed = q4_bank.get(expert_id)
            if packed is not None:
                self.hits += 1
                self.q4_hits += 1
                q4_bank.move_to_end(expert_id)
                return tuple(_q4_dequantize_matrix(m) for m in packed)

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
                compact = _fp8_quantize_entry(victim)
                fp8_bank = self.fp8_entries[layer]
                previous = fp8_bank.pop(victim_id, None)
                if previous is not None:
                    self._erase_bytes(layer, victim_id, "fp8")
                fp8_bank[victim_id] = compact
                self._record(layer, victim_id, "fp8", self._fp8_size(compact))
                self.fp16_to_fp8 += 1

                while len(fp8_bank) > self.fp8_slots:
                    cold_id, cold_fp8 = fp8_bank.popitem(last=False)
                    self._erase_bytes(layer, cold_id, "fp8")
                    if self.q4_slots > 0:
                        cold = _q4_quantize_entry_from_fp8(cold_fp8)
                        q4_bank = self.q4_entries[layer]
                        previous_q4 = q4_bank.pop(cold_id, None)
                        if previous_q4 is not None:
                            self._erase_bytes(layer, cold_id, "q4")
                        q4_bank[cold_id] = cold
                        self._record(layer, cold_id, "q4", self._q4_size(cold))
                        self.fp8_to_q4 += 1
                        while len(q4_bank) > self.q4_slots:
                            dropped_id, _ = q4_bank.popitem(last=False)
                            self._erase_bytes(layer, dropped_id, "q4")
                            self.q4_drops += 1
                            self.evictions += 1
                    else:
                        self.fp8_drops += 1
                        self.evictions += 1
            else:
                self.evictions += 1

        # A newly hot expert supersedes any compressed copy.
        if expert_id in self.fp8_entries[layer]:
            self.fp8_entries[layer].pop(expert_id, None)
            self._erase_bytes(layer, expert_id, "fp8")
        if expert_id in self.q4_entries[layer]:
            self.q4_entries[layer].pop(expert_id, None)
            self._erase_bytes(layer, expert_id, "q4")

        hot[expert_id] = entry
        self._record(layer, expert_id, "fp16", self._hot_size(entry))

    def put(self, layer: int, expert_id: int, entry: HotEntry):
        with self.lock:
            if self.hot_slots <= 0:
                return False
            self._insert_hot_locked(int(layer), int(expert_id), entry)
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
            raw_weight, raw_scale = self._raw_projection_for_gpu(
                store, expert_prefix + "." + name
            )
            raw_weights.append(raw_weight)
            raw_scales.append(raw_scale)

        if all(weight.dtype == torch.float8_e4m3fn for weight in raw_weights):
            gate_up_weights = torch.stack(raw_weights[:2], dim=0)
            gate_up_scales = torch.stack(raw_scales[:2], dim=0)
            gate_up_batch = dequant.dequantize_fp8_blockwise_batch(
                gate_up_weights, gate_up_scales
            ).to(dtype=torch.float16)
            down_output = dequant.dequantize_fp8_blockwise(
                raw_weights[2], raw_scales[2]
            ).to(dtype=torch.float16)
            entry = (gate_up_batch[0], gate_up_batch[1], down_output)
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
            hot_items = sum(len(bank) for bank in self.entries.values())
            fp8_items = sum(len(bank) for bank in self.fp8_entries.values())
            q4_items = sum(len(bank) for bank in self.q4_entries.values())
            return {
                "items": hot_items + fp8_items + q4_items,
                "bytes": self.bytes_used,
                "budget_bytes": self.budget_bytes,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / total * 100.0 if total else 0.0,
                "loads": self.loads,
                "evictions": self.evictions,
                "layers": self.layers,
                "layers_populated": sum(
                    bool(self.entries[layer] or self.fp8_entries[layer] or self.q4_entries[layer])
                    for layer in range(self.layers)
                ),
                "slots_per_layer": self.slots_per_layer,
                "total_slots": self.total_slots,
                "hot_slots_per_layer": self.hot_slots,
                "warm_slots_per_layer": self.fp8_slots,
                "cold_slots_per_layer": self.q4_slots,
                "hot_items": hot_items,
                "warm_items": fp8_items,
                "cold_items": q4_items,
                "hot_hits": self.hot_hits,
                "fp8_hits": self.fp8_hits,
                "q4_hits": self.q4_hits,
                "fp16_to_fp8": self.fp16_to_fp8,
                "fp8_to_q4": self.fp8_to_q4,
                "q4_drops": self.q4_drops,
                "stream_prefetch_hits": self.stream_prefetch_hits,
                "stream_prefetch_misses": self.stream_prefetch_misses,
                "shared_items": fp8_items + q4_items,
                "protected_items": hot_items,
                "min_slots_per_layer": self.hot_slots,
                "shared_slots": self.fp8_slots + self.q4_slots,
            }

    def clear(self) -> None:
        with self.lock:
            for layer in range(self.layers):
                self.entries[layer].clear()
                self.fp8_entries[layer].clear()
                self.q4_entries[layer].clear()
            self.entry_bytes.clear()
            self.bytes_used = 0
