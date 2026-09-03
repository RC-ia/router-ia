from __future__ import annotations

"""Persistent compressed GPU cache for Qwen3.6 routed experts.

The VRAM expert cache keeps raw FP8 experts as the main resident tier and a
smaller Q4 tier as a colder backup. FP16 is never kept persistently: when an
expert is requested, its compressed matrices are reconstructed to temporary
FP16 tensors for the existing GEMM path.
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

FP8_SLOTS_PER_LAYER = 8
Q4_SLOTS_PER_LAYER = 4
TOTAL_SLOTS_PER_LAYER = FP8_SLOTS_PER_LAYER + Q4_SLOTS_PER_LAYER

FP8Matrix = tuple[torch.Tensor, torch.Tensor]
WarmEntry = tuple[FP8Matrix, FP8Matrix, FP8Matrix]
Q4Matrix = tuple[torch.Tensor, torch.Tensor, tuple[int, int]]
ColdEntry = tuple[Q4Matrix, Q4Matrix, Q4Matrix]
FP16Entry = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _fp8_dequantize_matrix(matrix: FP8Matrix) -> torch.Tensor:
    weight, scales = matrix
    return dequant.dequantize_fp8_blockwise(weight, scales).to(dtype=torch.float16)


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


def _fp8_quantize_entry(entry: FP16Entry) -> WarmEntry:
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
    return tuple(_q4_quantize_matrix(_fp8_dequantize_matrix(m)) for m in entry)  # type: ignore[return-value]


class RoutedExpertCache:
    """Per-layer compressed expert cache: 8 FP8 + 4 Q4 slots."""

    def __init__(self, budget_bytes: int, layers: int = MODEL_LAYERS) -> None:
        self.budget_bytes = max(int(budget_bytes), 0)
        self.layers = max(int(layers), 1)
        self.slots_per_layer = min(TOTAL_SLOTS_PER_LAYER, self._budget_slots())
        self.total_slots = self.slots_per_layer * self.layers
        remaining = self.slots_per_layer
        self.fp8_slots = min(FP8_SLOTS_PER_LAYER, remaining)
        remaining -= self.fp8_slots
        self.q4_slots = min(Q4_SLOTS_PER_LAYER, remaining)

        self.fp8_entries: dict[int, OrderedDict[int, WarmEntry]] = {layer: OrderedDict() for layer in range(self.layers)}
        self.q4_entries: dict[int, OrderedDict[int, ColdEntry]] = {layer: OrderedDict() for layer in range(self.layers)}
        self.entry_bytes: dict[tuple[int, int, str], int] = {}
        self.bytes_used = 0
        self.hits = 0
        self.misses = 0
        self.loads = 0
        self.evictions = 0
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
        target_per_layer = (
            FP8_SLOTS_PER_LAYER * (FP16_EXPERT_BYTES_ESTIMATE // 2)
            + Q4_SLOTS_PER_LAYER * (FP16_EXPERT_BYTES_ESTIMATE // 4)
        )
        return TOTAL_SLOTS_PER_LAYER if self.budget_bytes // max(target_per_layer, 1) >= self.layers else 0

    @staticmethod
    def _fp8_size(entry: WarmEntry) -> int:
        return sum(int(w.numel()) * int(w.element_size()) + int(s.numel()) * int(s.element_size()) for w, s in entry)

    @staticmethod
    def _q4_size(entry: ColdEntry) -> int:
        return sum(int(p.numel()) * int(p.element_size()) + int(s.numel()) * int(s.element_size()) for p, s, _ in entry)

    def _record(self, layer: int, expert_id: int, tier: str, size: int) -> None:
        self.entry_bytes[(layer, expert_id, tier)] = size
        self.bytes_used += size

    def _erase(self, layer: int, expert_id: int, tier: str) -> None:
        self.bytes_used -= self.entry_bytes.pop((layer, expert_id, tier), 0)

    @staticmethod
    def _stream_key(proj: str, kind: str) -> str:
        return f"{proj}.{kind}.__expert_prefetch__"

    def _raw_projection_for_gpu(self, store, proj: str):
        if hasattr(store, "vram_cache"):
            wk = self._stream_key(proj, "weight")
            sk = self._stream_key(proj, "scale")
            w = store.vram_cache.get_stream(wk)
            s = store.vram_cache.get_stream(sk)
            if w is not None and s is not None:
                self.stream_prefetch_hits += 1
                return w, s
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

    def prefetch_expert_raw(self, store, layer_prefix: str, expert_id: int) -> None:
        prefix = f"{layer_prefix}mlp.experts.{int(expert_id)}"
        for name in ("gate_proj", "up_proj", "down_proj"):
            self._raw_projection_for_gpu(store, prefix + "." + name)

    def get(self, layer: int, expert_id: int):
        layer = int(layer)
        expert_id = int(expert_id)
        with self.lock:
            fp8 = self.fp8_entries.setdefault(layer, OrderedDict())
            entry = fp8.get(expert_id)
            if entry is not None:
                self.hits += 1
                self.fp8_hits += 1
                fp8.move_to_end(expert_id)
                hit_entry = entry
            else:
                q4 = self.q4_entries.setdefault(layer, OrderedDict())
                entry_q4 = q4.get(expert_id)
                if entry_q4 is not None:
                    self.hits += 1
                    self.q4_hits += 1
                    q4.move_to_end(expert_id)
                    hit_entry = entry_q4
                    entry = None
                else:
                    self.misses += 1
                    return None

        # Do not hold the cache lock while performing the expensive CUDA
        # dequantization. Multiple routed experts can now dequantize in parallel.
        if entry is not None:
            return tuple(_fp8_dequantize_matrix(m) for m in entry)
        return tuple(_q4_dequantize_matrix(m) for m in hit_entry)

    def _insert_fp8_locked(self, layer: int, expert_id: int, entry: WarmEntry) -> None:
        bank = self.fp8_entries.setdefault(layer, OrderedDict())
        if expert_id in bank:
            self._erase(layer, expert_id, "fp8")
            bank.pop(expert_id, None)
        bank[expert_id] = entry
        self._record(layer, expert_id, "fp8", self._fp8_size(entry))
        bank.move_to_end(expert_id)

        while len(bank) > self.fp8_slots:
            victim_id, victim = bank.popitem(last=False)
            self._erase(layer, victim_id, "fp8")
            if self.q4_slots > 0:
                cold = _q4_quantize_entry_from_fp8(victim)
                q4 = self.q4_entries.setdefault(layer, OrderedDict())
                old_q4 = q4.pop(victim_id, None)
                if old_q4 is not None:
                    self._erase(layer, victim_id, "q4")
                q4[victim_id] = cold
                self._record(layer, victim_id, "q4", self._q4_size(cold))
                self.fp8_to_q4 += 1
                while len(q4) > self.q4_slots:
                    dropped_id, _ = q4.popitem(last=False)
                    self._erase(layer, dropped_id, "q4")
                    self.q4_drops += 1
                    self.evictions += 1
            else:
                self.evictions += 1

    def put_fp16(self, layer: int, expert_id: int, entry: FP16Entry) -> None:
        # Quantization is intentionally GPU-only. Never run FP16→FP8 on CPU.
        if any(t.device.type != "cuda" for t in entry):
            entry = tuple(t.to(device="cuda", dtype=torch.float16) for t in entry)  # type: ignore[assignment]
        compact = _fp8_quantize_entry(entry)
        with self.lock:
            self._insert_fp8_locked(int(layer), int(expert_id), compact)
            self.loads += 1
            self.fp16_to_fp8 += 1

    def get_or_load(self, store, layer: int, expert_id: int, layer_prefix: str):
        hit = self.get(layer, expert_id)
        if hit is not None:
            return hit

        expert_prefix = f"{layer_prefix}mlp.experts.{int(expert_id)}"
        raw_weights = []
        raw_scales = []
        for name in ("gate_proj", "up_proj", "down_proj"):
            w, s = self._raw_projection_for_gpu(store, expert_prefix + "." + name)
            raw_weights.append(w)
            raw_scales.append(s)

        if all(w.dtype == torch.float8_e4m3fn for w in raw_weights):
            entry_fp8: WarmEntry = (
                (raw_weights[0], raw_scales[0]),
                (raw_weights[1], raw_scales[1]),
                (raw_weights[2], raw_scales[2]),
            )
            with self.lock:
                self._insert_fp8_locked(int(layer), int(expert_id), entry_fp8)
                self.loads += 1
            return tuple(_fp8_dequantize_matrix(m) for m in entry_fp8)

        entry_fp16: FP16Entry = tuple(w.to(device="cuda", dtype=torch.float16) for w in raw_weights)  # type: ignore[assignment]
        self.put_fp16(layer, expert_id, entry_fp16)
        return entry_fp16

    def snapshot(self) -> dict[str, int | float]:
        with self.lock:
            total = self.hits + self.misses
            fp8_items = sum(len(b) for b in self.fp8_entries.values())
            q4_items = sum(len(b) for b in self.q4_entries.values())
            return {
                "items": fp8_items + q4_items,
                "bytes": self.bytes_used,
                "budget_bytes": self.budget_bytes,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / total * 100.0 if total else 0.0,
                "loads": self.loads,
                "evictions": self.evictions,
                "layers": self.layers,
                "layers_populated": sum(bool(self.fp8_entries[l] or self.q4_entries[l]) for l in range(self.layers)),
                "slots_per_layer": self.slots_per_layer,
                "total_slots": self.total_slots,
                "hot_slots_per_layer": 0,
                "warm_slots_per_layer": self.fp8_slots,
                "cold_slots_per_layer": self.q4_slots,
                "hot_items": 0,
                "warm_items": fp8_items,
                "cold_items": q4_items,
                "hot_hits": 0,
                "fp8_hits": self.fp8_hits,
                "q4_hits": self.q4_hits,
                "fp16_to_fp8": self.fp16_to_fp8,
                "fp8_to_q4": self.fp8_to_q4,
                "q4_drops": self.q4_drops,
                "stream_prefetch_hits": self.stream_prefetch_hits,
                "stream_prefetch_misses": self.stream_prefetch_misses,
                "shared_items": q4_items,
                "protected_items": fp8_items,
                "min_slots_per_layer": self.fp8_slots,
                "shared_slots": self.q4_slots,
            }

    def clear(self) -> None:
        with self.lock:
            for layer in range(self.layers):
                self.fp8_entries[layer].clear()
                self.q4_entries[layer].clear()
            self.entry_bytes.clear()
            self.bytes_used = 0
