from __future__ import annotations

"""Persistent FP8 GPU cache with Q4 RAM backing for Qwen3.6 routed experts."""

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
# Q4 is a RAM backing tier now. Keep it bounded so it does not consume the
# model/runtime RAM budget indefinitely.
Q4_SLOTS_PER_LAYER = 3
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


def _q4_dequantize_matrix(matrix: Q4Matrix, device: str = "cuda") -> torch.Tensor:
    packed, scale, shape = matrix
    if device == "cuda":
        packed = packed.to(device="cuda", non_blocking=True)
        scale = scale.to(device="cuda", non_blocking=True)
    low = (packed & 0x0F).to(torch.int16) - 8
    high = ((packed >> 4) & 0x0F).to(torch.int16) - 8
    q = torch.stack((low, high), dim=1).reshape(-1)[: shape[0] * shape[1]]
    return (q.float() * scale.float()).reshape(shape).to(torch.float16)


def _q4_dequantize_entry_batch(entries: list[ColdEntry], projection: int) -> list[torch.Tensor]:
    if not entries:
        return []
    packed = torch.stack([entry[projection][0] for entry in entries], dim=0).to(device="cuda", non_blocking=True)
    scales = torch.stack([entry[projection][1] for entry in entries], dim=0).to(device="cuda", non_blocking=True)
    shapes = [entry[projection][2] for entry in entries]
    rows, cols = shapes[0]
    if any(shape != (rows, cols) for shape in shapes):
        return [_q4_dequantize_matrix(entry[projection]) for entry in entries]
    low = (packed & 0x0F).to(torch.int16) - 8
    high = ((packed >> 4) & 0x0F).to(torch.int16) - 8
    q = torch.stack((low, high), dim=2).reshape(len(entries), -1)[:, : rows * cols]
    return (q.float() * scales.float().reshape(len(entries), 1)).reshape(len(entries), rows, cols).to(torch.float16)


def _q4_quantize_entry_from_fp8(entry: WarmEntry) -> ColdEntry:
    # The input is already on CUDA. Quantization therefore stays on the GPU;
    # only the compressed Q4 result is copied to host RAM afterward.
    return tuple(_q4_quantize_matrix(_fp8_dequantize_matrix(m)) for m in entry)  # type: ignore[return-value]


def _move_q4_to_cpu(entry: ColdEntry) -> ColdEntry:
    return tuple(
        (packed.detach().to(device="cpu"), scale.detach().to(device="cpu"), shape)
        for packed, scale, shape in entry
    )  # type: ignore[return-value]


class RoutedExpertCache:
    """Per-layer FP8 GPU cache with a colder Q4 backing tier in system RAM."""

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
        self.q4_ram_bytes: dict[tuple[int, int], int] = {}
        self.bytes_used = 0
        self.q4_bytes_used = 0
        self.hits = 0
        self.misses = 0
        self.loads = 0
        self.evictions = 0
        self.fp8_hits = 0
        self.q4_hits = 0
        self.fp16_to_fp8 = 0
        self.fp8_to_q4 = 0
        self.q4_drops = 0
        self.q4_ram_evictions = 0
        self.stream_prefetch_hits = 0
        self.stream_prefetch_misses = 0
        self.lock = Lock()

    def _budget_slots(self) -> int:
        if not self.budget_bytes:
            return 0
        target_per_layer = FP8_SLOTS_PER_LAYER * (FP16_EXPERT_BYTES_ESTIMATE // 2)
        return FP8_SLOTS_PER_LAYER if self.budget_bytes // max(target_per_layer, 1) >= self.layers else 0

    @staticmethod
    def _fp8_size(entry: WarmEntry) -> int:
        return sum(int(w.numel()) * int(w.element_size()) + int(s.numel()) * int(s.element_size()) for w, s in entry)

    @staticmethod
    def _q4_size(entry: ColdEntry) -> int:
        return sum(int(p.numel()) * int(p.element_size()) + int(s.numel()) * int(s.element_size()) for p, s, _ in entry)

    def _record(self, layer: int, expert_id: int, tier: str, size: int) -> None:
        if tier == "q4":
            self.q4_ram_bytes[(layer, expert_id)] = size
            self.q4_bytes_used += size
        else:
            self.entry_bytes[(layer, expert_id, tier)] = size
            self.bytes_used += size

    def _erase(self, layer: int, expert_id: int, tier: str) -> None:
        if tier == "q4":
            self.q4_bytes_used -= self.q4_ram_bytes.pop((layer, expert_id), 0)
        else:
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

    def _lookup_batch_locked(self, layer: int, expert_ids: list[int]):
        found: list[tuple[str | None, WarmEntry | ColdEntry | None]] = []
        for expert_id in expert_ids:
            fp8 = self.fp8_entries.setdefault(layer, OrderedDict())
            entry = fp8.get(expert_id)
            if entry is not None:
                self.hits += 1
                self.fp8_hits += 1
                fp8.move_to_end(expert_id)
                found.append(("fp8", entry))
                continue
            q4 = self.q4_entries.setdefault(layer, OrderedDict())
            entry_q4 = q4.get(expert_id)
            if entry_q4 is not None:
                self.hits += 1
                self.q4_hits += 1
                q4.move_to_end(expert_id)
                found.append(("q4", entry_q4))
                continue
            self.misses += 1
            found.append((None, None))
        return found

    def _decode_found_batch(self, found):
        fp8_positions = [i for i, (tier, _) in enumerate(found) if tier == "fp8"]
        q4_positions = [i for i, (tier, _) in enumerate(found) if tier == "q4"]
        result: list[list[torch.Tensor | None] | None] = [None] * len(found)

        if fp8_positions:
            for projection in range(3):
                weights = torch.stack([found[i][1][projection][0] for i in fp8_positions], dim=0)  # type: ignore[index]
                scales = torch.stack([found[i][1][projection][1] for i in fp8_positions], dim=0)  # type: ignore[index]
                decoded = dequant.dequantize_fp8_blockwise_batch(weights, scales).to(dtype=torch.float16)
                for local, position in enumerate(fp8_positions):
                    if result[position] is None:
                        result[position] = [None, None, None]
                    result[position][projection] = decoded[local]

        if q4_positions:
            for projection in range(3):
                decoded = _q4_dequantize_entry_batch([found[i][1] for i in q4_positions], projection)  # type: ignore[list-item]
                for local, position in enumerate(q4_positions):
                    if result[position] is None:
                        result[position] = [None, None, None]
                    result[position][projection] = decoded[local]

        return [tuple(item) for item in result]  # type: ignore[arg-type]

    def get(self, layer: int, expert_id: int):
        results = self.get_batch(layer, [expert_id])
        return results[0] if results else None

    def get_batch(self, layer: int, expert_ids: list[int]) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        layer = int(layer)
        ids = [int(x) for x in expert_ids]
        with self.lock:
            found = self._lookup_batch_locked(layer, ids)
        return self._decode_found_batch(found)

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
                # Quantize while the victim is still on CUDA, then place only
                # the compressed Q4 representation in host RAM.
                cold_gpu = _q4_quantize_entry_from_fp8(victim)
                cold = _move_q4_to_cpu(cold_gpu)
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
                    self.q4_ram_evictions += 1
            else:
                self.evictions += 1

    def _insert_fp8_batch_locked(
        self,
        layer: int,
        entries: dict[int, WarmEntry],
    ) -> None:
        """Insert a routed batch while materializing only Q4 victims that survive.

        A top-k route can replace most of an 8-slot FP8 bank at once. The old
        per-entry insertion path converted every evicted expert to Q4 and then
        immediately discarded most of those conversions because the cold tier
        has only three slots. Batch insertion computes the final cold residency
        first, so only Q4 entries that can actually survive are materialized.
        """
        if not entries:
            return

        layer = int(layer)
        bank = self.fp8_entries.setdefault(layer, OrderedDict())
        ids = list(dict.fromkeys(int(x) for x in entries))

        existing = [expert_id for expert_id in ids if expert_id in bank]
        for expert_id in existing:
            self._erase(layer, expert_id, "fp8")
            bank.pop(expert_id, None)

        overflow = max(len(bank) + len(ids) - self.fp8_slots, 0)
        victims: list[tuple[int, WarmEntry]] = []
        for _ in range(overflow):
            victim_id, victim = bank.popitem(last=False)
            self._erase(layer, victim_id, "fp8")
            victims.append((victim_id, victim))

        q4 = self.q4_entries.setdefault(layer, OrderedDict())
        if victims and self.q4_slots > 0:
            final_q4_order = list(q4.keys())
            for victim_id, _victim in victims:
                if victim_id in final_q4_order:
                    final_q4_order.remove(victim_id)
                final_q4_order.append(victim_id)
                if len(final_q4_order) > self.q4_slots:
                    final_q4_order.pop(0)
            surviving_victims = set(final_q4_order).intersection(victim_id for victim_id, _ in victims)
        else:
            surviving_victims = set()

        if victims:
            # Victims can already have a stale Q4 copy while resident in FP8.
            # Remove those copies before installing the final surviving cold set.
            for victim_id, _victim in victims:
                old_q4 = q4.pop(victim_id, None)
                if old_q4 is not None:
                    self._erase(layer, victim_id, "q4")

        if self.q4_slots > 0:
            for victim_id, victim in victims:
                if victim_id not in surviving_victims:
                    continue
                cold_gpu = _q4_quantize_entry_from_fp8(victim)
                cold = _move_q4_to_cpu(cold_gpu)
                q4[victim_id] = cold
                self._record(layer, victim_id, "q4", self._q4_size(cold))
                self.fp8_to_q4 += 1
                while len(q4) > self.q4_slots:
                    dropped_id, _ = q4.popitem(last=False)
                    self._erase(layer, dropped_id, "q4")
                    self.q4_drops += 1
                    self.q4_ram_evictions += 1
        elif victims:
            self.evictions += len(victims)

        for expert_id in ids:
            bank[expert_id] = entries[expert_id]
            self._record(layer, expert_id, "fp8", self._fp8_size(entries[expert_id]))
            bank.move_to_end(expert_id)

        while len(bank) > self.fp8_slots:
            # This is only defensive: the routed top-k batch is expected to
            # fit the FP8 bank, but preserve the old correctness invariant if a
            # caller supplies a larger batch.
            victim_id, victim = bank.popitem(last=False)
            self._erase(layer, victim_id, "fp8")
            if self.q4_slots > 0:
                cold_gpu = _q4_quantize_entry_from_fp8(victim)
                cold = _move_q4_to_cpu(cold_gpu)
                q4[victim_id] = cold
                self._record(layer, victim_id, "q4", self._q4_size(cold))
                self.fp8_to_q4 += 1
            else:
                self.evictions += 1

            while len(q4) > self.q4_slots:
                dropped_id, _ = q4.popitem(last=False)
                self._erase(layer, dropped_id, "q4")
                self.q4_drops += 1
                self.q4_ram_evictions += 1

    def put_fp16(self, layer: int, expert_id: int, entry: FP16Entry) -> None:
        if any(t.device.type != "cuda" for t in entry):
            entry = tuple(t.to(device="cuda", dtype=torch.float16) for t in entry)  # type: ignore[assignment]
        compact = _fp8_quantize_entry(entry)
        with self.lock:
            self._insert_fp8_locked(int(layer), int(expert_id), compact)
            self.loads += 1
            self.fp16_to_fp8 += 1

    def get_or_load(self, store, layer: int, expert_id: int, layer_prefix: str):
        return self.get_or_load_batch(store, layer, [expert_id], layer_prefix)[0]

    def get_or_load_batch(self, store, layer: int, expert_ids: list[int], layer_prefix: str) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Load misses to GPU-compressed storage, then dequantize the full route in a batch."""
        layer = int(layer)
        ids = [int(x) for x in expert_ids]
        misses: list[int] = []
        with self.lock:
            for expert_id in ids:
                fp8 = self.fp8_entries.setdefault(layer, OrderedDict())
                q4 = self.q4_entries.setdefault(layer, OrderedDict())
                if expert_id not in fp8 and expert_id not in q4:
                    misses.append(expert_id)

        loaded: dict[int, WarmEntry | None] = {}
        for expert_id in misses:
            expert_prefix = f"{layer_prefix}mlp.experts.{expert_id}"
            raw_weights, raw_scales = [], []
            for name in ("gate_proj", "up_proj", "down_proj"):
                w, s = self._raw_projection_for_gpu(store, expert_prefix + "." + name)
                raw_weights.append(w)
                raw_scales.append(s)
            raw_is_fp8 = all(w.dtype == torch.float8_e4m3fn for w in raw_weights)
            if raw_is_fp8:
                compact: WarmEntry = (
                    (raw_weights[0], raw_scales[0]),
                    (raw_weights[1], raw_scales[1]),
                    (raw_weights[2], raw_scales[2]),
                )
            else:
                fp16 = tuple(w.to(device="cuda", dtype=torch.float16) for w in raw_weights)
                compact = _fp8_quantize_entry(fp16)  # type: ignore[arg-type]
                self.fp16_to_fp8 += 1
            loaded[expert_id] = compact

        with self.lock:
            for expert_id, compact in loaded.items():
                if compact is not None:
                    self._insert_fp8_locked(layer, expert_id, compact)
                    self.loads += 1

            found = self._lookup_batch_locked(layer, ids)

        decoded = self._decode_found_batch(found)
        if len(decoded) != len(ids):
            raise RuntimeError(f"Expert batch integrity error: requested {len(ids)}, decoded {len(decoded)}")
        return decoded

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
                "q4_ram_evictions": self.q4_ram_evictions,
                "q4_ram_bytes": self.q4_bytes_used,
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
            self.q4_ram_bytes.clear()
            self.bytes_used = 0
            self.q4_bytes_used = 0
