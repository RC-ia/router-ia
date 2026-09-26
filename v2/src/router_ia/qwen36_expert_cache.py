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
Q4_MIN_SLOTS_PER_LAYER = 3
PREDICTED_RAM_SLOTS_PER_LAYER = 4
FP8_EXPERT_BYTES_ESTIMATE = 3 * EXPERT_HIDDEN * HIDDEN + 4096
Q4_EXPERT_BYTES_ESTIMATE = 3 * ((EXPERT_HIDDEN * HIDDEN + 1) // 2) + 16

FP8Matrix = tuple[torch.Tensor, torch.Tensor]
WarmEntry = tuple[FP8Matrix, FP8Matrix, FP8Matrix]
Q4Matrix = tuple[torch.Tensor, torch.Tensor, tuple[int, int]]
ColdEntry = tuple[Q4Matrix, Q4Matrix, Q4Matrix]
FP16Entry = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
RamFP8Entry = tuple[FP8Matrix, FP8Matrix, FP8Matrix]


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
        self.fp8_slots = min(FP8_SLOTS_PER_LAYER, self._budget_fp8_slots())
        self.q4_slots = self._budget_q4_slots()
        self.slots_per_layer = self.fp8_slots + self.q4_slots
        self.total_slots = self.slots_per_layer * self.layers

        self.fp8_entries: dict[int, OrderedDict[int, WarmEntry]] = {layer: OrderedDict() for layer in range(self.layers)}
        self.q4_entries: dict[int, OrderedDict[int, ColdEntry]] = {layer: OrderedDict() for layer in range(self.layers)}
        self.predicted_ram_entries: dict[int, OrderedDict[int, RamFP8Entry]] = {layer: OrderedDict() for layer in range(self.layers)}
        self.predicted_ram_bytes: dict[tuple[int, int], int] = {}
        self.predicted_ram_bytes_used = 0
        self.predicted_ram_prefetches = 0
        self.predicted_ram_hits = 0
        self.predicted_ram_promotions = 0
        self.predicted_ram_drops = 0
        self.q4_promotions = 0
        self.q4_promotion_pending: set[tuple[int, int]] = set()
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

    def _budget_fp8_slots(self) -> int:
        if not self.budget_bytes:
            return 0
        return FP8_SLOTS_PER_LAYER if self.budget_bytes >= self.layers * FP8_EXPERT_BYTES_ESTIMATE else 0

    def _budget_q4_slots(self) -> int:
        if not self.budget_bytes or not self.layers:
            return 0
        predicted_bytes = (
            self.layers
            * PREDICTED_RAM_SLOTS_PER_LAYER
            * FP8_EXPERT_BYTES_ESTIMATE
        )
        remaining = max(self.budget_bytes - predicted_bytes, 0)
        per_layer = remaining // self.layers
        slots = int(per_layer // max(Q4_EXPERT_BYTES_ESTIMATE, 1))
        if slots <= 0:
            return 0
        return min(EXPERTS_PER_LAYER, max(Q4_MIN_SLOTS_PER_LAYER, slots))

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

    def _record_predicted_ram(self, layer: int, expert_id: int, entry: RamFP8Entry) -> None:
        key = (int(layer), int(expert_id))
        old = self.predicted_ram_bytes.pop(key, 0)
        self.predicted_ram_bytes_used -= old
        size = self._fp8_size(entry)
        self.predicted_ram_bytes[key] = size
        self.predicted_ram_bytes_used += size

    def _erase_predicted_ram(self, layer: int, expert_id: int) -> None:
        key = (int(layer), int(expert_id))
        self.predicted_ram_bytes_used -= self.predicted_ram_bytes.pop(key, 0)
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

    def prefetch_expert_to_ram(self, store, layer: int, expert_id: int, layer_prefix: str) -> bool:
        """Load a predicted expert into host RAM as FP8 without touching VRAM."""
        layer = int(layer)
        expert_id = int(expert_id)
        with self.lock:
            fp8_bank = self.fp8_entries.setdefault(layer, OrderedDict())
            predicted_bank = self.predicted_ram_entries.setdefault(layer, OrderedDict())
            q4_bank = self.q4_entries.setdefault(layer, OrderedDict())
            if expert_id in fp8_bank or expert_id in predicted_bank or expert_id in q4_bank:
                if expert_id in predicted_bank:
                    predicted_bank.move_to_end(expert_id)
                return False

        prefix = f"{layer_prefix}mlp.experts.{expert_id}"
        matrices = []
        for name in ("gate_proj", "up_proj", "down_proj"):
                weight = store._load_ssd(prefix + "." + name + ".weight")
            scale = store._load_ssd(prefix + "." + name + ".weight_scale_inv")
            if weight.dtype != torch.float8_e4m3fn:
                raise RuntimeError(f"Predicted RAM prefetch requires FP8 source weights: {prefix}.{name}")
            matrices.append((weight.contiguous(), scale.contiguous()))
        entry = (matrices[0], matrices[1], matrices[2])

        with self.lock:
            bank = self.predicted_ram_entries.setdefault(layer, OrderedDict())
            if expert_id in self.fp8_entries.setdefault(layer, OrderedDict()) or expert_id in self.q4_entries.setdefault(layer, OrderedDict()):
                return False
            old = bank.pop(expert_id, None)
            if old is not None:
                self._erase_predicted_ram(layer, expert_id)
            bank[expert_id] = entry
            self._record_predicted_ram(layer, expert_id, entry)
            self.predicted_ram_prefetches += 1
            while len(bank) > PREDICTED_RAM_SLOTS_PER_LAYER:
                victim_id, _ = bank.popitem(last=False)
                self._erase_predicted_ram(layer, victim_id)
                self.predicted_ram_drops += 1
        return True
    def promote_q4_to_vram(self, store, layer: int, expert_id: int, layer_prefix: str) -> bool:
        """Promote an actually-used Q4 RAM expert to FP8 VRAM in the background."""
        layer = int(layer)
        expert_id = int(expert_id)
        key = (layer, expert_id)
        with self.lock:
            fp8_bank = self.fp8_entries.setdefault(layer, OrderedDict())
            q4_bank = self.q4_entries.setdefault(layer, OrderedDict())
            if expert_id in fp8_bank or expert_id not in q4_bank or key in self.q4_promotion_pending:
                return False
            self.q4_promotion_pending.add(key)

        try:
            prefix = f"{layer_prefix}mlp.experts.{expert_id}"
            raw_weights = []
            raw_scales = []
            for name in ("gate_proj", "up_proj", "down_proj"):
                weight = store.load(prefix + "." + name + ".weight", device="cpu")
                scale = store.load(prefix + "." + name + ".weight_scale_inv", device="cpu")
                if weight.dtype != torch.float8_e4m3fn:
                    return False
                raw_weights.append(weight.to(device="cuda", non_blocking=True))
                raw_scales.append(scale.to(device="cuda", non_blocking=True))
            compact: WarmEntry = (
                (raw_weights[0], raw_scales[0]),
                (raw_weights[1], raw_scales[1]),
                (raw_weights[2], raw_scales[2]),
            )
            with self.lock:
                fp8_bank = self.fp8_entries.setdefault(layer, OrderedDict())
                q4_bank = self.q4_entries.setdefault(layer, OrderedDict())
                if expert_id in fp8_bank or expert_id not in q4_bank:
                    return False
                old_q4 = q4_bank.pop(expert_id, None)
                if old_q4 is not None:
                    self._erase(layer, expert_id, "q4")
                self._insert_fp8_locked(layer, expert_id, compact)
                self.q4_promotions += 1
                return True
        finally:
            with self.lock:
                self.q4_promotion_pending.discard(key)

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
            predicted = self.predicted_ram_entries.setdefault(layer, OrderedDict())
            predicted_entry = predicted.get(expert_id)
            if predicted_entry is not None:
                self.hits += 1
                self.predicted_ram_hits += 1
                predicted.move_to_end(expert_id)
                found.append(("predicted_ram", predicted_entry))
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

    def get_or_load_batch_tiered(self, store, layer: int, expert_ids: list[int], layer_prefix: str):
        """Resolve selected experts; predicted-RAM hits are promoted to VRAM."""
        layer = int(layer)
        ids = [int(x) for x in expert_ids]
        misses = []
        with self.lock:
            fp8_bank = self.fp8_entries.setdefault(layer, OrderedDict())
            predicted_bank = self.predicted_ram_entries.setdefault(layer, OrderedDict())
            q4_bank = self.q4_entries.setdefault(layer, OrderedDict())
            for expert_id in ids:
                if expert_id not in fp8_bank and expert_id not in predicted_bank and expert_id not in q4_bank:
                    misses.append(expert_id)

        loaded = {}
        for expert_id in misses:
            prefix = f"{layer_prefix}mlp.experts.{expert_id}"
            raw_weights, raw_scales = [], []
            for name in ("gate_proj", "up_proj", "down_proj"):
                w, s = self._raw_projection_for_gpu(store, prefix + "." + name)
                raw_weights.append(w)
                raw_scales.append(s)
            if not all(w.dtype == torch.float8_e4m3fn for w in raw_weights):
                raise RuntimeError(f"Unexpected non-FP8 expert source for layer {layer}, expert {expert_id}")
            loaded[expert_id] = ((raw_weights[0], raw_scales[0]), (raw_weights[1], raw_scales[1]), (raw_weights[2], raw_scales[2]))

        with self.lock:
            for expert_id, compact in loaded.items():
                self._insert_fp8_locked(layer, expert_id, compact)
                self.loads += 1
            found = self._lookup_batch_locked(layer, ids)
            for pos, (tier, entry) in enumerate(found):
                if tier != "predicted_ram" or entry is None:
                    continue
                bank = self.predicted_ram_entries.setdefault(layer, OrderedDict())
                promoted = bank.pop(ids[pos], None)
                if promoted is None:
                    continue
                self._erase_predicted_ram(layer, ids[pos])
                self._insert_fp8_locked(layer, ids[pos], promoted)
                self.predicted_ram_promotions += 1
                found[pos] = ("fp8", promoted)

        result = []
        for tier, entry in found:
            if tier not in ("fp8", "q4") or entry is None:
                raise RuntimeError("Expert cache returned an unresolved route entry")
            result.append((tier, entry))
        return result
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
                "predicted_ram_bytes": self.predicted_ram_bytes_used,
                "host_ram_bytes": self.q4_bytes_used + self.predicted_ram_bytes_used,
                "host_ram_budget": self.budget_bytes,
                "host_ram_utilization": (
                    (self.q4_bytes_used + self.predicted_ram_bytes_used)
                    / self.budget_bytes * 100.0
                    if self.budget_bytes else 0.0
                ),
                "predicted_ram_items": sum(len(b) for b in self.predicted_ram_entries.values()),
                "predicted_ram_prefetches": self.predicted_ram_prefetches,
                "predicted_ram_hits": self.predicted_ram_hits,
                "predicted_ram_promotions": self.predicted_ram_promotions,
                "predicted_ram_drops": self.predicted_ram_drops,
                "q4_promotions": self.q4_promotions,
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
                self.predicted_ram_entries[layer].clear()
            self.entry_bytes.clear()
            self.q4_ram_bytes.clear()
            self.predicted_ram_bytes.clear()
            self.predicted_ram_bytes_used = 0
            self.q4_promotion_pending.clear()
            self.bytes_used = 0
            self.q4_bytes_used = 0
