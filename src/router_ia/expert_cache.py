from __future__ import annotations

"""Minimal two-level LRU cache for packed Qwen3.6 GGUF experts.

This module is an experiment harness, not a model executor.
It manages residency:

    GGUF -> RAM -> CUDA VRAM

An expert is identified by ``(layer, expert)``. Qwen3.6 stores routed experts
packed along the last axis of three tensors per layer: gate, up and down.
"""

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from gguf import GGUFReader

ExpertKey = tuple[int, int]
EXPERT_KINDS = ("gate", "up", "down")


@dataclass(frozen=True)
class TensorSlice:
    tensor: str
    layer: int
    expert: int
    kind: str
    offset: int
    end_offset: int
    size: int
    shape: tuple[int, ...]
    dtype: str


@dataclass
class ExpertBlob:
    layer: int
    expert: int
    slices: dict[str, Any]
    size: int


@dataclass
class CacheStats:
    ram_hits: int = 0
    ram_misses: int = 0
    vram_hits: int = 0
    vram_misses: int = 0
    ram_evictions: int = 0
    vram_evictions: int = 0
    loads_from_file: int = 0
    ram_to_vram: int = 0
    vram_to_ram: int = 0

    @property
    def ram_hit_rate(self) -> float:
        total = self.ram_hits + self.ram_misses
        return self.ram_hits / total if total else 0.0

    @property
    def vram_hit_rate(self) -> float:
        total = self.vram_hits + self.vram_misses
        return self.vram_hits / total if total else 0.0


@dataclass
class CacheEntry:
    key: ExpertKey
    blob: ExpertBlob
    size: int


class ExpertIndex:
    """Index packed Qwen3.6 expert slices without decoding quantization."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        self.reader = GGUFReader(str(self.model_path))
        self.slices: dict[ExpertKey, dict[str, TensorSlice]] = {}
        self.expert_count = 0
        self.layer_count = 0
        self._build()

    @staticmethod
    def _nbytes(tensor: Any) -> int:
        for attr in ("n_bytes", "nbytes"):
            value = getattr(tensor, attr, None)
            if value is not None:
                return int(value)
        data = getattr(tensor, "data", None)
        if data is not None:
            return int(data.nbytes)
        raise ValueError(f"Unable to determine tensor size: {tensor.name}")

    @staticmethod
    def _offset(tensor: Any) -> int:
        value = getattr(tensor, "data_offset", None)
        if value is None:
            raise ValueError(f"Tensor has no data_offset: {tensor.name}")
        return int(value)

    def _build(self) -> None:
        packed: dict[int, dict[str, Any]] = {}

        for tensor in self.reader.tensors:
            parts = tensor.name.split(".")
            if len(parts) != 4 or parts[0] != "blk" or parts[3] != "weight":
                continue

            kind_name = parts[2]
            if kind_name not in {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"}:
                continue

            layer = int(parts[1])
            kind = {
                "ffn_gate_exps": "gate",
                "ffn_up_exps": "up",
                "ffn_down_exps": "down",
            }[kind_name]
            shape = tuple(int(x) for x in tensor.shape)
            if len(shape) != 3:
                raise ValueError(f"Expected 3D expert tensor, got {tensor.name}: {shape}")

            expert_count = shape[2]
            nbytes = self._nbytes(tensor)
            if expert_count <= 0 or nbytes % expert_count:
                raise ValueError(f"Invalid packed expert tensor: {tensor.name}")

            packed.setdefault(layer, {})[kind] = {
                "name": tensor.name,
                "shape": shape,
                "dtype": str(tensor.tensor_type),
                "offset": self._offset(tensor),
                "nbytes": nbytes,
                "expert_count": expert_count,
            }

        for layer, sources in packed.items():
            if set(sources) != set(EXPERT_KINDS):
                continue

            counts = {int(source["expert_count"]) for source in sources.values()}
            if len(counts) != 1:
                raise ValueError(f"Inconsistent expert counts at layer {layer}: {counts}")

            count = counts.pop()
            self.expert_count = max(self.expert_count, count)
            self.layer_count = max(self.layer_count, layer + 1)

            for expert in range(count):
                key = (layer, expert)
                self.slices[key] = {}
                for kind in EXPERT_KINDS:
                    source = sources[kind]
                    size = source["nbytes"] // count
                    offset = source["offset"] + expert * size
                    self.slices[key][kind] = TensorSlice(
                        tensor=source["name"],
                        layer=layer,
                        expert=expert,
                        kind=kind,
                        offset=offset,
                        end_offset=offset + size,
                        size=size,
                        shape=source["shape"],
                        dtype=source["dtype"],
                    )

    def get(self, layer: int, expert: int) -> dict[str, TensorSlice]:
        key = (int(layer), int(expert))
        if key not in self.slices:
            raise KeyError(f"Unknown expert: layer={layer}, expert={expert}")
        return self.slices[key]

    def size(self, layer: int, expert: int) -> int:
        return sum(part.size for part in self.get(layer, expert).values())


class ExpertCache:
    """Two-level LRU cache with explicit RAM <-> VRAM residency."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        ram_limit_bytes: int = 6 * 1024**3,
        vram_limit_bytes: int = 3 * 1024**3,
        device: str = "cuda",
    ) -> None:
        self.model_path = Path(model_path)
        self.index = ExpertIndex(self.model_path)
        self.ram_limit = int(ram_limit_bytes)
        self.vram_limit = int(vram_limit_bytes)
        self.device = device

        self.ram: OrderedDict[ExpertKey, CacheEntry] = OrderedDict()
        self.vram: OrderedDict[ExpertKey, CacheEntry] = OrderedDict()
        self.ram_used = 0
        self.vram_used = 0
        self.stats = CacheStats()

        if device.startswith("cuda"):
            if torch is None or not torch.cuda.is_available():
                raise RuntimeError("CUDA cache requested, but PyTorch CUDA is unavailable")

    def _read_slice(self, part: TensorSlice) -> bytes:
        with self.model_path.open("rb") as handle:
            handle.seek(part.offset)
            data = handle.read(part.size)
        if len(data) != part.size:
            raise IOError(
                f"Short read for {part.tensor}: expected {part.size}, got {len(data)}"
            )
        return data

    def _load_blob(self, key: ExpertKey) -> ExpertBlob:
        parts = self.index.get(*key)
        slices = {kind: self._read_slice(part) for kind, part in parts.items()}
        self.stats.loads_from_file += 1
        return ExpertBlob(
            layer=key[0],
            expert=key[1],
            slices=slices,
            size=sum(len(data) for data in slices.values()),
        )

    @staticmethod
    def _touch(cache: OrderedDict[ExpertKey, CacheEntry], key: ExpertKey) -> CacheEntry:
        entry = cache.pop(key)
        cache[key] = entry
        return entry

    def _evict_ram(self, required: int = 0, protected: set[ExpertKey] | None = None) -> None:
        protected = protected or set()
        while self.ram and self.ram_used + required > self.ram_limit:
            victim = next((key for key in self.ram if key not in protected), None)
            if victim is None:
                break
            entry = self.ram.pop(victim)
            self.ram_used -= entry.size
            self.stats.ram_evictions += 1

    def _evict_vram(self, required: int = 0) -> None:
        """Evict LRU VRAM experts; their CPU copy stays in RAM when possible."""
        while self.vram and self.vram_used + required > self.vram_limit:
            key, entry = self.vram.popitem(last=False)
            self.vram_used -= entry.size
            self.stats.vram_evictions += 1

            if key in self.ram:
                self._touch(self.ram, key)
                self.stats.vram_to_ram += 1
                continue

            # Defensive fallback if the CPU copy was evicted independently.
            blob = self._load_blob(key)
            self._evict_ram(blob.size)
            if blob.size <= self.ram_limit:
                self.ram[key] = CacheEntry(key=key, blob=blob, size=blob.size)
                self.ram_used += blob.size
                self.stats.vram_to_ram += 1

    def get_cpu(self, layer: int, expert: int) -> ExpertBlob:
        key = (int(layer), int(expert))
        entry = self.ram.get(key)
        if entry is not None:
            self.stats.ram_hits += 1
            return self._touch(self.ram, key).blob

        self.stats.ram_misses += 1
        blob = self._load_blob(key)
        # Do not evict RAM copies of experts currently resident in VRAM.
        self._evict_ram(blob.size, protected=set(self.vram.keys()))
        if blob.size <= self.ram_limit:
            self.ram[key] = CacheEntry(key=key, blob=blob, size=blob.size)
            self.ram_used += blob.size
        return blob

    def promote_to_vram(self, layer: int, expert: int) -> ExpertBlob:
        """Promote one quantized expert from RAM to CUDA VRAM."""
        if not self.device.startswith("cuda") or torch is None:
            raise RuntimeError("VRAM promotion requires a CUDA device")

        key = (int(layer), int(expert))
        entry = self.vram.get(key)
        if entry is not None:
            self.stats.vram_hits += 1
            return self._touch(self.vram, key).blob

        self.stats.vram_misses += 1
        cpu_blob = self.get_cpu(*key)
        if cpu_blob.size > self.vram_limit:
            raise MemoryError(
                f"Expert {key} needs {cpu_blob.size} bytes, above VRAM limit "
                f"{self.vram_limit} bytes"
            )

        self._evict_vram(cpu_blob.size)

        cuda_slices = {
            kind: torch.frombuffer(memoryview(data), dtype=torch.uint8).clone().to(self.device)
            for kind, data in cpu_blob.slices.items()
        }
        cuda_blob = ExpertBlob(
            layer=cpu_blob.layer,
            expert=cpu_blob.expert,
            slices=cuda_slices,
            size=cpu_blob.size,
        )
        self.vram[key] = CacheEntry(key=key, blob=cuda_blob, size=cpu_blob.size)
        self.vram_used += cpu_blob.size
        self.stats.ram_to_vram += 1
        return cuda_blob

    def get_vram(self, layer: int, expert: int) -> ExpertBlob:
        return self.promote_to_vram(layer, expert)

    def access(self, layer: int, expert: int, *, tier: str = "vram") -> ExpertBlob:
        tier = tier.lower()
        if tier == "ram":
            return self.get_cpu(layer, expert)
        if tier == "vram":
            return self.get_vram(layer, expert)
        raise ValueError("tier must be 'ram' or 'vram'")

    def access_sequence(self, sequence: Iterable[ExpertKey], *, tier: str = "vram") -> None:
        """Access experts in order and print residency changes."""
        for key in sequence:
            before = set(self.vram.keys())
            self.access(*key, tier=tier)
            after = set(self.vram.keys())
            entered = sorted(after - before)
            evicted = sorted(before - after)
            print(
                f"access=({key[0]},{key[1]}) "
                f"entered={entered} evicted={evicted} "
                f"vram_mru={list(reversed(self.vram.keys()))}"
            )

    def release(self, layer: int, expert: int) -> None:
        key = (int(layer), int(expert))
        entry = self.vram.pop(key, None)
        if entry is not None:
            self.vram_used -= entry.size
        entry = self.ram.pop(key, None)
        if entry is not None:
            self.ram_used -= entry.size

    def clear(self) -> None:
        self.ram.clear()
        self.vram.clear()
        self.ram_used = 0
        self.vram_used = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "ram_used_bytes": self.ram_used,
            "ram_limit_bytes": self.ram_limit,
            "vram_used_bytes": self.vram_used,
            "vram_limit_bytes": self.vram_limit,
            "ram_entries": len(self.ram),
            "vram_entries": len(self.vram),
            "ram_keys_mru": list(reversed(self.ram.keys())),
            "vram_keys_mru": list(reversed(self.vram.keys())),
            "stats": {
                "ram_hits": self.stats.ram_hits,
                "ram_misses": self.stats.ram_misses,
                "vram_hits": self.stats.vram_hits,
                "vram_misses": self.stats.vram_misses,
                "ram_evictions": self.stats.ram_evictions,
                "vram_evictions": self.stats.vram_evictions,
                "loads_from_file": self.stats.loads_from_file,
                "ram_to_vram": self.stats.ram_to_vram,
                "vram_to_ram": self.stats.vram_to_ram,
                "ram_hit_rate": self.stats.ram_hit_rate,
                "vram_hit_rate": self.stats.vram_hit_rate,
            },
        }


def _parse_sequence(values: list[str]) -> list[ExpertKey]:
    result: list[ExpertKey] = []
    for value in values:
        try:
            layer_text, expert_text = value.split(":", 1)
            result.append((int(layer_text), int(expert_text)))
        except ValueError as exc:
            raise SystemExit(f"Expert inválido '{value}'. Use LAYER:EXPERT.") from exc
    return result


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Router-IA expert cache probe")
    parser.add_argument("model", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--ram-gb", type=float, default=6.0)
    parser.add_argument("--vram-gb", type=float, default=3.0)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--sequence", nargs="*", help="Sequence such as 0:0 0:1 0:2 0:0")
    parser.add_argument("--tier", choices=("ram", "vram"), default="vram")
    args = parser.parse_args()

    cache = ExpertCache(
        args.model,
        ram_limit_bytes=int(args.ram_gb * 1024**3),
        vram_limit_bytes=int(args.vram_gb * 1024**3),
        device="cpu" if args.cpu_only else "cuda",
    )

    if args.sequence:
        cache.access_sequence(_parse_sequence(args.sequence), tier=args.tier)
    else:
        info = cache.index.get(args.layer, args.expert)
        print(json.dumps({
            "expert": [args.layer, args.expert],
            "size_bytes": cache.index.size(args.layer, args.expert),
            "slices": {kind: vars(part) for kind, part in info.items()},
        }, indent=2, default=str))
        cache.access(args.layer, args.expert, tier=args.tier)

    print(json.dumps(cache.snapshot(), indent=2, default=str))
