from __future__ import annotations

"""Minimal RAM/VRAM LRU cache for packed Qwen3.6 GGUF experts.

This module does not execute the model. It only provides the memory-management
primitive we need for the routing experiment:

    GGUF -> RAM -> CUDA VRAM

An expert is identified by (layer, expert).  Qwen3.6 stores the routed experts
packed into the last axis of three tensors per layer: gate, up and down.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - allows cache simulation without CUDA
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
    """One logical expert containing its three packed tensor slices."""

    layer: int
    expert: int
    slices: dict[str, bytes]
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

    @property
    def total_requests(self) -> int:
        return self.ram_hits + self.ram_misses

    @property
    def vram_hit_rate(self) -> float:
        total = self.vram_hits + self.vram_misses
        return self.vram_hits / total if total else 0.0

    @property
    def ram_hit_rate(self) -> float:
        total = self.ram_hits + self.ram_misses
        return self.ram_hits / total if total else 0.0


@dataclass
class CacheEntry:
    key: ExpertKey
    blob: ExpertBlob | Any
    size: int


class ExpertIndex:
    """Build an index over packed Qwen3.6 expert tensors without decoding them."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        self.reader = GGUFReader(str(self.model_path))
        self.slices: dict[ExpertKey, dict[str, TensorSlice]] = {}
        self.expert_count = 0
        self._build()

    @staticmethod
    def _nbytes(tensor: Any) -> int:
        value = getattr(tensor, "n_bytes", None)
        if value is not None:
            return int(value)
        value = getattr(tensor, "nbytes", None)
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
            name = tensor.name
            if not name.startswith("blk.") or not name.endswith(".weight"):
                continue

            parts = name.split(".")
            if len(parts) != 4 or parts[0] != "blk" or parts[2] not in {
                "ffn_gate_exps",
                "ffn_up_exps",
                "ffn_down_exps",
            }:
                continue

            layer = int(parts[1])
            kind = {
                "ffn_gate_exps": "gate",
                "ffn_up_exps": "up",
                "ffn_down_exps": "down",
            }[parts[2]]

            shape = tuple(int(x) for x in tensor.shape)
            if len(shape) != 3:
                raise ValueError(f"Expected 3D expert tensor, got {name}: {shape}")

            expert_count = shape[2]
            if expert_count <= 0:
                raise ValueError(f"Invalid expert count in {name}: {expert_count}")

            nbytes = self._nbytes(tensor)
            if nbytes % expert_count != 0:
                raise ValueError(
                    f"{name}: {nbytes} bytes is not divisible by "
                    f"{expert_count} experts"
                )

            packed.setdefault(layer, {})[kind] = {
                "name": name,
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

            expert_count = counts.pop()
            self.expert_count = max(self.expert_count, expert_count)

            for expert in range(expert_count):
                entry: dict[str, TensorSlice] = {}
                for kind in EXPERT_KINDS:
                    source = sources[kind]
                    size = source["nbytes"] // expert_count
                    offset = source["offset"] + expert * size
                    entry[kind] = TensorSlice(
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
                self.slices[(layer, expert)] = entry

    def get(self, layer: int, expert: int) -> dict[str, TensorSlice]:
        key = (int(layer), int(expert))
        try:
            return self.slices[key]
        except KeyError as exc:
            raise KeyError(f"Unknown expert: layer={layer}, expert={expert}") from exc

    def size(self, layer: int, expert: int) -> int:
        return sum(part.size for part in self.get(layer, expert).values())


class ExpertCache:
    """Two-level LRU cache with optional CUDA VRAM residency.

    The RAM layer is always available.  CUDA VRAM is optional so the same class
    can be used as a pure cache simulator on machines without CUDA.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        ram_limit_bytes: int = 6 * 1024**3,
        vram_limit_bytes: int = 3 * 1024**3,
        device: str = "cuda",
    ) -> None:
        self.index = ExpertIndex(Path(model_path))
        self.model_path = Path(model_path)
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

    def _evict_ram(self, required: int = 0) -> None:
        while self.ram and self.ram_used + required > self.ram_limit:
            key, entry = self.ram.popitem(last=False)
            self.ram_used -= entry.size
            self.stats.ram_evictions += 1

    def _evict_vram(self, required: int = 0) -> None:
        while self.vram and self.vram_used + required > self.vram_limit:
            key, entry = self.vram.popitem(last=False)
            self.vram_used -= entry.size
            self.stats.vram_evictions += 1

            # Keep a CPU/RAM copy when possible.  This is deliberately lazy:
            # if RAM is full, the CPU copy may be evicted as well.
            blob = entry.blob
            ram_entry = CacheEntry(key=key, blob=blob, size=entry.size)
            self._evict_ram(entry.size)
            if entry.size <= self.ram_limit:
                self.ram[key] = ram_entry
                self.ram_used += entry.size

    def _get_ram_blob(self, key: ExpertKey) -> ExpertBlob:
        entry = self.ram.get(key)
        if entry is not None:
            self.stats.ram_hits += 1
            return self._touch(self.ram, key).blob

        self.stats.ram_misses += 1
        blob = self._load_blob(key)
        self.stats.loads_from_file += 1

        self._evict_ram(blob.size)
        if blob.size <= self.ram_limit:
            self.ram[key] = CacheEntry(key=key, blob=blob, size=blob.size)
            self.ram_used += blob.size
        return blob

    def get_cpu(self, layer: int, expert: int) -> ExpertBlob:
        """Get an expert in CPU/RAM form, loading only its three slices."""
        return self._get_ram_blob((int(layer), int(expert)))

    def get_cuda(self, layer: int, expert: int):
        """Get an expert represented by CUDA tensors.

        The bytes remain quantized GGUF bytes for now.  This method tests the
        memory movement path only; a future executor will decode/compute them.
        """
        if not self.device.startswith("cuda"):
            raise RuntimeError("CUDA access requested while cache device is CPU-only")

        key = (int(layer), int(expert))
        entry = self.vram.get(key)
        if entry is not None:
            self.stats.vram_hits += 1
            return self._touch(self.vram, key).blob

        self.stats.vram_misses += 1
        blob = self._get_ram_blob(*key)
        self._evict_vram(blob.size)

        if blob.size > self.vram_limit:
            raise MemoryError(
                f"Expert {key} needs {blob.size} bytes, above VRAM cache limit "
                f"{self.vram_limit} bytes"
            )

        cuda_slices = {
            kind: torch.frombuffer(memoryview(data), dtype=torch.uint8).clone().to(self.device)
            for kind, data in blob.slices.items()
        }
        cuda_blob = ExpertBlob(
            layer=blob.layer,
            expert=blob.expert,
            slices=cuda_slices,
            size=blob.size,
        )
        self.vram[key] = CacheEntry(key=key, blob=cuda_blob, size=blob.size)
        self.vram_used += blob.size
        return cuda_blob

    def release(self, layer: int, expert: int) -> None:
        """Remove an expert from both cache levels."""
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
                "ram_hit_rate": self.stats.ram_hit_rate,
                "vram_hit_rate": self.stats.vram_hit_rate,
            },
        }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Probe the Router-IA expert cache")
    parser.add_argument("model", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--ram-gb", type=float, default=6.0)
    parser.add_argument("--vram-gb", type=float, default=3.0)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()

    cache = ExpertCache(
        args.model,
        ram_limit_bytes=int(args.ram_gb * 1024**3),
        vram_limit_bytes=int(args.vram_gb * 1024**3),
        device="cpu" if args.cpu_only else "cuda",
    )

    info = cache.index.get(args.layer, args.expert)
    print(json.dumps({
        "expert": [args.layer, args.expert],
        "size_bytes": cache.index.size(args.layer, args.expert),
        "slices": {kind: vars(part) for kind, part in info.items()},
    }, indent=2, default=str))

    cache.get_cpu(args.layer, args.expert)
    print(json.dumps(cache.snapshot(), indent=2, default=str))
