from __future__ import annotations

"""Two-level LRU cache for Qwen3.6 FP8 Safetensors experts.

Uses the Safetensors file header for metadata, avoiding safe_open.get_dtype()
which is unavailable in some installed versions. Tensor payloads are loaded
on demand and cached in RAM and/or CUDA VRAM.
"""

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import argparse
import json
import re
import struct

import torch
from safetensors import safe_open

ExpertKey = tuple[int, int]
KINDS = ("gate_proj", "up_proj", "down_proj")

_WEIGHT_RE = re.compile(
    r"^(?:model\.language_model\.)?layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<kind>gate_proj|up_proj|down_proj)\.weight$"
)
_SCALE_RE = re.compile(
    r"^(?:model\.language_model\.)?layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<kind>gate_proj|up_proj|down_proj)\.weight_scale_inv$"
)

@dataclass(frozen=True)
class TensorRef:
    name: str
    shard: str
    shape: tuple[int, ...]
    dtype: str

@dataclass
class ExpertRecord:
    layer: int
    expert: int
    weights: dict[str, TensorRef]
    scales: dict[str, TensorRef]

@dataclass
class ExpertBlob:
    layer: int
    expert: int
    weights: dict[str, Any]
    scales: dict[str, Any]
    size_bytes: int

@dataclass
class CacheStats:
    ram_hits: int = 0
    ram_misses: int = 0
    vram_hits: int = 0
    vram_misses: int = 0
    loads_from_disk: int = 0
    ram_to_vram: int = 0
    vram_evictions: int = 0
    ram_evictions: int = 0

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
    size_bytes: int


def read_header(path: Path) -> dict[str, Any]:
    """Read only the Safetensors JSON header."""
    with path.open("rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError(f"Invalid Safetensors file: {path}")
        length = struct.unpack("<Q", raw)[0]
        data = fh.read(length)
        if len(data) != length:
            raise ValueError(f"Truncated Safetensors header: {path}")
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid Safetensors header object: {path}")
    return value


class FP8ExpertIndex:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.records: dict[ExpertKey, ExpertRecord] = {}
        self.shards: set[str] = set()
        self.layer_count = 0
        self.expert_count = 0
        self._build()

    def _discover_shards(self) -> list[Path]:
        index_path = self.root / "model.safetensors.index.json"
        if index_path.is_file():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            names = sorted(set(payload.get("weight_map", {}).values()))
            return [self.root / name for name in names if (self.root / name).is_file()]
        return sorted(self.root.glob("*.safetensors"))

    def _build(self) -> None:
        weights: dict[ExpertKey, dict[str, TensorRef]] = {}
        scales: dict[ExpertKey, dict[str, TensorRef]] = {}

        for shard in self._discover_shards():
            self.shards.add(shard.name)
            for name, meta in read_header(shard).items():
                if name == "__metadata__" or not isinstance(meta, dict):
                    continue
                ref = TensorRef(
                    name=name,
                    shard=shard.name,
                    shape=tuple(int(x) for x in meta.get("shape", ())),
                    dtype=str(meta.get("dtype", "")),
                )
                match = _WEIGHT_RE.match(name)
                if match:
                    key = (int(match.group("layer")), int(match.group("expert")))
                    weights.setdefault(key, {})[match.group("kind")] = ref
                    continue
                match = _SCALE_RE.match(name)
                if match:
                    key = (int(match.group("layer")), int(match.group("expert")))
                    scales.setdefault(key, {})[match.group("kind")] = ref

        for key, parts in weights.items():
            scale_parts = scales.get(key, {})
            if set(parts) == set(KINDS) and set(scale_parts) == set(KINDS):
                self.records[key] = ExpertRecord(key[0], key[1], parts, scale_parts)

        if not self.records:
            raise RuntimeError("No complete FP8 routed experts found")
        self.layer_count = max(key[0] for key in self.records) + 1
        self.expert_count = max(key[1] for key in self.records) + 1

    def get(self, layer: int, expert: int) -> ExpertRecord:
        key = (int(layer), int(expert))
        try:
            return self.records[key]
        except KeyError as exc:
            raise KeyError(f"Unknown FP8 expert: layer={layer}, expert={expert}") from exc


class FP8ExpertCache:
    def __init__(self, root: str | Path, *, ram_limit_bytes: int = 6 * 1024**3,
                 vram_limit_bytes: int = 3 * 1024**3, device: str = "cuda") -> None:
        self.root = Path(root).resolve()
        self.index = FP8ExpertIndex(self.root)
        self.ram_limit = int(ram_limit_bytes)
        self.vram_limit = int(vram_limit_bytes)
        self.device = device
        self.ram: OrderedDict[ExpertKey, CacheEntry] = OrderedDict()
        self.vram: OrderedDict[ExpertKey, CacheEntry] = OrderedDict()
        self.ram_used = 0
        self.vram_used = 0
        self.stats = CacheStats()
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but PyTorch CUDA is unavailable")

    @staticmethod
    def _bytes(tensor: Any) -> int:
        return int(tensor.numel() * tensor.element_size())

    def _load_tensor(self, ref: TensorRef) -> torch.Tensor:
        with safe_open(str(self.root / ref.shard), framework="pt", device="cpu") as handle:
            return handle.get_tensor(ref.name)

    def _load_blob(self, key: ExpertKey) -> ExpertBlob:
        record = self.index.get(*key)
        weights = {kind: self._load_tensor(ref) for kind, ref in record.weights.items()}
        scales = {kind: self._load_tensor(ref) for kind, ref in record.scales.items()}
        size = sum(self._bytes(t) for t in weights.values()) + sum(self._bytes(t) for t in scales.values())
        self.stats.loads_from_disk += 1
        return ExpertBlob(key[0], key[1], weights, scales, size)

    @staticmethod
    def _touch(cache: OrderedDict[ExpertKey, CacheEntry], key: ExpertKey) -> ExpertBlob:
        entry = cache.pop(key)
        cache[key] = entry
        return entry.blob

    def _evict_ram(self, required: int = 0) -> None:
        while self.ram and self.ram_used + required > self.ram_limit:
            _, entry = self.ram.popitem(last=False)
            self.ram_used -= entry.size_bytes
            self.stats.ram_evictions += 1

    def _evict_vram(self, required: int = 0) -> None:
        while self.vram and self.vram_used + required > self.vram_limit:
            key, entry = self.vram.popitem(last=False)
            self.vram_used -= entry.size_bytes
            self.stats.vram_evictions += 1
            if key in self.ram:
                continue
            blob = self._load_blob(key)
            self._evict_ram(blob.size_bytes)
            if blob.size_bytes <= self.ram_limit:
                self.ram[key] = CacheEntry(key, blob, blob.size_bytes)
                self.ram_used += blob.size_bytes

    def get_ram(self, layer: int, expert: int) -> ExpertBlob:
        key = (int(layer), int(expert))
        entry = self.ram.get(key)
        if entry is not None:
            self.stats.ram_hits += 1
            return self._touch(self.ram, key)
        self.stats.ram_misses += 1
        blob = self._load_blob(key)
        self._evict_ram(blob.size_bytes)
        if blob.size_bytes <= self.ram_limit:
            self.ram[key] = CacheEntry(key, blob, blob.size_bytes)
            self.ram_used += blob.size_bytes
        return blob

    def promote_to_vram(self, layer: int, expert: int) -> ExpertBlob:
        if not self.device.startswith("cuda"):
            raise RuntimeError("VRAM promotion requires CUDA")
        key = (int(layer), int(expert))
        entry = self.vram.get(key)
        if entry is not None:
            self.stats.vram_hits += 1
            return self._touch(self.vram, key)
        self.stats.vram_misses += 1
        blob = self.get_ram(*key)
        if blob.size_bytes > self.vram_limit:
            raise MemoryError(f"Expert {key} needs {blob.size_bytes} bytes, above VRAM limit {self.vram_limit}")
        self._evict_vram(blob.size_bytes)
        weights = {k: t.to(self.device) for k, t in blob.weights.items()}
        scales = {k: t.to(self.device) for k, t in blob.scales.items()}
        cuda_blob = ExpertBlob(blob.layer, blob.expert, weights, scales, blob.size_bytes)
        self.vram[key] = CacheEntry(key, cuda_blob, blob.size_bytes)
        self.vram_used += blob.size_bytes
        self.stats.ram_to_vram += 1
        return cuda_blob

    def get(self, layer: int, expert: int, *, tier: str = "vram") -> ExpertBlob:
        if tier == "ram":
            return self.get_ram(layer, expert)
        if tier == "vram":
            return self.promote_to_vram(layer, expert)
        raise ValueError("tier must be 'ram' or 'vram'")

    def sequence(self, keys: Iterable[ExpertKey]) -> None:
        for layer, expert in keys:
            before = set(self.vram)
            self.get(layer, expert, tier="vram")
            after = set(self.vram)
            print(f"access=({layer},{expert}) entered={sorted(after-before)} evicted={sorted(before-after)} vram_mru={list(reversed(self.vram))}")

    def snapshot(self) -> dict[str, Any]:
        return {
            "ram_used_bytes": self.ram_used,
            "ram_limit_bytes": self.ram_limit,
            "vram_used_bytes": self.vram_used,
            "vram_limit_bytes": self.vram_limit,
            "ram_entries": len(self.ram),
            "vram_entries": len(self.vram),
            "ram_keys_mru": list(reversed(self.ram)),
            "vram_keys_mru": list(reversed(self.vram)),
            "stats": {
                "ram_hits": self.stats.ram_hits,
                "ram_misses": self.stats.ram_misses,
                "vram_hits": self.stats.vram_hits,
                "vram_misses": self.stats.vram_misses,
                "loads_from_disk": self.stats.loads_from_disk,
                "ram_to_vram": self.stats.ram_to_vram,
                "vram_evictions": self.stats.vram_evictions,
                "ram_evictions": self.stats.ram_evictions,
                "ram_hit_rate": self.stats.ram_hit_rate,
                "vram_hit_rate": self.stats.vram_hit_rate,
            },
        }


def parse_sequence(values: list[str]) -> list[ExpertKey]:
    out: list[ExpertKey] = []
    for value in values:
        try:
            layer, expert = value.split(":", 1)
            out.append((int(layer), int(expert)))
        except ValueError as exc:
            raise SystemExit(f"Invalid expert '{value}'. Use LAYER:EXPERT") from exc
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.6 FP8 expert cache")
    parser.add_argument("root", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--ram-gb", type=float, default=6.0)
    parser.add_argument("--vram-gb", type=float, default=3.0)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--sequence", nargs="*")
    args = parser.parse_args()

    cache = FP8ExpertCache(
        args.root,
        ram_limit_bytes=int(args.ram_gb * 1024**3),
        vram_limit_bytes=int(args.vram_gb * 1024**3),
        device="cpu" if args.cpu_only else "cuda",
    )

    if args.sequence:
        cache.sequence(parse_sequence(args.sequence))
    else:
        blob = cache.get(args.layer, args.expert, tier="ram" if args.cpu_only else "vram")
        print(json.dumps({
            "expert": [args.layer, args.expert],
            "size_bytes": blob.size_bytes,
            "weight_shapes": {k: list(v.shape) for k, v in blob.weights.items()},
            "scale_shapes": {k: list(v.shape) for k, v in blob.scales.items()},
        }, indent=2))
    print(json.dumps(cache.snapshot(), indent=2))


if __name__ == "__main__":
    main()
