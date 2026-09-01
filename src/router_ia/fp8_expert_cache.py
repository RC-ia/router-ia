from __future__ import annotations

"""Two-level LRU cache for Qwen3.6 FP8 Safetensors experts.

The cache stores a logical expert as its gate/up/down FP8 weight tensors plus
matching weight_scale_inv tensors. It reads only the requested expert from the
shard and can keep the result in RAM and/or CUDA VRAM.

This module deliberately does not execute the model yet.
"""

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import json
import re

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from safetensors import safe_open

ExpertKey = tuple[int, int]
KINDS = ("gate_proj", "up_proj", "down_proj")

_WEIGHT_RE = re.compile(
    r"^(?:model\.language_model\.)?layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<kind>gate_proj|up_proj|down_proj)\.weight$"
)

_SCALE_RE = re.compile(
    r"^(?:model\.language_model\.)?layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<kind>gate_proj|up_proj|down_proj)\.weight_scale_inv$"
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


class FP8ExpertIndex:
    """Index individual Qwen3.6 experts across Safetensors shards."""

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
        shards = self._discover_shards()
        if not shards:
            raise FileNotFoundError(f"No Safetensors shards found in {self.root}")

        partial_weights: dict[ExpertKey, dict[str, TensorRef]] = {}
        partial_scales: dict[ExpertKey, dict[str, TensorRef]] = {}

        for shard in shards:
            self.shards.add(shard.name)
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    if m := _WEIGHT_RE.match(name):
                        layer = int(m.group("layer"))
                        expert = int(m.group("expert"))
                        ref = TensorRef(
                            name=name,
                            shard=shard.name,
                            shape=tuple(int(x) for x in handle.get_slice(name).get_shape()),
                            dtype=str(handle.get_dtype(name)),
                        )
                        partial_weights.setdefault((layer, expert), {})[m.group("kind")] = ref
                    elif m := _SCALE_RE.match(name):
                        layer = int(m.group("layer"))
                        expert = int(m.group("expert"))
                        ref = TensorRef(
                            name=name,
                            shard=shard.name,
                            shape=tuple(int(x) for x in handle.get_slice(name).get_shape()),
                            dtype=str(handle.get_dtype(name)),
                        )
                        partial_scales.setdefault((layer, expert), {})[m.group("kind")] = ref

        for key, weights in partial_weights.items():
            scales = partial_scales.get(key, {})
            if set(weights) != set(KINDS) or set(scales) != set(KINDS):
                continue
            self.records[key] = ExpertRecord(
                layer=key[0],
                expert=key[1],
                weights=weights,
                scales=scales,
            )

        if not self.records:
            raise RuntimeError("No complete FP8 routed experts found")

        self.layer_count = max(layer for layer, _ in self.records) + 1
        self.expert_count = max(expert for _, expert in self.records) + 1

    def get(self, layer: int, expert: int) -> ExpertRecord:
        key = (int(layer), int(expert))
        try:
            return self.records[key]
        except KeyError as exc:
            raise KeyError(f"Unknown FP8 expert: layer={layer}, expert={expert}") from exc


class FP8ExpertCache:
    """RAM/VRAM LRU cache for complete FP8 experts."""

    def __init__(
        self,
        root: str | Path,
        *,
        ram_limit_bytes: int = 6 * 1024**3,
        vram_limit_bytes: int = 3 * 1024**3,
        device: str = "cuda",
    ) -> None:
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

        if device.startswith("cuda"):
            if torch is None or not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but PyTorch CUDA is unavailable")

    @staticmethod
    def _tensor_nbytes(tensor: Any) -> int:
        if hasattr(tensor, "nbytes"):
            return int(tensor.nbytes)
        return int(tensor.numel() * tensor.element_size())

    def _load_tensor(self, ref: TensorRef) -> Any:
        shard = self.root / ref.shard
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(ref.name)
        return tensor

    def _load_blob(self, key: ExpertKey) -> ExpertBlob:
        record = self.index.get(*key)
        weights = {kind: self._load_tensor(ref) for kind, ref in record.weights.items()}
        scales = {kind: self._load_tensor(ref) for kind, ref in record.scales.items()}
        self.stats.loads_from_disk += 1
        size = sum(self._tensor_nbytes(x) for x in weights.values())
        size += sum(self._tensor_nbytes(x) for x in scales.values())
        return ExpertBlob(key[0], key[1], weights, scales, size)

    @staticmethod
    def _touch(cache: OrderedDict[ExpertKey, CacheEntry], key: ExpertKey) -> CacheEntry:
        entry = cache.pop(key)
        cache[key] = entry
        return entry

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
            # CPU copy remains in RAM when already cached.
            if key in self.ram:
                continue
            try:
                blob = self._load_blob(key)
            except Exception:
                continue
            self._evict_ram(blob.size_bytes)
            if blob.size_bytes <= self.ram_limit:
                self.ram[key] = CacheEntry(key, blob, blob.size_bytes)
                self.ram_used += blob.size_bytes

    def get_ram(self, layer: int, expert: int) -> ExpertBlob:
        key = (int(layer), int(expert))
        entry = self.ram.get(key)
        if entry is not None:
            self.stats.ram_hits += 1
            return self._touch(self.ram, key).blob

        self.stats.ram_misses += 1
        blob = self._load_blob(key)
        self._evict_ram(blob.size_bytes)
        if blob.size_bytes <= self.ram_limit:
            self.ram[key] = CacheEntry(key, blob, blob.size_bytes)
            self.ram_used += blob.size_bytes
        return blob

    def promote_to_vram(self, layer: int, expert: int) -> ExpertBlob:
        if not self.device.startswith("cuda") or torch is None:
            raise RuntimeError("VRAM promotion requires CUDA")

        key = (int(layer), int(expert))
        entry = self.vram.get(key)
        if entry is not None:
            self.stats.vram_hits += 1
            return self._touch(self.vram, key).blob

        self.stats.vram_misses += 1
        cpu_blob = self.get_ram(*key)
        if cpu_blob.size_bytes > self.vram_limit:
            raise MemoryError(
                f"Expert {key} requires {cpu_blob.size_bytes} bytes, "
                f"above VRAM limit {self.vram_limit}"
            )

        self._evict_vram(cpu_blob.size_bytes)
        cuda_weights = {
            kind: tensor.to(self.device)
            for kind, tensor in cpu_blob.weights.items()
        }
        cuda_scales = {
            kind: tensor.to(self.device)
            for kind, tensor in cpu_blob.scales.items()
        }
        cuda_blob = ExpertBlob(
            cpu_blob.layer,
            cpu_blob.expert,
            cuda_weights,
            cuda_scales,
            cpu_blob.size_bytes,
        )
        self.vram[key] = CacheEntry(key, cuda_blob, cuda_blob.size_bytes)
        self.vram_used += cuda_blob.size_bytes
        self.stats.ram_to_vram += 1
        return cuda_blob

    def get(self, layer: int, expert: int, *, tier: str = "vram") -> ExpertBlob:
        tier = tier.lower()
        if tier == "ram":
            return self.get_ram(layer, expert)
        if tier == "vram":
            return self.promote_to_vram(layer, expert)
        raise ValueError("tier must be 'ram' or 'vram'")

    def sequence(self, keys: Iterable[ExpertKey], *, tier: str = "vram") -> None:
        for layer, expert in keys:
            before = set(self.vram)
            self.get(layer, expert, tier=tier)
            after = set(self.vram)
            print(
                f"access=({layer},{expert}) "
                f"entered={sorted(after - before)} "
                f"evicted={sorted(before - after)} "
                f"vram_mru={list(reversed(self.vram))}"
            )

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
    result: list[ExpertKey] = []
    for value in values:
        try:
            layer, expert = value.split(":", 1)
            result.append((int(layer), int(expert)))
        except ValueError as exc:
            raise SystemExit(f"Invalid expert '{value}'. Use LAYER:EXPERT") from exc
    return result


def main() -> None:
    parser = __import__("argparse").ArgumentParser(description="FP8 Qwen3.6 expert cache")
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
        cache.sequence(parse_sequence(args.sequence), tier="vram")
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
