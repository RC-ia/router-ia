from __future__ import annotations

"""Compact in-RAM expert location index for Qwen3.6 checkpoints.

The runtime only needs to know which Safetensors shard contains each routed
expert tensor. Keep this metadata tiny and keep tensor payloads out of RAM.
"""

import json
import re
from pathlib import Path
from typing import Any

EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)(\.(?:weight|weight_scale_inv))$"
)


def _tensor_key(name: str) -> tuple[int, int, str, str] | None:
    match = EXPERT_RE.match(name)
    if match is None:
        return None
    layer, expert, kind, suffix = match.groups()
    return int(layer), int(expert), kind, suffix[1:]


def build_compact_index(model_dir: Path, output: Path | None = None) -> Path:
    """Build a tiny expert-only index from model.safetensors.index.json."""
    model_dir = Path(model_dir).resolve()
    source = model_dir / "model.safetensors.index.json"
    if not source.is_file():
        raise FileNotFoundError(f"Missing {source}")

    payload = json.loads(source.read_text(encoding="utf-8"))
    weight_map: dict[str, str] = dict(payload.get("weight_map", {}))
    shards = sorted({str(v) for k, v in weight_map.items() if _tensor_key(k) is not None})
    shard_ids = {name: index for index, name in enumerate(shards)}

    # 40 layers x 256 experts x 6 tensor locations.
    # Each location is a compact uint-like shard id; -1 means absent.
    experts: list[list[list[int]]] = [[[-1] * 6 for _ in range(256)] for _ in range(40)]
    kinds = {"gate_proj": 0, "up_proj": 2, "down_proj": 4}
    for tensor_name, shard_name in weight_map.items():
        key = _tensor_key(tensor_name)
        if key is None:
            continue
        layer, expert, kind, suffix = key
        if layer >= 40 or expert >= 256:
            continue
        base = kinds[kind]
        offset = 0 if suffix == "weight" else 1
        experts[layer][expert][base + offset] = shard_ids[shard_name]

    compact: dict[str, Any] = {
        "version": 1,
        "layers": 40,
        "experts": 256,
        "shards": shards,
        "locations": experts,
    }
    if output is None:
        output = model_dir / "model-map.experts.min.json"
    output = Path(output)
    output.write_text(
        json.dumps(compact, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )
    return output


class CompactExpertIndex:
    """In-RAM lookup table for routed expert tensors."""

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir).resolve()
        self.path = self.model_dir / "model-map.experts.min.json"
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Compact expert index not found: {self.path}. "
                "Run qwen36_build_expert_index first."
            )
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.shards: tuple[str, ...] = tuple(payload["shards"])
        self.locations: list[list[list[int]]] = payload["locations"]
        self.lookup_count = 0
        self.miss_count = 0

    @staticmethod
    def _slot(kind: str, suffix: str) -> int:
        base = {"gate_proj": 0, "up_proj": 2, "down_proj": 4}[kind]
        return base + (0 if suffix == "weight" else 1)

    def lookup(self, tensor_name: str) -> Path | None:
        key = _tensor_key(tensor_name)
        if key is None:
            return None
        layer, expert, kind, suffix = key
        self.lookup_count += 1
        if layer >= len(self.locations) or expert >= len(self.locations[layer]):
            self.miss_count += 1
            return None
        shard_id = self.locations[layer][expert][self._slot(kind, suffix)]
        if shard_id < 0 or shard_id >= len(self.shards):
            self.miss_count += 1
            return None
        return self.model_dir / self.shards[shard_id]

    def stats(self) -> dict[str, int]:
        return {
            "shards": len(self.shards),
            "lookups": self.lookup_count,
            "misses": self.miss_count,
        }


__all__ = ["CompactExpertIndex", "build_compact_index"]
