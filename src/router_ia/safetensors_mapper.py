from __future__ import annotations

"""Inspect Qwen3.6 FP8 Safetensors shards without loading tensor payloads."""

import argparse
import json
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


EXPERT_PATTERNS = (
    re.compile(
        r"^(?P<prefix>.*?)(?:model\.layers\.)?(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\.(?P<kind>gate_proj|up_proj|down_proj)\.weight$"
    ),
    re.compile(
        r"^(?P<prefix>.*?)(?:layers\.)(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\.(?P<kind>gate_proj|up_proj|down_proj)\.weight$"
    ),
)
SCALE_RE = re.compile(
    r"^(?P<prefix>.*?)(?:(?:model\.)?layers\.)?(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\.(?P<kind>gate_proj|up_proj|down_proj)\.weight_scale_inv$"
)


@dataclass
class TensorInfo:
    name: str
    shape: list[int]
    dtype: str
    shard: str


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    """Read the JSON header of a safetensors file."""
    with open(path, "rb") as f:
        size_bytes = f.read(8)
        if len(size_bytes) < 8:
            raise ValueError(f"Invalid safetensors file: {path}")
        header_len = struct.unpack("<Q", size_bytes)[0]
        header_bytes = f.read(header_len)
        return json.loads(header_bytes.decode("utf-8"))


def _expert_match(name: str) -> re.Match[str] | None:
    for pattern in EXPERT_PATTERNS:
        match = pattern.match(name)
        if match:
            return match
    return None


def _load_index(root: Path) -> dict[str, Any] | None:
    path = root / "model.safetensors.index.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _metadata(root: Path) -> dict[str, Any]:
    config = root / "config.json"
    if not config.is_file():
        return {}
    try:
        return {"config": json.loads(config.read_text(encoding="utf-8"))}
    except json.JSONDecodeError:
        return {"config": {"error": "invalid config.json"}}


def _shards(root: Path, index: dict[str, Any] | None) -> list[Path]:
    if index and isinstance(index.get("weight_map"), dict):
        names = sorted(set(index["weight_map"].values()))
        return [root / name for name in names if (root / name).is_file()]
    return sorted(root.glob("*.safetensors"))


def inspect(root: Path, *, show_tensors: bool = False) -> dict[str, Any]:
    root = root.resolve()
    index = _load_index(root)
    shards = _shards(root, index)
    if not shards:
        raise FileNotFoundError(f"No .safetensors shards found in {root}")

    tensors: list[TensorInfo] = []
    experts: dict[tuple[int, int], dict[str, TensorInfo]] = {}
    scales: dict[tuple[int, int], dict[str, TensorInfo]] = {}

    for shard in shards:
        header = _read_safetensors_header(shard)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            # meta contains "dtype" (e.g., "F32", "BF16") and "shape" list
            info = TensorInfo(
                name=name,
                shape=meta["shape"],
                dtype=meta["dtype"],
                shard=shard.name,
            )
            tensors.append(info)

            match = _expert_match(name)
            if match:
                key = (int(match.group("layer")), int(match.group("expert")))
                experts.setdefault(key, {})[match.group("kind")] = info
                continue

            scale_match = SCALE_RE.match(name)
            if scale_match:
                key = (int(scale_match.group("layer")), int(scale_match.group("expert")))
                scales.setdefault(key, {})[scale_match.group("kind")] = info

    layer_experts: dict[int, set[int]] = {}
    for layer, expert in experts:
        layer_experts.setdefault(layer, set()).add(expert)

    expert_summary: list[dict[str, Any]] = []
    for key, parts in sorted(experts.items()):
        layer, expert = key
        expert_summary.append(
            {
                "layer": layer,
                "expert": expert,
                "weights": {kind: asdict(info) for kind, info in sorted(parts.items())},
                "scales": {
                    kind: asdict(info)
                    for kind, info in sorted(scales.get(key, {}).items())
                },
            }
        )

    result: dict[str, Any] = {
        "root": str(root),
        "shard_count": len(shards),
        "tensor_count": len(tensors),
        "expert_tensor_count": sum(len(parts) for parts in experts.values()),
        "scale_tensor_count": sum(len(parts) for parts in scales.values()),
        "expert_count": len(experts),
        "layer_count_with_experts": len(layer_experts),
        "experts_per_layer": {
            str(layer): len(experts_for_layer)
            for layer, experts_for_layer in sorted(layer_experts.items())
        },
        "model_metadata": _metadata(root),
        "index": {
            "present": index is not None,
            "weight_map_entries": len(index.get("weight_map", {})) if index else 0,
        },
        "shards": [path.name for path in shards],
        "experts": expert_summary,
    }

    if show_tensors:
        result["tensors"] = [asdict(tensor) for tensor in tensors]

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Map Qwen3.6 FP8 Safetensors experts")
    parser.add_argument("root", type=Path, help="Directory containing the checkpoint")
    parser.add_argument("-o", "--output", type=Path, default=Path("fp8-model-map.json"))
    parser.add_argument("--show-tensors", action="store_true")
    args = parser.parse_args()

    result = inspect(args.root, show_tensors=args.show_tensors)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Checkpoint: {args.root.resolve()}")
    print(f"Shards: {result['shard_count']}")
    print(f"Tensores: {result['tensor_count']}")
    print(f"Tensores de experts: {result['expert_tensor_count']}")
    print(f"Tensores de escala: {result['scale_tensor_count']}")
    print(f"Experts mapeados: {result['expert_count']}")
    print(f"Camadas com experts: {result['layer_count_with_experts']}")
    print(f"Mapa salvo em: {args.output}")

    if args.show_tensors:
        print("\nTensor names:")
        for index, tensor in enumerate(result["tensors"]):
            print(
                f"[{index:04d}] {tensor['name']} | {tensor['shape']} | "
                f"{tensor['dtype']} | {tensor['shard']}"
            )


if __name__ == "__main__":
    main()
