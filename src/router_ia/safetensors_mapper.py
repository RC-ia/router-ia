from __future__ import annotations

"""Inspect Qwen3.6 FP8 Safetensors shards without loading the model.

The official Qwen3.6-35B-A3B-FP8 checkpoint uses Hugging Face Safetensors
shards and fine-grained block-wise FP8 weights. This mapper reads the
Safetensors headers/index and builds a compact inventory of MoE experts.

It deliberately does not instantiate the model or copy all weights to RAM.
"""

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

try:
    from safetensors import safe_open
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "safetensors is required. Install with: python -m pip install safetensors"
    ) from exc


EXPERT_PATTERNS = (
    re.compile(
        r"^(?P<prefix>.*?)(?:model\.layers\.)?(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\.(?P<kind>gate_proj|up_proj|down_proj)\.weight$"
    ),
    re.compile(
        r"^(?P<prefix>.*?)(?:layers\.)(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\.(?P<kind>gate_proj|up_proj|down_proj)\.weight$"
    ),
    re.compile(
        r"^(?P<prefix>.*?)(?:layers\.)(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\.(?P<kind>gate_proj|up_proj|down_proj)\.weight_scale_inv$"
    ),
)


@dataclass
class TensorInfo:
    name: str
    shape: list[int]
    dtype: str
    shard: str


def _expert_match(name: str) -> re.Match[str] | None:
    for pattern in EXPERT_PATTERNS:
        match = pattern.match(name)
        if match and match.group("kind") in {"gate_proj", "up_proj", "down_proj"}:
            return match
    return None


def _list_safetensors(root: Path) -> list[Path]:
    return sorted(root.glob("*.safetensors"))


def _load_index(root: Path) -> dict[str, Any] | None:
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        return None
    return json.loads(index_path.read_text(encoding="utf-8"))


def _metadata(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    config = root / "config.json"
    if config.is_file():
        try:
            result["config"] = json.loads(config.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            result["config"] = {"error": "invalid config.json"}
    return result


def inspect(root: Path, *, show_tensors: bool = False) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")

    index = _load_index(root)
    shards = _list_safetensors(root)
    if index and isinstance(index.get("weight_map"), dict):
        shard_names = sorted(set(index["weight_map"].values()))
        shards = [root / name for name in shard_names if (root / name).is_file()]

    if not shards:
        raise FileNotFoundError(f"No .safetensors shards found in {root}")

    tensors: list[TensorInfo] = []
    experts: dict[tuple[int, int], dict[str, TensorInfo]] = {}
    scale_tensors: dict[tuple[int, int], dict[str, TensorInfo]] = {}

    for shard in shards:
        with safe_open(str(shard), framework="pt", device="meta") as handle:
            for name in handle.keys():
                info = TensorInfo(
                    name=name,
                    shape=list(handle.get_slice(name).get_shape()),
                    dtype=str(handle.get_slice(name).dtype),
                    shard=shard.name,
                )
                tensors.append(info)

                match = _expert_match(name)
                if not match:
                    # Handle associated FP8 scale tensors separately.
                    scale_match = re.match(
                        r"^(?P<prefix>.*?)(?:model\.layers\.)?(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\.(?P<kind>gate_proj|up_proj|down_proj)\.weight_scale_inv$",
                        name,
                    )
                    if scale_match:
                        key = (int(scale_match.group("layer")), int(scale_match.group("expert")))
                        scale_tensors.setdefault(key, {})[scale_match.group("kind")] = info
                    continue

                key = (int(match.group("layer")), int(match.group("expert")))
                experts.setdefault(key, {})[match.group("kind")] = info

    layers: dict[int, set[int]] = {}
    for layer, expert in experts:
        layers.setdefault(layer, set()).add(expert)

    complete = sum(
        1
        for layer in layers
        if sum(1 for (l, _e) in experts if l == layer) > 0
    )

    expert_summary: list[dict[str, Any]] = []
    for (layer, expert), parts in sorted(experts.items()):
        scales = scale_tensors.get((layer, expert), {})
        expert_summary.append(
            {
                "layer": layer,
                "expert": expert,
                "weights": {kind: asdict(info) for kind, info in sorted(parts.items())},
                "scales": {kind: asdict(info) for kind, info in sorted(scales.items())},
            }
        )

    result = {
        "root": str(root),
        "shard_count": len(shards),
        "tensor_count": len(tensors),
        "expert_tensor_count": sum(len(parts) for parts in experts.values()),
        "expert_count": len(experts),
        "layer_count_with_experts": len(layers),
        "layers": {str(layer): sorted(experts_for_layer) for layer, experts_for_layer in sorted(layers.items())},
        "model_metadata": _metadata(root),
        "index": {
            "present": index is not None,
            "weight_map_entries": len(index.get("weight_map", {})) if index else 0,
        },
        "experts": expert_summary,
    }

    if show_tensors:
        result["tensors"] = [asdict(tensor) for tensor in tensors]

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Map Qwen3.6 FP8 Safetensors experts")
    parser.add_argument("root", type=Path, help="Directory containing the Safetensors checkpoint")
    parser.add_argument("-o", "--output", type=Path, default=Path("fp8-model-map.json"))
    parser.add_argument("--show-tensors", action="store_true")
    args = parser.parse_args()

    result = inspect(args.root, show_tensors=args.show_tensors)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Checkpoint: {args.root.resolve()}")
    print(f"Shards: {result['shard_count']}")
    print(f"Tensores: {result['tensor_count']}")
    print(f"Tensores de experts: {result['expert_tensor_count']}")
    print(f"Experts mapeados: {result['expert_count']}")
    print(f"Camadas com experts: {result['layer_count_with_experts']}")
    print(f"Mapa salvo em: {args.output}")

    if args.show_tensors:
        print("\nTensor names:")
        for index, tensor in enumerate(result["tensors"]):
            print(f"[{index:04d}] {tensor['name']} | {tensor['shape']} | {tensor['dtype']} | {tensor['shard']}")


if __name__ == "__main__":
    main()
