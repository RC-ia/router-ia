from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gguf import GGUFReader


@dataclass
class TensorInfo:
    name: str
    shape: list[int]
    dtype: str
    nbytes: int


EXPERT_TENSOR_RE = re.compile(
    r"^blk\.(?P<layer>\d+)\.ffn_(?P<kind>gate|up|down)_exps\.weight$"
)


def json_safe(value: Any) -> Any:
    """Convert arbitrary GGUF/numpy values into JSON-compatible Python types."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return json_safe(item())
        except (TypeError, ValueError):
            pass

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return json_safe(tolist())
        except (TypeError, ValueError):
            pass

    return str(value)


def tensor_nbytes(tensor) -> int:
    value = getattr(tensor, "nbytes", None)
    if value is not None:
        return int(value)

    data = getattr(tensor, "data", None)
    if data is not None:
        return int(data.nbytes)

    return 0


def inspect(path: Path) -> dict:
    reader = GGUFReader(str(path))

    metadata: dict[str, Any] = {}
    for key, field in reader.fields.items():
        value = getattr(field, "parts", None)
        if value:
            metadata[key] = json_safe(value[-1])

    tensors: list[TensorInfo] = []
    expert_tensors: list[dict[str, Any]] = []

    for tensor in reader.tensors:
        info = TensorInfo(
            name=tensor.name,
            shape=[int(x) for x in tensor.shape],
            dtype=str(tensor.tensor_type),
            nbytes=tensor_nbytes(tensor),
        )
        tensors.append(info)

        match = EXPERT_TENSOR_RE.match(tensor.name)
        if match:
            shape = info.shape
            expert_count = shape[-1] if shape else None
            bytes_per_expert = None
            if expert_count and info.nbytes:
                bytes_per_expert = info.nbytes // expert_count

            expert_tensors.append(
                {
                    "layer": int(match.group("layer")),
                    "kind": match.group("kind"),
                    "tensor": info.name,
                    "shape": shape,
                    "dtype": info.dtype,
                    "nbytes": info.nbytes,
                    "expert_count_from_shape": expert_count,
                    "bytes_per_expert_if_last_axis": bytes_per_expert,
                }
            )

    expert_layers: dict[str, dict[str, Any]] = {}
    for item in expert_tensors:
        layer = str(item["layer"])
        entry = expert_layers.setdefault(layer, {})
        entry[item["kind"]] = item

    complete_layers = sum(
        1
        for entry in expert_layers.values()
        if {"gate", "up", "down"}.issubset(entry)
    )

    inferred_expert_counts = sorted(
        {
            item["expert_count_from_shape"]
            for item in expert_tensors
            if item["expert_count_from_shape"] is not None
        }
    )

    return {
        "file": str(path),
        "metadata": metadata,
        "tensor_count": len(tensors),
        "expert_tensor_count": len(expert_tensors),
        "expert_layer_count": len(expert_layers),
        "complete_expert_layers": complete_layers,
        "inferred_expert_counts": inferred_expert_counts,
        "expert_tensors": expert_tensors,
        "expert_layers": expert_layers,
        "tensor_names": [x.name for x in tensors],
        "tensors": [asdict(x) for x in tensors],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect GGUF tensors and map packed MoE expert tensors."
    )
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("model-map.json"),
    )
    parser.add_argument(
        "--show-tensors",
        action="store_true",
        help="Print every tensor name after inspection.",
    )
    parser.add_argument(
        "--show-experts",
        action="store_true",
        help="Print only packed MoE expert tensors and their shapes/sizes.",
    )

    args = parser.parse_args()

    if not args.model.is_file():
        raise SystemExit(f"Arquivo não encontrado: {args.model}")

    result = inspect(args.model)

    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"GGUF: {args.model}")
    print(f"Tensores: {result['tensor_count']}")
    print(f"Tensores de experts: {result['expert_tensor_count']}")
    print(f"Camadas MoE completas: {result['complete_expert_layers']}")
    print(f"Experts inferidos pela última dimensão: {result['inferred_expert_counts']}")
    print(f"Mapa salvo em: {args.output}")

    if args.show_experts:
        print("\nExpert tensors:")
        for item in result["expert_tensors"]:
            mib = item["nbytes"] / (1024**2)
            per_expert = item["bytes_per_expert_if_last_axis"]
            per_expert_mib = (
                per_expert / (1024**2) if per_expert is not None else None
            )
            if per_expert_mib is not None:
                print(
                    f"layer={item['layer']:02d} "
                    f"kind={item['kind']:4s} "
                    f"shape={item['shape']} "
                    f"dtype={item['dtype']} "
                    f"size={mib:.2f} MiB "
                    f"~per-expert={per_expert_mib:.2f} MiB"
                )
            else:
                print(
                    f"layer={item['layer']:02d} "
                    f"kind={item['kind']:4s} "
                    f"shape={item['shape']} "
                    f"dtype={item['dtype']} "
                    f"size={mib:.2f} MiB"
                )

    if args.show_tensors:
        print("\nTensor names:")
        for index, name in enumerate(result["tensor_names"]):
            print(f"[{index:03d}] {name}")


if __name__ == "__main__":
    main()
