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
    data_offset: int | None = None


EXPERT_TENSOR_RE = re.compile(
    r"^blk\.(?P<layer>\d+)\.ffn_(?P<kind>gate|up|down)_exps\.weight$"
)


# Qwen3.6 stores routed experts packed along GGML axis 2.  The gguf-py
# ReaderTensor exposes the tensor's absolute data_offset and n_bytes, so the
# byte region for expert X is:
#
#     data_offset + X * (n_bytes / n_experts)
#
# The important point is that we do NOT decode/dequantize the tensor here.
# This map describes where the already-quantized expert bytes live in GGUF.
EXPERT_AXIS = 2


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
    value = getattr(tensor, "n_bytes", None)
    if value is not None:
        return int(value)

    value = getattr(tensor, "nbytes", None)
    if value is not None:
        return int(value)

    data = getattr(tensor, "data", None)
    if data is not None:
        return int(data.nbytes)

    return 0


def tensor_data_offset(tensor) -> int | None:
    value = getattr(tensor, "data_offset", None)
    if value is None:
        return None
    return int(value)


def build_expert_entry(
    *,
    layer: int,
    expert: int,
    sources: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build a single logical (layer, expert) entry from packed tensors."""
    tensors: dict[str, dict[str, Any]] = {}
    total_bytes = 0

    for kind in ("gate", "up", "down"):
        source = sources[kind]
        expert_count = source["expert_count"]
        nbytes = source["nbytes"]
        data_offset = source["data_offset"]

        if expert >= expert_count:
            raise ValueError(
                f"Expert {expert} is outside {kind} tensor range "
                f"(count={expert_count}) at layer {layer}"
            )

        if nbytes % expert_count != 0:
            raise ValueError(
                f"Tensor {source['tensor']} has {nbytes} bytes, which is not "
                f"divisible by {expert_count} experts"
            )

        bytes_per_expert = nbytes // expert_count
        offset = None
        end_offset = None

        if data_offset is not None:
            offset = data_offset + expert * bytes_per_expert
            end_offset = offset + bytes_per_expert

        tensors[kind] = {
            "tensor": source["tensor"],
            "axis": EXPERT_AXIS,
            "index": expert,
            "shape": source["shape"],
            "dtype": source["dtype"],
            "bytes": bytes_per_expert,
            "data_offset": data_offset,
            "offset": offset,
            "end_offset": end_offset,
        }
        total_bytes += bytes_per_expert

    return {
        "layer": layer,
        "expert": expert,
        "total_bytes": total_bytes,
        "total_mib": round(total_bytes / (1024**2), 6),
        "tensors": tensors,
    }


def inspect(path: Path, *, expand_experts: bool = True) -> dict:
    reader = GGUFReader(str(path))

    metadata: dict[str, Any] = {}
    for key, field in reader.fields.items():
        value = getattr(field, "parts", None)
        if value:
            metadata[key] = json_safe(value[-1])

    tensors: list[TensorInfo] = []
    packed: dict[int, dict[str, dict[str, Any]]] = {}

    for tensor in reader.tensors:
        info = TensorInfo(
            name=tensor.name,
            shape=[int(x) for x in tensor.shape],
            dtype=str(tensor.tensor_type),
            nbytes=tensor_nbytes(tensor),
            data_offset=tensor_data_offset(tensor),
        )
        tensors.append(info)

        match = EXPERT_TENSOR_RE.match(tensor.name)
        if not match:
            continue

        layer = int(match.group("layer"))
        kind = match.group("kind")
        shape = info.shape

        if len(shape) != 3:
            raise ValueError(
                f"Packed expert tensor {info.name} is expected to be 3D, "
                f"got shape={shape}"
            )

        expert_count = shape[EXPERT_AXIS]
        if expert_count <= 0:
            raise ValueError(
                f"Packed expert tensor {info.name} has invalid expert count "
                f"{expert_count}"
            )

        layer_entry = packed.setdefault(layer, {})
        layer_entry[kind] = {
            "tensor": info.name,
            "shape": shape,
            "dtype": info.dtype,
            "nbytes": info.nbytes,
            "data_offset": info.data_offset,
            "expert_count": expert_count,
        }

    complete_layers = sorted(
        layer for layer, entry in packed.items() if {"gate", "up", "down"}.issubset(entry)
    )

    expert_counts = sorted(
        {
            int(source["expert_count"])
            for entry in packed.values()
            for source in entry.values()
        }
    )

    expert_layers: dict[str, Any] = {}

    for layer in complete_layers:
        sources = packed[layer]
        count = next(iter(sources.values()))["expert_count"]

        if any(source["expert_count"] != count for source in sources.values()):
            raise ValueError(
                f"Layer {layer} has inconsistent expert counts: "
                f"{[source['expert_count'] for source in sources.values()]}"
            )

        layer_info: dict[str, Any] = {
            "expert_count": count,
            "tensor_sources": sources,
        }

        if expand_experts:
            layer_info["experts"] = {
                str(expert): build_expert_entry(
                    layer=layer,
                    expert=expert,
                    sources=sources,
                )
                for expert in range(count)
            }

        expert_layers[str(layer)] = layer_info

    total_experts = sum(
        int(info["expert_count"]) for info in expert_layers.values()
    )

    return {
        "file": str(path),
        "metadata": metadata,
        "tensor_count": len(tensors),
        "expert_tensor_count": sum(len(entry) for entry in packed.values()),
        "expert_layer_count": len(expert_layers),
        "complete_expert_layers": len(complete_layers),
        "inferred_expert_counts": expert_counts,
        "expert_count_total": total_experts,
        "expert_axis": EXPERT_AXIS,
        "expert_layers": expert_layers,
        "tensor_names": [x.name for x in tensors],
        "tensors": [asdict(x) for x in tensors],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect GGUF tensors and build a packed MoE expert map."
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
        help="Print packed MoE tensor information and expert slices.",
    )
    parser.add_argument(
        "--no-expand-experts",
        action="store_true",
        help="Do not write 256 individual expert entries per layer to JSON.",
    )

    args = parser.parse_args()

    if not args.model.is_file():
        raise SystemExit(f"Arquivo não encontrado: {args.model}")

    result = inspect(
        args.model,
        expand_experts=not args.no_expand_experts,
    )

    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"GGUF: {args.model}")
    print(f"Tensores: {result['tensor_count']}")
    print(f"Tensores de experts: {result['expert_tensor_count']}")
    print(f"Camadas MoE completas: {result['complete_expert_layers']}")
    print(f"Experts por camada: {result['inferred_expert_counts']}")
    print(f"Experts totais mapeados: {result['expert_count_total']}")
    print(f"Eixo dos experts: {result['expert_axis']}")
    print(f"Mapa salvo em: {args.output}")

    if args.show_experts:
        print("\nPacked expert tensors:")
        for layer in result["expert_layers"].values():
            print(f"\nLayer {layer['experts']['0']['layer'] if 'experts' in layer else '?'}")
            for kind in ("gate", "up", "down"):
                source = layer["tensor_sources"][kind]
                mib = source["nbytes"] / (1024**2)
                per_expert = source["nbytes"] // source["expert_count"]
                per_expert_mib = per_expert / (1024**2)
                print(
                    f"  {kind:4s} {source['tensor']} "
                    f"shape={source['shape']} dtype={source['dtype']} "
                    f"size={mib:.2f} MiB "
                    f"per-expert={per_expert_mib:.4f} MiB "
                    f"data_offset={source['data_offset']}"
                )

            if "experts" in layer:
                for expert_id in (0, layer["expert_count"] - 1):
                    expert = layer["experts"][str(expert_id)]
                    print(
                        f"  example expert={expert_id} "
                        f"total={expert['total_bytes'] / (1024**2):.4f} MiB"
                    )
                    for kind, tensor in expert["tensors"].items():
                        print(
                            f"    {kind:4s} offset={tensor['offset']} "
                            f"end={tensor['end_offset']} "
                            f"bytes={tensor['bytes']}"
                        )

    if args.show_tensors:
        print("\nTensor names:")
        for index, name in enumerate(result["tensor_names"]):
            print(f"[{index:03d}] {name}")


if __name__ == "__main__":
    main()
