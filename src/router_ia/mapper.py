from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from gguf import GGUFReader


@dataclass
class TensorInfo:
    name: str
    shape: list[int]
    dtype: str
    nbytes: int


@dataclass
class ExpertInfo:
    layer: int
    expert: int
    tensors: list[TensorInfo]
    total_bytes: int


EXPERT_PATTERNS = [
    # padrões comuns de MoE; não assumimos que todos serão usados
    re.compile(r"(?:blk|block)\.(\d+).*?expert.*?(\d+)", re.I),
    re.compile(r"(?:blk|block)\.(\d+).*?experts.*?(\d+)", re.I),
]


def tensor_nbytes(tensor) -> int:
    # O gguf.py expõe nbytes nas versões atuais.
    value = getattr(tensor, "nbytes", None)

    if value is not None:
        return int(value)

    # Fallback conservador.
    data = getattr(tensor, "data", None)
    if data is not None:
        return int(data.nbytes)

    return 0


def detect_expert(name: str) -> tuple[int, int] | None:
    for pattern in EXPERT_PATTERNS:
        match = pattern.search(name)
        if match:
            return int(match.group(1)), int(match.group(2))

    return None


def inspect(path: Path) -> dict:
    reader = GGUFReader(str(path))

    metadata = {}
    for key, field in reader.fields.items():
        value = getattr(field, "parts", None)
        if value:
            metadata[key] = value[-1]

    tensors: list[TensorInfo] = []
    experts: dict[tuple[int, int], list[TensorInfo]] = defaultdict(list)

    for tensor in reader.tensors:
        info = TensorInfo(
            name=tensor.name,
            shape=[int(x) for x in tensor.shape],
            dtype=str(tensor.tensor_type),
            nbytes=tensor_nbytes(tensor),
        )

        tensors.append(info)

        expert_id = detect_expert(tensor.name)
        if expert_id is not None:
            experts[expert_id].append(info)

    expert_map = {}

    for (layer, expert), items in sorted(experts.items()):
        expert_map[f"{layer}:{expert}"] = asdict(
            ExpertInfo(
                layer=layer,
                expert=expert,
                tensors=items,
                total_bytes=sum(x.nbytes for x in items),
            )
        )

    return {
        "file": str(path),
        "metadata": metadata,
        "tensor_count": len(tensors),
        "expert_count": len(expert_map),
        "experts": expert_map,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect GGUF tensors and build a MoE expert map."
    )
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("model-map.json"),
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
    print(f"Experts encontrados: {result['expert_count']}")
    print(f"Mapa salvo em: {args.output}")


if __name__ == "__main__":
    main()
