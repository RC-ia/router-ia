from __future__ import annotations

"""Inspect the tensor structure of one Qwen3.6 layer in Safetensors."""

import argparse
import json
from pathlib import Path

from safetensors import safe_open


def discover_shards(root: Path) -> list[Path]:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        names = sorted(set(payload.get("weight_map", {}).values()))
        shards = [root / name for name in names if (root / name).is_file()]
        if shards:
            return shards
    return sorted(root.glob("*.safetensors"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one Qwen3.6 layer")
    parser.add_argument("root", type=Path)
    parser.add_argument("--layer", type=int, default=1)
    args = parser.parse_args()

    prefix = f"model.language_model.layers.{args.layer}."
    found: list[tuple[str, tuple[int, ...], str]] = []
    seen: set[str] = set()

    for shard in discover_shards(args.root.resolve()):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if not name.startswith(prefix) or name in seen:
                    continue
                sl = handle.get_slice(name)
                found.append((name, tuple(int(x) for x in sl.get_shape()), str(sl.get_dtype())))
                seen.add(name)

    found.sort()
    print(f"layer: {args.layer}")
    print(f"tensor count: {len(found)}")
    for name, shape, dtype in found:
        print(f"{name} shape={shape} dtype={dtype}")


if __name__ == "__main__":
    main()
