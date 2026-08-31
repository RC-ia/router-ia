from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

from gguf import GGUFReader


EXPERT_PATTERNS = (
    re.compile(r"(?:^|\.)experts?\.(\d+)\.(?:.*)$", re.IGNORECASE),
    re.compile(r"(?:^|\.)(?:expert|experts)[_/](\d+)(?:[./_]|$)", re.IGNORECASE),
)
LAYER_PATTERNS = (
    re.compile(r"(?:^|\.)blk\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)layers?\.(\d+)(?:\.|$)", re.IGNORECASE),
)


def infer_layer(name: str) -> int | None:
    for pattern in LAYER_PATTERNS:
        match = pattern.search(name)
        if match:
            return int(match.group(1))
    return None


def infer_expert(name: str) -> int | None:
    for pattern in EXPERT_PATTERNS:
        match = pattern.search(name)
        if match:
            return int(match.group(1))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a GGUF file and detect MoE expert tensors.")
    parser.add_argument("model", type=Path)
    args = parser.parse_args()

    if not args.model.is_file():
        raise SystemExit(f"GGUF not found: {args.model}")

    reader = GGUFReader(str(args.model))

    print(f"file: {args.model}")
    print(f"tensors: {len(reader.tensors)}")

    metadata = reader.fields
    for key in sorted(metadata):
        if any(word in key.lower() for word in ("architecture", "expert", "layer", "context", "block")):
            field = metadata[key]
            value = getattr(field, "parts", field)
            print(f"meta.{key} = {value}")

    by_layer: dict[int, set[int]] = defaultdict(set)
    expert_tensors = []
    tensor_prefixes = Counter()

    for tensor in reader.tensors:
        name = tensor.name
        layer = infer_layer(name)
        expert = infer_expert(name)

        if expert is not None:
            expert_tensors.append(name)
            if layer is not None:
                by_layer[layer].add(expert)

        prefix = name.split(".", 1)[0]
        tensor_prefixes[prefix] += 1

    print(f"expert-like tensors: {len(expert_tensors)}")
    if by_layer:
        print("expert groups by layer:")
        for layer in sorted(by_layer):
            experts = sorted(by_layer[layer])
            print(f"  layer {layer}: {len(experts)} experts ({experts[:12]}{'...' if len(experts) > 12 else ''})")
    else:
        print("No experts detected by the generic naming patterns.")
        print("First tensor names:")
        for tensor in reader.tensors[:40]:
            print(f"  {tensor.name}")

    print("tensor top-level prefixes:")
    for prefix, count in tensor_prefixes.most_common():
        print(f"  {prefix}: {count}")


if __name__ == "__main__":
    main()
