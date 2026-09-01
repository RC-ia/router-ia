from __future__ import annotations

"""Inspect real Qwen3.6 Safetensors tensors for the first forward pass.

This probe reads only Safetensors headers. It never materializes model weights,
so it works with low RAM/VRAM and avoids version-specific ``safe_open`` device
arguments.
"""

import argparse
import json
import re
import struct
from pathlib import Path
from typing import Any

ROUTER_RE = re.compile(r"(?:router|gate_inp|router_logits|e_score|score|route)", re.IGNORECASE)
EMBED_RE = re.compile(r"(?:token_embed|embed_tokens|word_embeddings|token_embedding)", re.IGNORECASE)
OUTPUT_RE = re.compile(r"(?:lm_head|output(?:_weight)?\.weight|output_weight)$", re.IGNORECASE)
NORM_RE = re.compile(r"norm", re.IGNORECASE)
ATTN_RE = re.compile(r"(?:self_attn|attention|attn|deltanet|delta_net|linear_attn|ssm)", re.IGNORECASE)
EXPERT_RE = re.compile(
    r"(?:^|\.)experts\.(?P<expert>\d+)\.(?:gate_proj|up_proj|down_proj)\.weight$",
    re.IGNORECASE,
)
LAYER_RE = re.compile(r"(?:^|\.)layers\.(?P<layer>\d+)\.", re.IGNORECASE)


def read_safetensors_header(path: Path) -> dict[str, Any]:
    """Read the JSON header without reading any tensor payload."""
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"Invalid Safetensors file (missing header length): {path}")
        header_size = struct.unpack("<Q", raw)[0]
        header = handle.read(header_size)
        if len(header) != header_size:
            raise ValueError(f"Truncated Safetensors header: {path}")
    value = json.loads(header.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid Safetensors header object: {path}")
    return value


def discover_shards(root: Path) -> list[Path]:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        names = sorted(set(payload.get("weight_map", {}).values()))
        shards = [root / str(name) for name in names if (root / str(name)).is_file()]
        if shards:
            return shards

    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"No Safetensors shards found in: {root}")
    return shards


def collect_metadata(root: Path, shards: list[Path]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        metadata["index_keys"] = sorted(payload.keys())
        metadata["weight_map_entries"] = len(payload.get("weight_map", {}))

    model_meta: dict[str, Any] = {}
    for shard in shards:
        header = read_safetensors_header(shard)
        meta = header.get("__metadata__", {})
        if not isinstance(meta, dict):
            continue
        for key, value in meta.items():
            low = str(key).lower()
            if any(token in low for token in ("model", "architecture", "transform", "qwen", "layer", "expert", "hidden")):
                model_meta[str(key)] = value
    metadata["safetensors_metadata"] = model_meta
    return metadata


def tensor_records(shards: list[Path]) -> list[tuple[str, Path, str, tuple[int, ...]]]:
    records: list[tuple[str, Path, str, tuple[int, ...]]] = []
    for shard in shards:
        header = read_safetensors_header(shard)
        for name, info in header.items():
            if name == "__metadata__" or not isinstance(info, dict):
                continue
            dtype = str(info.get("dtype", "?"))
            shape = tuple(int(x) for x in info.get("shape", ()))
            records.append((str(name), shard, dtype, shape))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Qwen3.6 Safetensors tensors")
    parser.add_argument("root", type=Path, help="Directory containing Qwen3.6 .safetensors shards")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--all-layers", action="store_true", help="Print every layer's tensors")
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Safetensors directory not found: {root}")

    shards = discover_shards(root)
    metadata = collect_metadata(root, shards)
    records = tensor_records(shards)

    print(f"Safetensors directory: {root}")
    print(f"Shards: {len(shards)}")
    print(f"Tensors: {len(records)}")

    if metadata["safetensors_metadata"]:
        print("\n[SAFETENSORS METADATA]")
        for key, value in sorted(metadata["safetensors_metadata"].items()):
            print(f"  {key} = {value}")

    router = [record for record in records if ROUTER_RE.search(record[0])]
    embeddings = [record for record in records if EMBED_RE.search(record[0])]
    outputs = [record for record in records if OUTPUT_RE.search(record[0])]
    experts = [record for record in records if EXPERT_RE.search(record[0])]

    print("\n[ROUTER / GATING CANDIDATES]")
    for name, shard, dtype, shape in router:
        print(f"  {name} shape={shape} dtype={dtype} shard={shard.name}")
    if not router:
        print("  <none>")

    print("\n[EMBEDDING CANDIDATES]")
    for name, shard, dtype, shape in embeddings:
        print(f"  {name} shape={shape} dtype={dtype} shard={shard.name}")
    if not embeddings:
        print("  <none>")

    print("\n[OUTPUT / LM HEAD CANDIDATES]")
    for name, shard, dtype, shape in outputs:
        print(f"  {name} shape={shape} dtype={dtype} shard={shard.name}")
    if not outputs:
        print("  <none>")

    print("\n[EXPERT SUMMARY]")
    layer_experts: dict[int, set[int]] = {}
    for name, _shard, _dtype, _shape in experts:
        expert_match = EXPERT_RE.search(name)
        layer_match = LAYER_RE.search(name)
        if expert_match and layer_match:
            layer = int(layer_match.group("layer"))
            expert = int(expert_match.group("expert"))
            layer_experts.setdefault(layer, set()).add(expert)

    for layer in sorted(layer_experts):
        ids = sorted(layer_experts[layer])
        print(f"  layer {layer}: {len(ids)} experts; range={ids[0]}..{ids[-1]}")
    print(f"  expert weight tensors: {len(experts)}")

    layers_to_show = sorted(layer_experts) if args.all_layers else [args.layer]
    for layer in layers_to_show:
        prefix = f"layers.{layer}."
        candidates = [record for record in records if record[0].startswith(prefix)]
        print(f"\n[LAYER {layer} TENSORS]")
        if not candidates:
            print("  <none>")
            continue
        for name, shard, dtype, shape in candidates:
            tags: list[str] = []
            if NORM_RE.search(name):
                tags.append("NORM")
            if ATTN_RE.search(name):
                tags.append("ATTN")
            if ROUTER_RE.search(name):
                tags.append("ROUTER")
            if EXPERT_RE.search(name):
                tags.append("EXPERT")
            tag = f" [{' '.join(tags)}]" if tags else ""
            print(f"  {name} shape={shape} dtype={dtype} shard={shard.name}{tag}")

    print("\n[LIKELY ROUTER TENSORS IN SELECTED LAYER]")
    selected = [
        record
        for record in records
        if record[0].startswith(f"layers.{args.layer}.") and ROUTER_RE.search(record[0])
    ]
    for name, shard, dtype, shape in selected:
        print(f"  {name} shape={shape} dtype={dtype} shard={shard.name}")
    if not selected:
        print("  <none>")


if __name__ == "__main__":
    main()
