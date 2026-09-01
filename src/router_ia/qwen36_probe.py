from __future__ import annotations

"""Inspect real Qwen3.6 Safetensors tensors for the first forward pass.

The probe works directly on the Safetensors checkpoint directory. It reads
only Safetensors headers, so no model weights are loaded into RAM/VRAM.
"""

import argparse
import json
import re
import struct
from pathlib import Path
from typing import Any


# Qwen3.6 FP8 checkpoint names are of the form:
# model.language_model.layers.N....
LAYER_RE = re.compile(r"(?:^|\\.)layers\\.(\\d+)(?:\\.|$)", re.IGNORECASE)
ROUTER_RE = re.compile(r"(?:^|\\.)(?:mlp\\.)gate\\.(?:weight|bias)$|(?:router|gate_inp|router_logits|shared_expert_gate)", re.IGNORECASE)
EMBED_RE = re.compile(r"(?:^|\\.)(?:embed_tokens|token_embed|word_embeddings|token_embedding)\\.weight$", re.IGNORECASE)
OUTPUT_RE = re.compile(r"(?:^|\\.)(?:lm_head|output)\\.weight$", re.IGNORECASE)
NORM_RE = re.compile(r"norm", re.IGNORECASE)
ATTN_RE = re.compile(r"(?:self_attn|linear_attn|attention|attn|deltanet|delta_net|ssm)", re.IGNORECASE)
EXPERT_RE = re.compile(
    r"(?:^|\\.)layers\\.(?P<layer>\\d+)\\.mlp\\.experts\\.(?P<expert>\\d+)\\.(?P<kind>gate_proj|up_proj|down_proj)\\.weight$",
    re.IGNORECASE,
)


def _read_header(path: Path) -> dict[str, Any]:
    """Read only the Safetensors header."""
    with path.open("rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError(f"Invalid Safetensors file: {path}")
        header_len = struct.unpack("<Q", raw)[0]
        header = fh.read(header_len)
        if len(header) != header_len:
            raise ValueError(f"Truncated Safetensors header: {path}")
    value = json.loads(header.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid Safetensors header object: {path}")
    return value


def discover_shards(root: Path) -> list[Path]:
    """Find shards using the HF index when available, otherwise all shards."""
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        names = sorted(set(payload.get("weight_map", {}).values()))
        shards = [root / name for name in names if (root / name).is_file()]
        if shards:
            return shards

    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"No Safetensors shards found in: {root}")
    return shards


def read_shard_tensors(shard: Path) -> list[tuple[str, Path, str, tuple[int, ...]]]:
    """Return tensor metadata directly from the Safetensors header."""
    header = _read_header(shard)
    records: list[tuple[str, Path, str, tuple[int, ...]]] = []
    for name, meta in header.items():
        if name == "__metadata__" or not isinstance(meta, dict):
            continue
        dtype = str(meta.get("dtype", "?"))
        shape = tuple(int(x) for x in meta.get("shape", ()))
        records.append((name, shard, dtype, shape))
    return records


def collect_metadata(root: Path, shards: list[Path]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        metadata["weight_map_entries"] = len(payload.get("weight_map", {}))
        metadata["index_metadata"] = payload.get("metadata", {})

    shard_metadata: dict[str, Any] = {}
    for shard in shards:
        header = _read_header(shard)
        meta = header.get("__metadata__", {})
        if isinstance(meta, dict):
            shard_metadata.update(meta)
    metadata["safetensors_metadata"] = shard_metadata
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Qwen3.6 Safetensors tensors")
    parser.add_argument("root", type=Path, help="Directory containing Qwen3.6 .safetensors shards")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--all-layers", action="store_true", help="Print every transformer layer")
    parser.add_argument("--all-tensors", action="store_true", help="Print all tensor names")
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Safetensors directory not found: {root}")

    shards = discover_shards(root)
    metadata = collect_metadata(root, shards)

    tensor_records: list[tuple[str, Path, str, tuple[int, ...]]] = []
    for shard in shards:
        tensor_records.extend(read_shard_tensors(shard))

    print(f"Safetensors directory: {root}")
    print(f"Shards: {len(shards)}")
    print(f"Tensors: {len(tensor_records)}")

    if metadata.get("index_metadata"):
        print("\n[INDEX METADATA]")
        for key, value in metadata["index_metadata"].items():
            print(f"  {key} = {value}")

    router = [r for r in tensor_records if ROUTER_RE.search(r[0])]
    embeddings = [r for r in tensor_records if EMBED_RE.search(r[0])]
    outputs = [r for r in tensor_records if OUTPUT_RE.search(r[0])]
    experts = [r for r in tensor_records if EXPERT_RE.search(r[0])]

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

    layer_experts: dict[int, set[int]] = {}
    for name, _shard, _dtype, _shape in experts:
        match = EXPERT_RE.search(name)
        if match:
            layer = int(match.group("layer"))
            expert = int(match.group("expert"))
            layer_experts.setdefault(layer, set()).add(expert)

    print("\n[EXPERT SUMMARY]")
    for layer in sorted(layer_experts):
        ids = sorted(layer_experts[layer])
        print(f"  layer {layer}: {len(ids)} experts; range={ids[0]}..{ids[-1]}")
    print(f"  expert weight tensors: {len(experts)}")

    selected_layers = sorted(layer_experts) if args.all_layers else [args.layer]
    for layer in selected_layers:
        candidates = [r for r in tensor_records if LAYER_RE.search(r[0]) and int(LAYER_RE.search(r[0]).group(1)) == layer]
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
    selected_router = []
    for record in tensor_records:
        name = record[0]
        match = LAYER_RE.search(name)
        if match and int(match.group(1)) == args.layer and ROUTER_RE.search(name):
            selected_router.append(record)
    for name, shard, dtype, shape in selected_router:
        print(f"  {name} shape={shape} dtype={dtype} shard={shard.name}")
    if not selected_router:
        print("  <none>")

    if args.all_tensors:
        print("\n[ALL TENSORS]")
        for name, shard, dtype, shape in tensor_records:
            print(f"  {name} shape={shape} dtype={dtype} shard={shard.name}")


if __name__ == "__main__":
    main()
