from __future__ import annotations

"""Inspect real Qwen3.6 Safetensors tensors for the first forward pass.

The current FP8 expert runner uses the original Safetensors checkpoint, so
this probe intentionally works on a checkpoint directory rather than GGUF.
It discovers metadata from model.safetensors.index.json when available and
scans shard headers without loading tensor payloads into memory.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

from safetensors import safe_open


ROUTER_RE = re.compile(r"(?:router|gate_inp|router_logits|e_score|score)", re.IGNORECASE)
EMBED_RE = re.compile(r"(?:token_embed|embed_tokens|word_embeddings|token_embedding)", re.IGNORECASE)
OUTPUT_RE = re.compile(r"(?:lm_head|output\.weight|output_weight)$", re.IGNORECASE)
NORM_RE = re.compile(r"norm", re.IGNORECASE)
ATTN_RE = re.compile(r"(?:self_attn|attention|attn|deltanet|delta_net|linear_attn|ssm)", re.IGNORECASE)
EXPERT_RE = re.compile(r"(?:^|\.)experts\.(\d+)\.(?:gate_proj|up_proj|down_proj)\.weight$", re.IGNORECASE)
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.", re.IGNORECASE)


def _dtype_shape(handle: Any, name: str) -> tuple[str, tuple[int, ...]]:
    info = handle.get_tensor(name, device="meta")
    return str(info.dtype), tuple(int(x) for x in info.shape)


def discover_shards(root: Path) -> list[Path]:
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


def collect_metadata(root: Path, shards: list[Path]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        metadata["index_keys"] = sorted(payload.keys())
        metadata["weight_map_entries"] = len(payload.get("weight_map", {}))

    # Safetensors metadata is available from the header without loading the
    # actual tensor data. Keep only likely model/config fields for readability.
    model_meta: dict[str, Any] = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="meta") as handle:
            meta = handle.metadata() or {}
            for key, value in meta.items():
                low = key.lower()
                if any(token in low for token in ("model", "architecture", "transform", "qwen", "layer", "expert", "hidden")):
                    model_meta[key] = value
    metadata["safetensors_metadata"] = model_meta
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Qwen3.6 Safetensors tensors")
    parser.add_argument("root", type=Path, help="Directory containing Qwen3.6 .safetensors shards")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--all-layers", action="store_true", help="Print a compact summary for every transformer layer")
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Safetensors directory not found: {root}")

    shards = discover_shards(root)
    metadata = collect_metadata(root, shards)

    tensor_records: list[tuple[str, Path, str, tuple[int, ...]]] = []
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="meta") as handle:
            for name in handle.keys():
                dtype, shape = _dtype_shape(handle, name)
                tensor_records.append((name, shard, dtype, shape))

    print(f"Safetensors directory: {root}")
    print(f"Shards: {len(shards)}")
    print(f"Tensors: {len(tensor_records)}")
    if metadata["safetensors_metadata"]:
        print("\n[SAFETENSORS METADATA]")
        for key, value in sorted(metadata["safetensors_metadata"].items()):
            print(f"  {key} = {value}")

    router = [record for record in tensor_records if ROUTER_RE.search(record[0])]
    embeddings = [record for record in tensor_records if EMBED_RE.search(record[0])]
    outputs = [record for record in tensor_records if OUTPUT_RE.search(record[0])]
    experts = [record for record in tensor_records if EXPERT_RE.search(record[0])]

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
        match = EXPERT_RE.search(name)
        layer_match = LAYER_RE.search(name)
        if match and layer_match:
            layer = int(layer_match.group(1))
            expert = int(match.group(1))
            layer_experts.setdefault(layer, set()).add(expert)
    for layer in sorted(layer_experts):
        ids = sorted(layer_experts[layer])
        print(f"  layer {layer}: {len(ids)} experts; range={ids[0]}..{ids[-1]}")
    print(f"  expert weight tensors: {len(experts)}")

    layers_to_show = sorted(layer_experts) if args.all_layers else [args.layer]
    for layer in layers_to_show:
        prefix = f"layers.{layer}."
        candidates = [r for r in tensor_records if r[0].startswith(prefix)]
        print(f"\n[LAYER {layer} TENSORS]")
        if not candidates:
            print("  <none>")
            continue
        for name, shard, dtype, shape in candidates:
            tag_parts = []
            if NORM_RE.search(name):
                tag_parts.append("NORM")
            if ATTN_RE.search(name):
                tag_parts.append("ATTN")
            if ROUTER_RE.search(name):
                tag_parts.append("ROUTER")
            if EXPERT_RE.search(name):
                tag_parts.append("EXPERT")
            tag = f" [{' '.join(tag_parts)}]" if tag_parts else ""
            print(f"  {name} shape={shape} dtype={dtype} shard={shard.name}{tag}")

    print("\n[LIKELY ROUTER TENSORS IN SELECTED LAYER]")
    selected = [r for r in tensor_records if r[0].startswith(f"layers.{args.layer}.") and ROUTER_RE.search(r[0])]
    for name, shard, dtype, shape in selected:
        print(f"  {name} shape={shape} dtype={dtype} shard={shard.name}")
    if not selected:
        print("  <none>")


if __name__ == "__main__":
    main()
