from __future__ import annotations

"""Probe Qwen3.6 GGUF tensors needed to build the first real forward pass.

This deliberately does not execute the model yet. It resolves the model's
actual GGUF tensor names/shapes for embeddings, per-layer norms/attention,
and the MoE router. The output is intended to guide the next implementation
step without hard-coding names that may differ between converters.
"""

import argparse
import re
from pathlib import Path

from gguf import GGUFReader


ROUTER_RE = re.compile(r"(?:^|\.)(?:ffn_)?gate_inp|router|gate_inp", re.IGNORECASE)
EMBED_RE = re.compile(r"token_embd|embed", re.IGNORECASE)
OUTPUT_RE = re.compile(r"(?:^|\.)(?:output|out)\.weight$", re.IGNORECASE)
NORM_RE = re.compile(r"norm", re.IGNORECASE)
ATTN_RE = re.compile(r"attn|delta|ssm", re.IGNORECASE)
EXPERT_RE = re.compile(r"ffn_(?:gate|up|down)_exps\.weight$", re.IGNORECASE)


def describe(tensor) -> str:
    shape = tuple(int(x) for x in tensor.shape)
    dtype = str(tensor.tensor_type)
    nbytes = getattr(tensor, "n_bytes", None)
    if nbytes is None:
        data = getattr(tensor, "data", None)
        nbytes = getattr(data, "nbytes", None)
    return f"shape={shape} dtype={dtype} bytes={int(nbytes) if nbytes is not None else '?'}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Qwen3.6 GGUF tensors")
    parser.add_argument("model", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    args = parser.parse_args()

    if not args.model.is_file():
        raise SystemExit(f"GGUF not found: {args.model}")

    reader = GGUFReader(str(args.model))
    wanted_layer = args.layer
    prefix = f"blk.{wanted_layer}."

    print(f"GGUF: {args.model}")
    print(f"Tensors: {len(reader.tensors)}")
    print(f"Inspecting layer: {wanted_layer}")

    metadata = reader.fields
    for key in sorted(metadata):
        low = key.lower()
        if any(word in low for word in ("architecture", "block_count", "expert", "attention", "embedding", "vocab")):
            field = metadata[key]
            value = getattr(field, "parts", field)
            print(f"META {key} = {value}")

    groups: dict[str, list] = {
        "ROUTER_CANDIDATES": [],
        "EMBEDDING_CANDIDATES": [],
        "LAYER_CANDIDATES": [],
        "EXPERT_TENSORS": [],
        "OUTPUT_CANDIDATES": [],
    }

    for tensor in reader.tensors:
        name = tensor.name
        if ROUTER_RE.search(name):
            groups["ROUTER_CANDIDATES"].append(tensor)
        if EMBED_RE.search(name):
            groups["EMBEDDING_CANDIDATES"].append(tensor)
        if name.startswith(prefix):
            groups["LAYER_CANDIDATES"].append(tensor)
        if EXPERT_RE.search(name):
            groups["EXPERT_TENSORS"].append(tensor)
        if OUTPUT_RE.search(name):
            groups["OUTPUT_CANDIDATES"].append(tensor)

    for title in ("ROUTER_CANDIDATES", "EMBEDDING_CANDIDATES", "OUTPUT_CANDIDATES"):
        print(f"\n[{title}]")
        for tensor in groups[title]:
            print(f"  {tensor.name} {describe(tensor)}")
        if not groups[title]:
            print("  <none>")

    print(f"\n[LAYER {wanted_layer} TENSORS]")
    for tensor in groups["LAYER_CANDIDATES"]:
        print(f"  {tensor.name} {describe(tensor)}")

    if groups["EXPERT_TENSORS"]:
        print("\n[EXPERT SUMMARY]")
        for tensor in groups["EXPERT_TENSORS"][:12]:
            print(f"  {tensor.name} {describe(tensor)}")
        if len(groups["EXPERT_TENSORS"]) > 12:
            print(f"  ... {len(groups['EXPERT_TENSORS']) - 12} more")

    print("\n[LIKELY ROUTER TENSORS IN THIS LAYER]")
    for tensor in groups["LAYER_CANDIDATES"]:
        name = tensor.name.lower()
        if any(token in name for token in ("gate_inp", "router", "gate")):
            print(f"  {tensor.name} {describe(tensor)}")


if __name__ == "__main__":
    main()
