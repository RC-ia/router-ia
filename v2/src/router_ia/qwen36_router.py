from __future__ import annotations

"""Qwen3.6 routed-expert selector for Safetensors checkpoints.

Loads only the requested layer's ``mlp.gate.weight`` and performs the
router projection for one or more hidden states. This module deliberately
stops at routing: expert execution remains in the existing expert runner.
"""

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open


ROUTER_SUFFIX = ".mlp.gate.weight"
DEFAULT_HIDDEN_SIZE = 2048
DEFAULT_EXPERTS = 256
DEFAULT_TOP_K = 8


@dataclass(frozen=True)
class RouteResult:
    expert_ids: torch.Tensor
    weights: torch.Tensor
    logits: torch.Tensor

    def as_python(self) -> dict[str, object]:
        return {
            "expert_ids": self.expert_ids.detach().cpu().tolist(),
            "weights": self.weights.detach().cpu().tolist(),
            "logits": self.logits.detach().cpu().tolist(),
        }


def _read_header(path: Path) -> dict[str, object]:
    with path.open("rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError(f"Invalid Safetensors file: {path}")
        size = struct.unpack("<Q", raw)[0]
        payload = fh.read(size)
        if len(payload) != size:
            raise ValueError(f"Truncated Safetensors header: {path}")
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid Safetensors header: {path}")
    return value


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
        raise FileNotFoundError(f"No Safetensors shards found in {root}")
    return shards


def find_router_tensor(root: Path, layer: int) -> tuple[Path, str, tuple[int, ...], str]:
    name = f"model.language_model.layers.{int(layer)}.mlp.gate.weight"
    for shard in discover_shards(root):
        header = _read_header(shard)
        meta = header.get(name)
        if isinstance(meta, dict):
            shape = tuple(int(x) for x in meta.get("shape", ()))
            dtype = str(meta.get("dtype", "?"))
            return shard, name, shape, dtype
    raise KeyError(f"Router tensor not found: {name}")


def load_router(root: str | Path, layer: int, device: str = "cpu") -> torch.Tensor:
    """Load one router matrix as float32 on ``device``."""
    root_path = Path(root).resolve()
    shard, name, shape, _dtype = find_router_tensor(root_path, layer)
    if shape != (DEFAULT_EXPERTS, DEFAULT_HIDDEN_SIZE):
        raise ValueError(
            f"Unexpected router shape for layer {layer}: {shape}; "
            f"expected {(DEFAULT_EXPERTS, DEFAULT_HIDDEN_SIZE)}"
        )
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(name)
    return weight.to(dtype=torch.float32, device=device)


def route(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> RouteResult:
    """Select top-k routed experts for hidden states.

    ``hidden`` may be ``[hidden]`` or ``[tokens, hidden]``.
    Router logits are computed as ``hidden @ gate_weight.T``.
    Top-k scores are softmax-normalized across the selected experts.
    """
    if hidden.ndim not in (1, 2):
        raise ValueError(f"hidden must be 1-D or 2-D, got shape={tuple(hidden.shape)}")
    if gate_weight.ndim != 2:
        raise ValueError(f"gate_weight must be 2-D, got shape={tuple(gate_weight.shape)}")
    if gate_weight.shape[1] != hidden.shape[-1]:
        raise ValueError(
            f"hidden size {hidden.shape[-1]} does not match router input {gate_weight.shape[1]}"
        )
    experts = gate_weight.shape[0]
    if not 1 <= top_k <= experts:
        raise ValueError(f"top_k must be in [1, {experts}], got {top_k}")

    original_dtype = hidden.dtype
    x = hidden.to(dtype=gate_weight.dtype)
    logits = torch.matmul(x, gate_weight.transpose(0, 1))
    values, ids = torch.topk(logits, k=top_k, dim=-1)
    weights = torch.softmax(values, dim=-1)

    # Return float32 weights for stable downstream aggregation.
    return RouteResult(
        expert_ids=ids,
        weights=weights.to(dtype=torch.float32),
        logits=values.to(dtype=torch.float32),
    )


def route_from_model(
    root: str | Path,
    layer: int,
    hidden: torch.Tensor,
    *,
    top_k: int = DEFAULT_TOP_K,
    device: str = "cpu",
) -> RouteResult:
    gate = load_router(root, layer, device=device)
    return route(hidden.to(device), gate, top_k=top_k)


def _deterministic_hidden(seed: int, hidden_size: int, device: str) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return torch.randn(hidden_size, generator=gen, dtype=torch.float32).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Qwen3.6 Safetensors router")
    parser.add_argument("root", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this Python environment.")

    hidden = _deterministic_hidden(args.seed, DEFAULT_HIDDEN_SIZE, args.device)
    result = route_from_model(
        args.root,
        args.layer,
        hidden,
        top_k=args.top_k,
        device=args.device,
    )

    print(f"Layer: {args.layer}")
    print(f"Top-k: {args.top_k}")
    print("Expert IDs:", result.expert_ids.detach().cpu().tolist())
    print("Weights:", [round(float(x), 8) for x in result.weights.detach().cpu().tolist()])
    print("Selected logits:", [round(float(x), 8) for x in result.logits.detach().cpu().tolist()])
    print(f"Weight sum: {result.weights.sum().item():.8f}")


if __name__ == "__main__":
    main()
