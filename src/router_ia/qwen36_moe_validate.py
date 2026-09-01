from __future__ import annotations

"""Validate the Qwen3.6 partial MoE path against direct checkpoint math.

This validator compares two paths for the same real embedding row:

1. Direct checkpoint path: load router/expert tensors straight from Safetensors
   and compute the FP8 expert MLPs.
2. Runtime path: use qwen36_router + FP8ExpertCache and run the selected experts.

The comparison validates routing, expert selection, FP8 dequantization, expert
math, weighted aggregation, and cache residency. It is not a full reference
implementation of the entire Qwen3.6 transformer block.
"""

import argparse
import json
import struct
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from safetensors import safe_open

from .fp8_expert_cache import FP8ExpertCache
from .fp8_expert_runner import _dequantize_blockwise
from .qwen36_router import DEFAULT_EXPERTS, DEFAULT_HIDDEN_SIZE, DEFAULT_TOP_K, route

EMBEDDING_NAME = "model.language_model.embed_tokens.weight"
ROUTER_TEMPLATE = "model.language_model.layers.{layer}.mlp.gate.weight"
EXPERT_TEMPLATE = "model.language_model.layers.{layer}.mlp.experts.{expert}.{kind}.weight"
SCALE_TEMPLATE = "model.language_model.layers.{layer}.mlp.experts.{expert}.{kind}.weight_scale_inv"
KINDS = ("gate_proj", "up_proj", "down_proj")


def read_index(root: Path) -> dict[str, str]:
    path = root / "model.safetensors.index.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in payload.get("weight_map", {}).items()}


def discover_tensor_shard(root: Path, name: str, weight_map: dict[str, str]) -> Path:
    mapped = weight_map.get(name)
    if mapped:
        path = root / mapped
        if path.is_file():
            return path

    for shard in sorted(root.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            if name in handle.keys():
                return shard
    raise KeyError(f"Tensor not found: {name}")


def load_tensor(root: Path, name: str, weight_map: dict[str, str]) -> torch.Tensor:
    shard = discover_tensor_shard(root, name, weight_map)
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def load_embedding(root: Path, token_id: int, weight_map: dict[str, str], device: str) -> torch.Tensor:
    embedding = load_tensor(root, EMBEDDING_NAME, weight_map)
    if embedding.ndim != 2 or embedding.shape[1] != DEFAULT_HIDDEN_SIZE:
        raise ValueError(f"Unexpected embedding shape: {tuple(embedding.shape)}")
    if not 0 <= token_id < embedding.shape[0]:
        raise ValueError(f"token_id {token_id} outside vocabulary 0..{embedding.shape[0]-1}")
    return embedding[token_id].to(dtype=torch.float32, device=device)


def direct_router(root: Path, layer: int, hidden: torch.Tensor, weight_map: dict[str, str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    name = ROUTER_TEMPLATE.format(layer=layer)
    gate = load_tensor(root, name, weight_map).to(dtype=torch.float32, device=hidden.device)
    if tuple(gate.shape) != (DEFAULT_EXPERTS, DEFAULT_HIDDEN_SIZE):
        raise ValueError(f"Unexpected router shape: {tuple(gate.shape)}")
    logits = torch.matmul(hidden, gate.transpose(0, 1))
    values, ids = torch.topk(logits, k=DEFAULT_TOP_K, dim=-1)
    weights = torch.softmax(values, dim=-1)
    return ids, weights, values


def direct_expert(root: Path, layer: int, expert: int, hidden: torch.Tensor, weight_map: dict[str, str]) -> torch.Tensor:
    matrices: dict[str, torch.Tensor] = {}
    for kind in KINDS:
        weight_name = EXPERT_TEMPLATE.format(layer=layer, expert=expert, kind=kind)
        scale_name = SCALE_TEMPLATE.format(layer=layer, expert=expert, kind=kind)
        weight = load_tensor(root, weight_name, weight_map)
        scale = load_tensor(root, scale_name, weight_map)
        matrix = _dequantize_blockwise(weight, scale).to(hidden.device)
        matrices[kind] = matrix

    gate = matrices["gate_proj"]
    up = matrices["up_proj"]
    down = matrices["down_proj"]
    gate_out = F.linear(hidden, gate)
    up_out = F.linear(hidden, up)
    expert_hidden = F.silu(gate_out) * up_out
    return F.linear(expert_hidden, down)


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.max(torch.abs(a - b)).item())


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = max(float(torch.linalg.vector_norm(b).item()), 1e-12)
    return float(torch.linalg.vector_norm(a - b).item()) / denom


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Qwen3.6 partial MoE runtime")
    parser.add_argument("root", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--token-id", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--ram-gb", type=float, default=6.0)
    parser.add_argument("--vram-gb", type=float, default=3.0)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this Python environment.")

    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Safetensors directory not found: {root}")

    weight_map = read_index(root)
    x = load_embedding(root, args.token_id, weight_map, args.device)

    print(f"Model: {root}")
    print(f"Layer: {args.layer}")
    print(f"Token ID: {args.token_id}")
    print(f"Hidden norm: {torch.linalg.vector_norm(x).item():.8f}")

    # Independent routing path: direct tensor load, no qwen36_router helper.
    start = perf_counter()
    ref_ids, ref_weights, ref_logits = direct_router(root, args.layer, x, weight_map)
    if args.device == "cuda":
        torch.cuda.synchronize()
    router_ms = (perf_counter() - start) * 1000.0

    print("\nREFERENCE ROUTER")
    print("  IDs:", ref_ids.detach().cpu().tolist())
    print("  Weights:", [round(float(v), 8) for v in ref_weights.detach().cpu().reshape(-1)])
    print("  Logits:", [round(float(v), 8) for v in ref_logits.detach().cpu().reshape(-1)])

    # Runtime router using the package implementation.
    runtime_gate = load_tensor(root, ROUTER_TEMPLATE.format(layer=args.layer), weight_map)
    runtime_result = route(x, runtime_gate.to(device=x.device), top_k=DEFAULT_TOP_K)
    ids_match = torch.equal(ref_ids, runtime_result.expert_ids)
    weights_diff = max_abs(ref_weights.float(), runtime_result.weights.float())
    logits_diff = max_abs(ref_logits.float(), runtime_result.logits.float())

    print("\nROUTER COMPARISON")
    print(f"  IDs match: {ids_match}")
    print(f"  max |weight diff|: {weights_diff:.9g}")
    print(f"  max |logit diff|: {logits_diff:.9g}")
    print(f"  router time: {router_ms:.3f} ms")

    cache = FP8ExpertCache(
        root,
        ram_limit_bytes=int(args.ram_gb * 1024**3),
        vram_limit_bytes=int(args.vram_gb * 1024**3),
        device=args.device,
    )

    aggregate_ref = torch.zeros(DEFAULT_HIDDEN_SIZE, dtype=torch.float32, device=x.device)
    aggregate_runtime = torch.zeros_like(aggregate_ref)

    print("\nEXPERT COMPARISON")
    all_expert_ok = True
    for expert, weight in zip(ref_ids.reshape(-1).tolist(), ref_weights.reshape(-1).tolist()):
        start = perf_counter()
        ref_out = direct_expert(root, args.layer, int(expert), x, weight_map)
        if args.device == "cuda":
            torch.cuda.synchronize()
        ref_ms = (perf_counter() - start) * 1000.0

        start = perf_counter()
        blob = cache.get(args.layer, int(expert), tier="vram")
        gate = _dequantize_blockwise(blob.weights["gate_proj"], blob.scales["gate_proj"])
        up = _dequantize_blockwise(blob.weights["up_proj"], blob.scales["up_proj"])
        down = _dequantize_blockwise(blob.weights["down_proj"], blob.scales["down_proj"])
        runtime_out = F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)
        if args.device == "cuda":
            torch.cuda.synchronize()
        runtime_ms = (perf_counter() - start) * 1000.0

        abs_diff = max_abs(runtime_out, ref_out)
        rel_diff = rel_l2(runtime_out, ref_out)
        ok = abs_diff <= args.atol + args.rtol * max(float(torch.max(torch.abs(ref_out)).item()), 1e-12)
        all_expert_ok = all_expert_ok and ok
        aggregate_ref.add_(ref_out, alpha=float(weight))
        aggregate_runtime.add_(runtime_out, alpha=float(weight))

        print(
            f"  expert={int(expert):3d} ok={ok} "
            f"max_abs={abs_diff:.9g} rel_l2={rel_diff:.9g} "
            f"direct={ref_ms:.3f}ms runtime={runtime_ms:.3f}ms"
        )

    aggregate_abs = max_abs(aggregate_runtime, aggregate_ref)
    aggregate_rel = rel_l2(aggregate_runtime, aggregate_ref)

    print("\nAGGREGATE COMPARISON")
    print(f"  Reference norm: {torch.linalg.vector_norm(aggregate_ref).item():.8f}")
    print(f"  Runtime norm:   {torch.linalg.vector_norm(aggregate_runtime).item():.8f}")
    print(f"  max_abs: {aggregate_abs:.9g}")
    print(f"  rel_l2:  {aggregate_rel:.9g}")
    print(f"  all experts within tolerance: {all_expert_ok}")
    print("  First-run cache:", cache.snapshot())

    # A second pass should hit all eight VRAM entries.
    for expert in ref_ids.reshape(-1).tolist():
        cache.get(args.layer, int(expert), tier="vram")
    print("  Second-pass cache:", cache.snapshot())

    overall = ids_match and weights_diff <= args.atol + args.rtol and logits_diff <= args.atol + args.rtol and all_expert_ok and aggregate_abs <= args.atol + args.rtol * max(float(torch.max(torch.abs(aggregate_ref)).item()), 1e-12)
    print(f"\nVALIDATION: {'PASS' if overall else 'FAIL'}")

    if not overall:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
