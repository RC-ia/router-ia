from __future__ import annotations

"""Run the first real Qwen3.6 Safetensors MoE path.

This is intentionally a partial forward-path probe, not the complete
transformer layer. It executes:

    token id -> real embedding row -> Qwen3.6 router -> top-k experts
              -> FP8 expert MLPs -> weighted aggregation

Attention/DeltaNet, normalization, shared expert gating, residuals and the
LM head are deliberately left for later stages.
"""

import argparse
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from safetensors import safe_open

from .fp8_expert_cache import FP8ExpertCache
from .fp8_expert_runner import _dequantize_blockwise
from .qwen36_router import DEFAULT_EXPERTS, DEFAULT_HIDDEN_SIZE, DEFAULT_TOP_K, route_from_model


EMBEDDING_NAME = "model.language_model.embed_tokens.weight"
EXPERT_HIDDEN_SIZE = 512


def discover_embedding_shard(root: Path) -> Path:
    """Find the shard containing the Qwen3.6 embedding tensor."""
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        import json

        payload = json.loads(index_path.read_text(encoding="utf-8"))
        shard_name = payload.get("weight_map", {}).get(EMBEDDING_NAME)
        if shard_name:
            shard = root / shard_name
            if shard.is_file():
                return shard

    for shard in sorted(root.glob("*.safetensors")):
        # Read only keys: no tensor payload is loaded here.
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            if EMBEDDING_NAME in handle.keys():
                return shard
    raise KeyError(f"Embedding tensor not found: {EMBEDDING_NAME}")


def load_embedding(root: str | Path, token_id: int, device: str) -> torch.Tensor:
    """Load one real embedding row and return it as float32."""
    root = Path(root).resolve()
    shard = discover_embedding_shard(root)
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        embedding = handle.get_tensor(EMBEDDING_NAME)

        if embedding.ndim != 2 or embedding.shape[1] != DEFAULT_HIDDEN_SIZE:
            raise ValueError(
                f"Unexpected embedding shape: {tuple(embedding.shape)}; "
                f"expected [vocab, {DEFAULT_HIDDEN_SIZE}]"
            )
        if not 0 <= token_id < embedding.shape[0]:
            raise ValueError(
                f"token_id {token_id} is outside vocabulary range 0..{embedding.shape[0] - 1}"
            )

        return embedding[token_id].to(dtype=torch.float32, device=device)


def run_expert(
    cache: FP8ExpertCache,
    layer: int,
    expert: int,
    x: torch.Tensor,
) -> torch.Tensor:
    """Run one FP8 expert using the existing cache and dequantization path."""
    blob = cache.get(layer, expert, tier="vram")
    gate = _dequantize_blockwise(
        blob.weights["gate_proj"], blob.scales["gate_proj"]
    )
    up = _dequantize_blockwise(
        blob.weights["up_proj"], blob.scales["up_proj"]
    )
    down = _dequantize_blockwise(
        blob.weights["down_proj"], blob.scales["down_proj"]
    )

    gate = gate.to(device=x.device)
    up = up.to(device=x.device)
    down = down.to(device=x.device)

    gate_out = F.linear(x, gate)
    up_out = F.linear(x, up)
    hidden = F.silu(gate_out) * up_out
    return F.linear(hidden, down)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the first real Qwen3.6 MoE path")
    parser.add_argument("root", type=Path, help="Qwen3.6 Safetensors directory")
    parser.add_argument("--token-id", type=int, default=0)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--ram-gb", type=float, default=6.0)
    parser.add_argument("--vram-gb", type=float, default=3.0)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this Python environment.")

    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Safetensors directory not found: {root}")
    if args.top_k > DEFAULT_EXPERTS:
        raise SystemExit(f"top-k cannot exceed {DEFAULT_EXPERTS}")

    print(f"Model: {root}")
    print(f"Layer: {args.layer}")
    print(f"Token ID: {args.token_id}")

    start = perf_counter()
    x = load_embedding(root, args.token_id, args.device)
    embedding_ms = (perf_counter() - start) * 1000.0

    print(f"Embedding shape: {tuple(x.shape)}")
    print(f"Embedding norm: {torch.linalg.vector_norm(x).item():.6f}")
    print(f"Embedding load: {embedding_ms:.3f} ms")

    start = perf_counter()
    route_result = route_from_model(
        root,
        args.layer,
        x,
        top_k=args.top_k,
        device=args.device,
    )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    router_ms = (perf_counter() - start) * 1000.0

    ids = route_result.expert_ids.detach().cpu().tolist()
    weights = route_result.weights.detach().cpu().tolist()
    logits = route_result.logits.detach().cpu().tolist()

    print("\nROUTER")
    print("  Expert IDs:", ids)
    print("  Weights:", [round(float(v), 8) for v in weights])
    print("  Logits:", [round(float(v), 8) for v in logits])
    print(f"  Weight sum: {route_result.weights.sum().item():.8f}")
    print(f"  Router: {router_ms:.3f} ms")

    cache = FP8ExpertCache(
        root,
        ram_limit_bytes=int(args.ram_gb * 1024**3),
        vram_limit_bytes=int(args.vram_gb * 1024**3),
        device=args.device,
    )

    aggregate = torch.zeros(DEFAULT_HIDDEN_SIZE, dtype=torch.float32, device=args.device)

    print("\nEXPERT EXECUTION")
    total_expert_ms = 0.0
    for expert, weight in zip(ids, weights):
        start = perf_counter()
        output = run_expert(cache, args.layer, int(expert), x)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed_ms = (perf_counter() - start) * 1000.0
        total_expert_ms += elapsed_ms
        aggregate.add_(output, alpha=float(weight))
        print(
            f"  expert={expert:3d} weight={float(weight):.8f} "
            f"shape={tuple(output.shape)} norm={torch.linalg.vector_norm(output).item():.6f} "
            f"time={elapsed_ms:.3f} ms"
        )

    print("\nAGGREGATED")
    print(f"  Shape: {tuple(aggregate.shape)}")
    print(f"  Norm: {torch.linalg.vector_norm(aggregate).item():.6f}")
    print(f"  Mean: {aggregate.mean().item():.6f}")
    print(f"  Std: {aggregate.std().item():.6f}")
    print(f"  Expert compute total: {total_expert_ms:.3f} ms")
    print("  Cache:", cache.snapshot())
    print("\nNOTE: this is the real embedding -> router -> 8 experts path, not the full transformer layer yet.")


if __name__ == "__main__":
    main()
