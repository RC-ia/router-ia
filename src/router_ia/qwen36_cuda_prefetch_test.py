from __future__ import annotations

"""Experimental Qwen3.6 runner with a GPU-resident layer prefetch cache.

The cache keeps original FP8/BF16 layer tensors on CUDA. The existing CUDA
executor is reused, while tensor loading is intercepted for cached tensors.
Only tensors actually needed by an operation are dequantized during execution.
"""

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import torch
from safetensors import safe_open

from . import qwen36_cuda_loop as runner
from . import qwen36_op_probe as ops

DEFAULT_BUDGET_GIB = 2.5
DEFAULT_LAYERS = 40


def tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def load_index(root: Path) -> dict[str, str]:
    path = root / "model.safetensors.index.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("weight_map", {})


def names_for_layer(root: Path, layer: int, weight_map: dict[str, str]) -> list[str]:
    prefix = f"model.language_model.layers.{layer}."
    return sorted(name for name in weight_map if name.startswith(prefix))


def load_raw_tensor(root: Path, name: str, weight_map: dict[str, str]) -> torch.Tensor:
    shard_name = weight_map.get(name)
    if not shard_name:
        raise KeyError(f"Tensor not indexed: {name}")
    shard = root / shard_name
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        if name not in handle.keys():
            raise KeyError(f"Tensor not found: {name}")
        return handle.get_tensor(name)


class GPUCache:
    def __init__(self, root: Path, budget_bytes: int, weight_map: dict[str, str]) -> None:
        self.root = root
        self.budget_bytes = budget_bytes
        self.weight_map = weight_map
        self.used_bytes = 0
        self.tensors: dict[str, torch.Tensor] = {}
        self.layers: list[int] = []

    def preload_layer(self, layer: int) -> bool:
        names = names_for_layer(self.root, layer, self.weight_map)
        layer_bytes = 0
        cpu_tensors: list[tuple[str, torch.Tensor]] = []
        for name in names:
            tensor = load_raw_tensor(self.root, name, self.weight_map)
            layer_bytes += tensor_nbytes(tensor)
            cpu_tensors.append((name, tensor))

        if self.used_bytes + layer_bytes > self.budget_bytes:
            print(
                f"prefetch stop: layer {layer} = {layer_bytes / 1024**2:.1f} MiB; "
                f"remaining = {(self.budget_bytes - self.used_bytes) / 1024**2:.1f} MiB"
            )
            del cpu_tensors
            gc.collect()
            return False

        for name, tensor in cpu_tensors:
            self.tensors[name] = tensor.to("cuda")
            del tensor

        self.used_bytes += layer_bytes
        self.layers.append(layer)
        print(
            f"prefetched layer {layer}: {layer_bytes / 1024**2:.1f} MiB "
            f"(cache {self.used_bytes / 1024**2:.1f} / {self.budget_bytes / 1024**2:.1f} MiB)"
        )
        del cpu_tensors
        gc.collect()
        return True

    def load_tensor(self, root: Path, name: str, device: str = "cpu") -> torch.Tensor:
        cached = self.tensors.get(name)
        if cached is not None:
            return cached
        return ops.load_tensor(root, name, device=device)


def install_cache(cache: GPUCache) -> None:
    def cached_load_tensor(root: Path, name: str, device: str = "cpu") -> torch.Tensor:
        return cache.load_tensor(root, name, device)

    # Functions in qwen36_cuda_loop resolve these globals at runtime.
    runner.load_tensor = cached_load_tensor
    runner.load_layer_weight = lambda root, layer, suffix, device: cached_load_tensor(
        root, runner.layer_prefix(layer) + suffix, device
    )

    # Functions imported from qwen36_op_probe, including load_projection and
    # load_embedding_row, resolve qwen36_op_probe.load_tensor at runtime.
    ops.load_tensor = cached_load_tensor


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.6 CUDA loop with GPU-prefetched layers")
    parser.add_argument("root", type=Path)
    parser.add_argument("--token-id", type=int, default=0)
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--end-layer", type=int, default=DEFAULT_LAYERS - 1)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--budget-gib", type=float, default=DEFAULT_BUDGET_GIB)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    if not 0 <= args.start_layer <= args.end_layer < DEFAULT_LAYERS:
        raise SystemExit(f"layer range must be inside 0..{DEFAULT_LAYERS - 1}")
    if args.budget_gib <= 0:
        raise SystemExit("budget-gib must be positive")

    root = args.root.resolve()
    budget_bytes = int(args.budget_gib * 1024**3)
    weight_map = load_index(root)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch CUDA: {torch.version.cuda}")
    print(f"prefetch budget: {args.budget_gib:.2f} GiB")

    cache = GPUCache(root, budget_bytes, weight_map)
    prefetch_start = perf_counter()
    for layer in range(args.start_layer, args.end_layer + 1):
        if not cache.preload_layer(layer):
            break
    torch.cuda.synchronize()
    prefetch_ms = (perf_counter() - prefetch_start) * 1000.0

    install_cache(cache)
    print(f"prefetch layers: {cache.layers}")
    print(f"prefetch time: {prefetch_ms:.3f} ms")
    print(f"VRAM allocated after prefetch: {torch.cuda.memory_allocated() / 1024**2:.1f} MiB")
    print(f"VRAM reserved after prefetch: {torch.cuda.memory_reserved() / 1024**2:.1f} MiB")

    x = runner.load_embedding_row(root, args.token_id).reshape(1, runner.HIDDEN).to("cuda").float()
    start_total = perf_counter()

    if not args.quiet:
        print("op=cuda_prefetch_loop")
        print(f"token id: {args.token_id}")
        print(f"layers: {args.start_layer}..{args.end_layer}")
        print(f"cached layers: {cache.layers}")
        print(f"input norm: {torch.linalg.vector_norm(x).item():.8f}")

    for layer in range(args.start_layer, args.end_layer + 1):
        start_layer = perf_counter()
        x_before = x
        kind = runner.attention_type(root, layer)
        if kind == "linear_attention":
            residual = runner.linear_attention_step(root, layer, x_before, "cuda")
        else:
            residual = runner.full_attention_step(root, layer, x_before, "cuda")
        x, expert_ids, weights, shared_gate, moe_input_norm = runner.moe_step(
            root, layer, residual, args.top_k, "cuda"
        )
        torch.cuda.synchronize()
        layer_ms = (perf_counter() - start_layer) * 1000.0

        if not args.quiet:
            print(f"layer {layer} ({kind}):")
            print(f"  router top-{args.top_k}: {expert_ids}")
            print(f"  router weights: {[round(v, 8) for v in weights]}")
            print(f"  shared gate: {shared_gate:.8f}")
            print(f"  moe input norm: {moe_input_norm:.8f}")
            print(f"  output shape: {tuple(x.shape)}")
            print(f"  output norm: {torch.linalg.vector_norm(x).item():.8f}")
            print(f"  output mean: {x.mean().item():.8f}")
            print(f"  VRAM allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MiB")
            print(f"  VRAM reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MiB")
            print(f"  time: {layer_ms:.3f} ms")
        del x_before, residual

    torch.cuda.synchronize()
    total_ms = (perf_counter() - start_total) * 1000.0
    print(f"final output shape: {tuple(x.shape)}")
    print(f"final output norm: {torch.linalg.vector_norm(x).item():.8f}")
    print(f"final output mean: {x.mean().item():.8f}")
    print(f"final output min: {x.min().item():.8f}")
    print(f"final output max: {x.max().item():.8f}")
    print(f"total time: {total_ms:.3f} ms")

    del x, cache
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
