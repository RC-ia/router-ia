from __future__ import annotations

"""Stateful Qwen3.6 chat generator using hierarchical RAM/VRAM caches."""

import argparse
import gc
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from . import qwen36_attention_cache as attention_cache
from . import qwen36_cached_loop as cached
from . import qwen36_40layer_loop as base
from .qwen36_mini_chat import load_final_norm, load_lm_head, load_tokenizer, sample_next

DEFAULT_MAX_NEW_TOKENS = 4
EXPERT_LOAD_WORKERS = max(1, int(os.getenv("QWEN36_EXPERT_LOAD_WORKERS", "8")))


def cache_stats(root: Path) -> dict[str, int | float]:
    store = cached._stores.get(root.resolve())
    if store is None:
        return {}
    ram = store.ram_cache.snapshot()
    vram = store.vram_cache.snapshot()
    hits = int(ram["hits"] + vram["hits"])
    misses = int(ram["misses"] + vram["misses"])
    total = hits + misses
    return {
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / total * 100.0 if total else 0.0,
        "ram_bytes": int(ram["bytes"]),
        "ram_hit_rate": float(ram["hit_rate"]),
        "vram_bytes": int(vram["bytes"]),
        "vram_hit_rate": float(vram["hit_rate"]),
        "vram_resident_bytes": int(vram["resident_bytes"]),
        "vram_resident_budget": cached.RESIDENT_VRAM_BUDGET_BYTES,
        "vram_stream_bytes": int(vram["stream_bytes"]),
        "vram_stream_budget": cached.STREAM_BUDGET_BYTES,
        "vram_stream_hit_rate": float(vram["stream_hit_rate"]),
        "vram_expert_bytes": int(vram["expert_bytes"]),
        "vram_expert_budget": cached.EXPERT_VRAM_BUDGET_BYTES,
        "vram_expert_hit_rate": float(vram["expert_pool_hit_rate"]),
        "vram_expert_evictions": int(vram["expert_evictions"]),
    }


def print_cache(root: Path, label: str) -> None:
    stats = cache_stats(root)
    if not stats:
        print(f"  cache {label}: unavailable")
        return
    print(
        f"  cache {label}: "
        f"ram={stats['ram_bytes'] / 1024**2:.1f}/{cached.CACHE_BUDGET_BYTES / 1024**2:.1f} MiB | "
        f"vram={stats['vram_bytes'] / 1024**2:.1f}/{cached.VRAM_CACHE_BUDGET_BYTES / 1024**2:.1f} MiB | "
        f"resident={stats['vram_resident_bytes'] / 1024**2:.1f}/{stats['vram_resident_budget'] / 1024**2:.1f} MiB | "
        f"experts={stats['vram_expert_bytes'] / 1024**2:.1f}/{stats['vram_expert_budget'] / 1024**2:.1f} MiB | "
        f"stream={stats['vram_stream_bytes'] / 1024**2:.1f}/{stats['vram_stream_budget'] / 1024**2:.1f} MiB | "
        f"hit_rate={stats['hit_rate']:.2f}% | "
        f"ram_hit={stats['ram_hit_rate']:.2f}% | "
        f"vram_hit={stats['vram_hit_rate']:.2f}% | "
        f"expert_vram_hit={stats['vram_expert_hit_rate']:.2f}% | "
        f"stream_hit={stats['vram_stream_hit_rate']:.2f}% | "
        f"expert_evictions={stats['vram_expert_evictions']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stateful Qwen3.6 router mini-chat test")
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prompt", type=str, default=None, help="Run a single custom prompt instead of the default 4-prompt benchmark")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--sampling-top-k", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    root = args.model_dir.resolve()
    device = args.device.lower()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    cached._configure_vram_limit(device)
    tokenizer = load_tokenizer(root)
    final_norm_name, final_norm = load_final_norm(root)
    lm_head_name, lm_head, _lm_head_mode = load_lm_head(root)
    print("op=batch-chat")
    print("mode=stateful-autoregressive-prefill")
    print("attention_state=persistent-deltanet-recurrence|full-attention-kv|linear-conv")
    print("cache=hierarchical-vram-ram-ssd")
    print("vram_policy=resident-60pct-hot-experts-20pct-stream-20pct")
    print("vram_dequantized_cache=fp16-temporary")
    print("cuda_compute=fp16-autocast")
    print(f"expert_load_workers={EXPERT_LOAD_WORKERS}")
    print("expert_prefetch=stream-parallel")
    print("expert_compressed_tiers=FP8-resident|Q4-cold")
    print("expert_eviction=FP8-to-Q4-then-drop")
    print("expert_fp8_promotion=disabled")
    print("expert_fp16_persistent_cache=disabled")
    print("expert_kernel_fused_dequant=not-yet")
    print(f"prompts={'1(custom)' if args.prompt is not None else '4'}")
    print(f"device={device}")
    print(f"max_new_tokens={args.max_new_tokens}")
    print(f"sampling_top_k={args.sampling_top_k}")
    print(f"temperature={args.temperature}")
    print(f"lm_head={lm_head_name} shape={tuple(lm_head.shape)}")
    print(f"final_norm={final_norm_name} shape={tuple(final_norm.shape)}")
    print_cache(root, "initial")
    print_attention(root, "initial")
    prompts = [args.prompt] if args.prompt is not None else ["Olá", "Como você está?", "Explique o que é uma CPU", "Quanto é 2 + 2?"]
    for prompt in prompts:
        generate_response(root, prompt, tokenizer, final_norm, lm_head, final_norm_name, lm_head_name, device, args.max_new_tokens, args.sampling_top_k, args.temperature)
    print("\n===== SUMMARY =====")
    print(f"turns={len(prompts)}")
    print_cache(root, "final")
    print_attention(root, "final")


if __name__ == "__main__":
    main()
