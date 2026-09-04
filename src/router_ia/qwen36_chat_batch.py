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
        "ram_items": int(ram["items"]),
        "ram_bytes": int(ram["bytes"]),
        "ram_hit_rate": float(ram["hit_rate"]),
        "ram_evictions": int(ram["evictions"]),
        "vram_items": int(vram["items"]),
        "vram_bytes": int(vram["bytes"]),
        "vram_hit_rate": float(vram["hit_rate"]),
        "vram_evictions": int(vram["evictions"]),
        "vram_expert_share": float(vram["expert_share"]),
        "vram_resident_bytes": int(vram["resident_bytes"]),
        "vram_resident_budget": int(vram["resident_budget_bytes"]),
        "vram_expert_bytes": int(vram["expert_bytes"]),
        "vram_expert_budget": int(vram["expert_budget_bytes"]),
        "vram_expert_hit_rate": float(vram["expert_pool_hit_rate"]),
        "vram_expert_evictions": int(vram["expert_evictions"]),
        "vram_stream_bytes": int(vram["stream_bytes"]),
        "vram_stream_budget": int(vram["stream_budget_bytes"]),
        "vram_stream_hit_rate": float(vram["stream_hit_rate"]),
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


def print_attention(root: Path, label: str) -> None:
    stats = attention_cache.stats(root)
    print(
        f"  attention {label}: "
        f"tokens={stats['tokens_seen']} | "
        f"full_kv_layers={stats['full_layers_cached']} | "
        f"full_kv_tokens={stats['full_tokens']} | "
        f"full_kv={stats['full_bytes'] / 1024**2:.1f} MiB | "
        f"delta_state={stats['linear_bytes'] / 1024**2:.1f} MiB | "
        f"conv_state={stats['linear_conv_bytes'] / 1024**2:.2f} MiB | "
        f"total={stats['bytes'] / 1024**2:.1f} MiB"
    )


def _projection(root: Path, prefix: str, device: str) -> torch.Tensor:
    """Load a dequantized projection through the hierarchical cache."""
    return cached._cached_load_projection(root, prefix, device)


def _expert_projection_triplet(
    root: Path,
    layer_prefix: str,
    expert_id: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    expert_prefix = f"{layer_prefix}mlp.experts.{expert_id}"
    return (
        _projection(root, expert_prefix + ".gate_proj", device),
        _projection(root, expert_prefix + ".up_proj", device),
        _projection(root, expert_prefix + ".down_proj", device),
    )


def _prefetch_one_expert_raw(root: Path, layer_prefix: str, expert_id: int) -> None:
    cache = __import__("router_ia.qwen36_expert_cache", fromlist=["RoutedExpertCache"])
    store = cached._store(root)
    expert_cache = cache._EXPERT_CACHES.get(root.resolve()) if hasattr(cache, "_EXPERT_CACHES") else None
    if expert_cache is None:
        return
    expert_cache.prefetch_expert_raw(store, layer_prefix, expert_id)


def _warm_expert_raw_cache(root: Path, layer_prefix: str, expert_ids: list[int]) -> None:
    """Prefetch routed FP8 weights/scales into the rotating VRAM stream in parallel."""
    if not expert_ids:
        return
    store = cached._store(root)
    from .qwen36_expert_cache import RoutedExpertCache
    from .qwen36_chat_batch_fused import _EXPERT_CACHES  # type: ignore

    expert_cache = _EXPERT_CACHES.get(root.resolve())
    if expert_cache is None or not isinstance(expert_cache, RoutedExpertCache):
        return

    workers = min(EXPERT_LOAD_WORKERS, len(expert_ids))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(expert_cache.prefetch_expert_raw, store, layer_prefix, expert_id)
            for expert_id in expert_ids
        ]
        for future in futures:
            future.result()


def batched_moe_step(
    root: Path,
    layer: int,
    residual: torch.Tensor,
    top_k: int,
    device: str,
) -> tuple[torch.Tensor, list[int], list[float], float, float]:
    prefix = base.layer_prefix(layer)
    post_norm = base.load_layer_weight(root, layer, "post_attention_layernorm.weight", device)
    moe_in = base.rmsnorm(residual, post_norm).reshape(1, base.HIDDEN).float()
    router_w = base.load_layer_weight(root, layer, "mlp.gate.weight", device).float()
    routed = base.route(moe_in.reshape(-1), router_w, top_k=top_k)
    expert_ids = [int(v) for v in routed.expert_ids.detach().cpu().tolist()]
    weights = [float(v) for v in routed.weights.detach().cpu().tolist()]

    if device == "cuda":
        _warm_expert_raw_cache(root, prefix, expert_ids)

    with ThreadPoolExecutor(max_workers=min(EXPERT_LOAD_WORKERS, len(expert_ids))) as pool:
        futures = [
            pool.submit(_expert_projection_triplet, root, prefix, expert_id, device)
            for expert_id in expert_ids
        ]
        triplets = [future.result() for future in futures]

    gate_w = torch.stack([triplet[0] for triplet in triplets], dim=0)
    up_w = torch.stack([triplet[1] for triplet in triplets], dim=0)
    down_w = torch.stack([triplet[2] for triplet in triplets], dim=0)
    batch_x = moe_in.expand(len(expert_ids), -1)

    if device == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            batch_x_compute = batch_x.to(dtype=torch.float16)
            gate = torch.bmm(gate_w, batch_x_compute.unsqueeze(-1)).squeeze(-1)
            up = torch.bmm(up_w, batch_x_compute.unsqueeze(-1)).squeeze(-1)
            hidden = F.silu(gate) * up
            expert_out = torch.bmm(down_w, hidden.unsqueeze(-1)).squeeze(-1)
            routing = torch.tensor(weights, device=device, dtype=expert_out.dtype).reshape(-1, 1)
            routed_sum = (expert_out * routing).sum(dim=0, keepdim=True)
    else:
        gate = torch.bmm(gate_w, batch_x.unsqueeze(-1)).squeeze(-1)
        up = torch.bmm(up_w, batch_x.unsqueeze(-1)).squeeze(-1)
        hidden = F.silu(gate) * up
        expert_out = torch.bmm(down_w, hidden.unsqueeze(-1)).squeeze(-1)
        routing = torch.tensor(weights, device=device, dtype=expert_out.dtype).reshape(-1, 1)
        routed_sum = (expert_out * routing).sum(dim=0, keepdim=True)
        batch_x_compute = batch_x

    shared_gate_w = base.load_layer_weight(root, layer, "mlp.shared_expert_gate.weight", device).float()
    shared_gate_proj = _projection(root, f"{prefix}mlp.shared_expert.gate_proj", device)
    shared_up_proj = _projection(root, f"{prefix}mlp.shared_expert.up_proj", device)
    shared_down_proj = _projection(root, f"{prefix}mlp.shared_expert.down_proj", device)

    if device == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            shared_gate = torch.sigmoid(F.linear(moe_in, shared_gate_w))
            shared_hidden = F.silu(F.linear(moe_in.to(shared_gate_proj.dtype), shared_gate_proj)) * F.linear(moe_in.to(shared_up_proj.dtype), shared_up_proj)
            shared_out = F.linear(shared_hidden, shared_down_proj) * shared_gate
    else:
        shared_gate = torch.sigmoid(F.linear(moe_in, shared_gate_w))
        shared_hidden = F.silu(F.linear(moe_in, shared_gate_proj)) * F.linear(moe_in, shared_up_proj)
        shared_out = F.linear(shared_hidden, shared_down_proj) * shared_gate

    moe_out = routed_sum.float() + shared_out.float()
    layer_out = residual + moe_out
    shared_gate_value = float(shared_gate.float().item())
    moe_input_norm = float(torch.linalg.vector_norm(moe_in).item())

    del post_norm, moe_in, router_w, routed
    del triplets, gate_w, up_w, down_w, batch_x, batch_x_compute
    del gate, up, hidden, expert_out, routing, routed_sum
    del shared_gate_w, shared_gate, shared_gate_proj, shared_up_proj, shared_down_proj
    del shared_hidden, shared_out, moe_out
    return layer_out, expert_ids, weights, shared_gate_value, moe_input_norm


def run_forward_token(
    root: Path,
    token_id: int,
    final_norm: torch.Tensor,
    lm_head: torch.Tensor,
    final_norm_name: str,
    lm_head_name: str,
    device: str,
    advance_state: bool = True,
) -> tuple[torch.Tensor, float, float]:
    """Run one token through the full 40-layer stack using persistent attention state."""
    start = perf_counter()
    x = base.load_embedding_row(root, token_id).reshape(1, base.HIDDEN).to(device).float()
    for layer in range(base.DEFAULT_LAYERS):
        residual = attention_cache.step_attention(root, layer, x, device)
        x, *_ = batched_moe_step(root, layer, residual, top_k=8, device=device)
        del residual

    if device == "cuda":
        final_norm_runtime = cached.cached_runtime_tensor(root, final_norm_name, device, dtype=torch.float32)
        lm_head_runtime = cached.cached_runtime_tensor(root, lm_head_name, device, dtype=torch.float16)
    else:
        final_norm_runtime = final_norm
        lm_head_runtime = lm_head

    x = base.rmsnorm(x, final_norm_runtime)
    if device == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = F.linear(x, lm_head_runtime)
    else:
        logits = F.linear(x, lm_head_runtime)

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = perf_counter() - start
    peak_logit = float(torch.max(logits.float()).item())

    if advance_state:
        state = attention_cache.active(root, device)
        state.tokens_seen += 1

    del x
    if device == "cuda":
        del final_norm_runtime, lm_head_runtime
    gc.collect()
    return logits, elapsed, peak_logit


def run_generated_token(
    root: Path,
    token_id: int,
    final_norm: torch.Tensor,
    lm_head: torch.Tensor,
    final_norm_name: str,
    lm_head_name: str,
    device: str,
    sampling_top_k: int,
    temperature: float,
) -> tuple[int, float, float]:
    """Compatibility shim for runtime_optimizations' legacy hook.

    State advancement is deliberately left to the legacy optimization wrapper,
    which increments ``tokens_seen`` after this hook returns.
    """
    logits, elapsed, peak_logit = run_forward_token(
        root,
        token_id,
        final_norm,
        lm_head,
        final_norm_name,
        lm_head_name,
        device,
        advance_state=False,
    )
    next_id = sample_next(logits, temperature, sampling_top_k)
    del logits
    gc.collect()
    return next_id, elapsed, peak_logit


def generate_response(
    root: Path,
    prompt: str,
    tokenizer,
    final_norm: torch.Tensor,
    lm_head: torch.Tensor,
    final_norm_name: str,
    lm_head_name: str,
    device: str,
    max_new_tokens: int,
    sampling_top_k: int,
    temperature: float,
) -> None:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        print("IA> [nenhum token produzido pelo tokenizer]")
        return

    state = attention_cache.state_for(root, device)
    state.reset()
    attention_cache.activate(root, state)

    eos_id = getattr(tokenizer, "eos_token_id", None)
    generated: list[int] = []
    print(f"\nVocê> {prompt}")
    print(f"  prompt tokens={len(prompt_ids)} | prefill=stateful")
    print("IA> ", end="", flush=True)
    turn_start = perf_counter()

    try:
        prompt_logits: torch.Tensor | None = None
        prompt_elapsed = 0.0
        prefill_start = perf_counter()
        for prompt_id in prompt_ids:
            if prompt_logits is not None:
                del prompt_logits
            prompt_logits, prompt_elapsed, _ = run_forward_token(
                root, int(prompt_id), final_norm, lm_head, final_norm_name, lm_head_name, device
            )
        prefill_elapsed = perf_counter() - prefill_start

        next_id = sample_next(prompt_logits, temperature, sampling_top_k)
        del prompt_logits
        generated.append(next_id)
        first_text = tokenizer.decode([next_id], skip_special_tokens=True)
        print(first_text, end="", flush=True)
        print(
            f"\n  [step 01] token={next_id} | source=prefill | prefill_time={prefill_elapsed:.3f}s | "
            f"attn_tokens={attention_cache.stats(root)['tokens_seen']} | "
            f"kv_tokens={attention_cache.stats(root)['full_tokens']}",
            flush=True,
        )

        if eos_id is None or next_id != int(eos_id):
            for step in range(2, max_new_tokens + 1):
                before = cache_stats(root)
                logits, elapsed, peak = run_forward_token(
                    root, next_id, final_norm, lm_head, final_norm_name, lm_head_name, device
                )
                after = cache_stats(root)
                next_id = sample_next(logits, temperature, sampling_top_k)
                delta_hits = int(after.get("hits", 0)) - int(before.get("hits", 0))
                delta_misses = int(after.get("misses", 0)) - int(before.get("misses", 0))
                step_hit_rate = delta_hits / max(delta_hits + delta_misses, 1) * 100.0
                generated.append(next_id)
                current_id = next_id
                text = tokenizer.decode([next_id], skip_special_tokens=True)
                print(text, end="", flush=True)
                attn = attention_cache.stats(root)
                print(
                    f"\n  [step {step:02d}] token={next_id} | time={elapsed:.3f}s | "
                    f"attn_tokens={attn['tokens_seen']} | kv_tokens={attn['full_tokens']} | "
                    f"attn_mem={attn['bytes'] / 1024**2:.1f}MiB | step_hit_rate={step_hit_rate:.1f}% | "
                    f"global_hit_rate={after.get('hit_rate', 0.0):.2f}% | "
                    f"ram_hit={after.get('ram_hit_rate', 0.0):.2f}% | vram_hit={after.get('vram_hit_rate', 0.0):.2f}% | "
                    f"expert_vram_hit={after.get('vram_expert_hit_rate', 0.0):.2f}% | "
                    f"stream_hit={after.get('vram_stream_hit_rate', 0.0):.2f}% | "
                    f"hits+{delta_hits} misses+{delta_misses} | peak_logit={peak:.4f}",
                    flush=True,
                )
                del logits
                if eos_id is not None and next_id == int(eos_id):
                    break

        print()
        print(f"  resposta: {len(generated)} tokens | wall={perf_counter() - turn_start:.3f}s")
        print_attention(root, "after turn")
        print_cache(root, "after turn")
    finally:
        attention_cache.deactivate(root)
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stateful Qwen3.6 router mini-chat test")
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--device", default="cpu")
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
    print("prompts=4")
    print(f"device={device}")
    print(f"max_new_tokens={args.max_new_tokens}")
    print(f"sampling_top_k={args.sampling_top_k}")
    print(f"temperature={args.temperature}")
    print(f"lm_head={lm_head_name} shape={tuple(lm_head.shape)}")
    print(f"final_norm={final_norm_name} shape={tuple(final_norm.shape)}")
    print_cache(root, "initial")
    print_attention(root, "initial")
    prompts = ["Olá", "Como você está?", "Explique o que é uma CPU", "Quanto é 2 + 2?"]
    for prompt in prompts:
        generate_response(root, prompt, tokenizer, final_norm, lm_head, final_norm_name, lm_head_name, device, args.max_new_tokens, args.sampling_top_k, args.temperature)
    print("\n===== SUMMARY =====")
    print("turns=4")
    print_cache(root, "final")
    print_attention(root, "final")


if __name__ == "__main__":
    main()
