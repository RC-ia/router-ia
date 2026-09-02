from __future__ import annotations

"""Batch mini-chat smoke test for Qwen3.6 hierarchical RAM/VRAM caches."""

import argparse
import gc
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from . import qwen36_cached_loop as cached
from . import qwen36_40layer_loop as base
from .qwen36_mini_chat import load_final_norm, load_lm_head, load_tokenizer, sample_next

DEFAULT_MAX_NEW_TOKENS = 4


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


def _projection(root: Path, prefix: str, device: str) -> torch.Tensor:
    """Load a dequantized projection through the hierarchical cache."""
    return cached._cached_load_projection(root, prefix, device)


def batched_moe_step(
    root: Path,
    layer: int,
    residual: torch.Tensor,
    top_k: int,
    device: str,
) -> tuple[torch.Tensor, list[int], list[float], float, float]:
    """Run selected routed experts as one batched GPU workload."""
    prefix = base.layer_prefix(layer)
    post_norm = base.load_layer_weight(root, layer, "post_attention_layernorm.weight", device)
    moe_in = base.rmsnorm(residual, post_norm).reshape(1, base.HIDDEN).float()
    router_w = base.load_layer_weight(root, layer, "mlp.gate.weight", device).float()

    routed = base.route(moe_in.reshape(-1), router_w, top_k=top_k)
    expert_ids = [int(v) for v in routed.expert_ids.detach().cpu().tolist()]
    weights = [float(v) for v in routed.weights.detach().cpu().tolist()]

    gate_weights = []
    up_weights = []
    down_weights = []
    for expert_id in expert_ids:
        expert_prefix = f"{prefix}mlp.experts.{expert_id}"
        gate_weights.append(_projection(root, expert_prefix + ".gate_proj", device))
        up_weights.append(_projection(root, expert_prefix + ".up_proj", device))
        down_weights.append(_projection(root, expert_prefix + ".down_proj", device))

    gate_w = torch.stack(gate_weights, dim=0)
    up_w = torch.stack(up_weights, dim=0)
    down_w = torch.stack(down_weights, dim=0)

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
    del gate_weights, up_weights, down_weights
    del gate_w, up_w, down_w, batch_x, batch_x_compute, gate, up, hidden, expert_out, routing, routed_sum
    del shared_gate_w, shared_gate, shared_gate_proj, shared_up_proj, shared_down_proj
    del shared_hidden, shared_out, moe_out
    gc.collect()
    return layer_out, expert_ids, weights, shared_gate_value, moe_input_norm


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
    start = perf_counter()
    x = base.load_embedding_row(root, token_id).reshape(1, base.HIDDEN).to(device).float()

    for layer in range(base.DEFAULT_LAYERS):
        kind = base.attention_type(root, layer)
        if kind == "linear_attention":
            residual = base.linear_attention_step(root, layer, x, device)
        else:
            residual = base.full_attention_step(root, layer, x, device)
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
    next_id = sample_next(logits, temperature, sampling_top_k)

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = perf_counter() - start
    peak_logit = float(torch.max(logits.float()).item())

    del x, logits
    if device == "cuda":
        del final_norm_runtime, lm_head_runtime
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

    current_id = int(prompt_ids[-1])
    eos_id = getattr(tokenizer, "eos_token_id", None)
    generated: list[int] = []

    print(f"\nVocê> {prompt}")
    print(f"  prompt tokens={len(prompt_ids)} | seed_token={current_id}")
    print("IA> ", end="", flush=True)

    turn_start = perf_counter()
    for step in range(1, max_new_tokens + 1):
        before = cache_stats(root)
        next_id, elapsed, peak = run_generated_token(
            root, current_id, final_norm, lm_head,
            final_norm_name, lm_head_name, device,
            sampling_top_k, temperature,
        )
        after = cache_stats(root)

        delta_hits = int(after.get("hits", 0)) - int(before.get("hits", 0))
        delta_misses = int(after.get("misses", 0)) - int(before.get("misses", 0))
        step_hit_rate = delta_hits / max(delta_hits + delta_misses, 1) * 100.0

        generated.append(next_id)
        current_id = next_id
        text = tokenizer.decode([next_id], skip_special_tokens=True)
        print(text, end="", flush=True)
        print(
            f"\n  [step {step:02d}] token={next_id} | "
            f"time={elapsed:.3f}s | "
            f"step_hit_rate={step_hit_rate:.1f}% | "
            f"global_hit_rate={after.get('hit_rate', 0.0):.2f}% | "
            f"ram_hit={after.get('ram_hit_rate', 0.0):.2f}% | "
            f"vram_hit={after.get('vram_hit_rate', 0.0):.2f}% | "
            f"expert_vram_hit={after.get('vram_expert_hit_rate', 0.0):.2f}% | "
            f"stream_hit={after.get('vram_stream_hit_rate', 0.0):.2f}% | "
            f"hits+{delta_hits} misses+{delta_misses} | "
            f"peak_logit={peak:.4f}",
            flush=True,
        )

        if eos_id is not None and next_id == int(eos_id):
            break

    print()
    print(f"  resposta: {len(generated)} tokens | wall={perf_counter() - turn_start:.3f}s")
    print_cache(root, "after turn")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch Qwen3.6 router mini-chat test")
    parser.add_argument("root", type=Path)
    parser.add_argument("--prompt", action="append", required=True, help="Prompt to test; repeat --prompt for multiple turns")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=20, help="Sampling top-k")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be >= 1")
    if args.top_k < 1:
        raise SystemExit("--top-k must be >= 1")

    root = args.root.resolve()
    tokenizer = load_tokenizer(root)
    lm_name, lm_head, _ = load_lm_head(root)
    norm_name, final_norm = load_final_norm(root)

    if args.device == "cuda":
        cached._configure_vram_limit("cuda")
        cached.cached_runtime_tensor(root, norm_name, "cuda", dtype=torch.float32)
        cached.cached_runtime_tensor(root, lm_name, "cuda", dtype=torch.float16)

    print("op=batch-mini-chat")
    print("mode=experimental-stateless-autoregressive")
    print("warning=no DeltaNet recurrent state or full-attention KV cache yet")
    print("cache=hierarchical-vram-ram-ssd")
    print("vram_policy=resident-60pct-hot-experts-20pct-stream-20pct")
    print("vram_dequantized_cache=fp16")
    print("cuda_compute=fp16-autocast")
    print(f"prompts={len(args.prompt)}")
    print(f"device={args.device}")
    print(f"max_new_tokens={args.max_new_tokens}")
    print(f"sampling_top_k={args.top_k}")
    print(f"temperature={args.temperature}")
    print(f"lm_head={lm_name} shape={tuple(lm_head.shape)}")
    print(f"final_norm={norm_name} shape={tuple(final_norm.shape)}")
    print_cache(root, "initial")

    total_start = perf_counter()
    for index, prompt in enumerate(args.prompt, start=1):
        print(f"\n===== TURN {index}/{len(args.prompt)} =====")
        generate_response(
            root, prompt, tokenizer, final_norm, lm_head,
            norm_name, lm_name, args.device,
            max_new_tokens=args.max_new_tokens,
            sampling_top_k=args.top_k,
            temperature=args.temperature,
        )

    print("\n===== SUMMARY =====")
    print(f"turns={len(args.prompt)}")
    print(f"total_wall={perf_counter() - total_start:.3f}s")
    print_cache(root, "final")

    del lm_head, final_norm
    gc.collect()


if __name__ == "__main__":
    main()
