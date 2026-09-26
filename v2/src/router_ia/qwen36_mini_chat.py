from __future__ import annotations

"""Experimental Qwen3.6 mini chat using the router inference path.

IMPORTANT:
    This is a router/runtime smoke-test, not yet a faithful autoregressive
    Qwen3.6 decoder. The current 40-layer runner does not preserve the
    recurrent DeltaNet state or full-attention KV cache across tokens.

The chat therefore performs a real token -> router -> hidden -> lm_head cycle,
but each generated token is decoded from the previous token alone. It is useful
to measure end-to-end routing, weight-cache reuse, output projection and token
selection while the stateful decoder is being implemented.
"""

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from . import qwen36_cached_loop as cached
from . import qwen36_40layer_loop as base

DEFAULT_MAX_NEW_TOKENS = 16
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_K = 20


def find_tensor_name(root: Path, suffixes: tuple[str, ...]) -> str:
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise SystemExit(f"Missing index: {index_path}")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    names = list(payload.get("weight_map", {}).keys())
    for suffix in suffixes:
        matches = [name for name in names if name.endswith(suffix)]
        if len(matches) == 1:
            return matches[0]
        if matches:
            # Prefer the text-model path when multiple names are present.
            for name in matches:
                if "language_model" in name:
                    return name
            return matches[0]
    raise KeyError(f"Could not find tensor with suffixes={suffixes}")


def load_lm_head(root: Path) -> tuple[str, torch.Tensor, str]:
    name = find_tensor_name(root, ("lm_head.weight",))
    weight = cached._cached_load_tensor(root, name, device="cpu").float()
    return name, weight, "direct"


def load_final_norm(root: Path) -> tuple[str, torch.Tensor]:
    name = find_tensor_name(root, ("language_model.norm.weight", "model.norm.weight", ".norm.weight"))
    weight = cached._cached_load_tensor(root, name, device="cpu").float()
    return name, weight


def load_tokenizer(root: Path):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "transformers is required for chat mode. Install it with: pip install transformers"
        ) from exc

    return AutoTokenizer.from_pretrained(str(root), local_files_only=True, use_fast=True)


def sample_next(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    logits = logits.float().reshape(-1)
    if temperature <= 0:
        return int(torch.argmax(logits).item())

    temperature = max(float(temperature), 1e-5)
    if top_k > 0 and top_k < logits.numel():
        values, indices = torch.topk(logits, top_k)
        probs = torch.softmax(values / temperature, dim=-1)
        choice = torch.multinomial(probs, 1)
        return int(indices[choice].item())

    probs = torch.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(probs, 1).item())


def generate_token(
    root: Path,
    token_id: int,
    final_norm: torch.Tensor,
    lm_head: torch.Tensor,
    device: str,
    top_k: int,
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
        x, *_ = base.moe_step(root, layer, residual, top_k=base.DEFAULT_LAYERS * 0 + 8, device=device)
        del residual

    x = base.rmsnorm(x, final_norm.to(device))
    logits = F.linear(x, lm_head.to(device))
    next_id = sample_next(logits, temperature, top_k)

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = perf_counter() - start
    peak = float(torch.max(logits).item())

    del x, logits
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return next_id, elapsed, peak


def cache_stats(root: Path) -> dict[str, int | float]:
    store = cached._stores.get(root.resolve())
    if store is None:
        return {}
    return store.cache.snapshot()


def print_cache(root: Path, label: str) -> None:
    stats = cache_stats(root)
    if not stats:
        print(f"  LRU {label}: unavailable")
        return
    print(
        f"  LRU {label}: "
        f"items={stats['items']} | "
        f"ram={stats['bytes'] / 1024**2:.1f}/{cached.CACHE_BUDGET_BYTES / 1024**2:.1f} MiB | "
        f"hits={stats['hits']} | misses={stats['misses']} | "
        f"hit_rate={stats['hit_rate']:.2f}% | "
        f"evictions={stats['evictions']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimental Qwen3.6 router mini chat")
    parser.add_argument("root", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
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

    print("op=mini-chat")
    print("mode=experimental-stateless-autoregressive")
    print("warning=no DeltaNet recurrent state or full-attention KV cache yet")
    print(f"lm_head: {lm_name} shape={tuple(lm_head.shape)} dtype={lm_head.dtype}")
    print(f"final_norm: {norm_name} shape={tuple(final_norm.shape)}")
    print(f"device: {args.device}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"temperature: {args.temperature}")
    print(f"sampling_top_k: {args.top_k}")
    print_cache(root, "initial")

    eos_id = getattr(tokenizer, "eos_token_id", None)
    print("\nMini chat. Type /exit to quit, /clear to reset.")

    conversation_token_ids: list[int] = []
    while True:
        try:
            prompt = input("\nVocê> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if prompt == "/exit":
            break
        if prompt == "/clear":
            conversation_token_ids.clear()
            print("Histórico limpo.")
            continue
        if not prompt:
            continue

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not prompt_ids:
            print("IA> [nenhum token produzido pelo tokenizer]")
            continue

        # The current runtime cannot carry sequence state correctly. Keep the
        # latest user token as the model input for this experimental chat.
        current_id = int(prompt_ids[-1])
        conversation_token_ids.extend(prompt_ids)

        generated_ids: list[int] = []
        prompt_start = perf_counter()
        print("IA> ", end="", flush=True)

        for step in range(1, args.max_new_tokens + 1):
            before = cache_stats(root)
            next_id, elapsed, peak_logit = generate_token(
                root,
                current_id,
                final_norm,
                lm_head,
                args.device,
                base.DEFAULT_TOP_K,
                args.temperature,
            )
            after = cache_stats(root)

            generated_ids.append(next_id)
            current_id = next_id
            delta_hits = int(after.get("hits", 0)) - int(before.get("hits", 0))
            delta_misses = int(after.get("misses", 0)) - int(before.get("misses", 0))
            step_rate = delta_hits / max(delta_hits + delta_misses, 1) * 100.0

            text = tokenizer.decode([next_id], skip_special_tokens=True)
            print(text, end="", flush=True)
            print(
                f"\n  [step {step:02d}] token={next_id} | "
                f"time={elapsed:.3f}s | "
                f"step_hit_rate={step_rate:.1f}% | "
                f"peak_logit={peak_logit:.4f}",
                flush=True,
            )

            if eos_id is not None and next_id == int(eos_id):
                break

        print()
        total = perf_counter() - prompt_start
        print(f"  resposta: {len(generated_ids)} tokens | wall={total:.3f}s")
        print_cache(root, "after turn")

    del lm_head, final_norm
    gc.collect()
    print_cache(root, "final")


if __name__ == "__main__":
    main()
