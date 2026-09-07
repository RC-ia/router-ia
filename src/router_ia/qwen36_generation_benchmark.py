from __future__ import annotations

"""Small autoregressive speed benchmark for the Qwen3.6 Router-IA runtime."""

import argparse
import time
from pathlib import Path

import torch

from . import qwen36_chat_batch as chat
from . import qwen36_mini_chat as mini


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Qwen3.6 generation speed")
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--prompt", default="Olá, explique em uma frase o que é uma CPU.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--sampling-top-k", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--indexed", action="store_true", help="Use the compact expert-location index")
    args = parser.parse_args()

    root = args.model_dir.resolve()
    device = args.device.lower()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    if args.indexed:
        # Apply the runtime patch before the first cache store is created.
        from . import qwen36_chat_indexed  # noqa: F401

    chat.cached._configure_vram_limit(device)
    tokenizer = mini.load_tokenizer(root)
    final_norm_name, final_norm = mini.load_final_norm(root)
    lm_head_name, lm_head, _ = mini.load_lm_head(root)

    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    if not prompt_ids:
        raise RuntimeError("Tokenizer produced no prompt tokens")

    state = chat.attention_cache.state_for(root, device)
    state.reset()
    chat.attention_cache.activate(root, state)

    print("op=generation-benchmark")
    print(f"mode={'indexed' if args.indexed else 'baseline'}")
    print(f"device={device}")
    print(f"prompt_tokens={len(prompt_ids)}")
    print(f"max_new_tokens={args.max_new_tokens}")
    print(f"prompt={args.prompt}")

    # Prefill: measure separately because it is not equivalent to decode speed.
    t0 = time.perf_counter()
    logits = None
    for token_id in prompt_ids:
        logits, _, _ = chat.run_forward_token(
            root,
            int(token_id),
            final_norm,
            lm_head,
            final_norm_name,
            lm_head_name,
            device,
        )
    if logits is None:
        raise RuntimeError("No logits produced during prefill")
    prefill_time = time.perf_counter() - t0
    next_id = chat.sample_next(logits, args.temperature, args.sampling_top_k)
    del logits

    # Decode benchmark: each iteration consumes the previously sampled token.
    decode_times: list[float] = []
    generated = []
    for _ in range(max(args.max_new_tokens - 1, 0)):
        start = time.perf_counter()
        next_id, elapsed, _ = chat.run_generated_token(
            root,
            int(next_id),
            final_norm,
            lm_head,
            final_norm_name,
            lm_head_name,
            device,
            args.sampling_top_k,
            args.temperature,
        )
        wall = time.perf_counter() - start
        decode_times.append(max(elapsed, wall))
        generated.append(int(next_id))

    decode_total = sum(decode_times)
    decode_tokens = len(decode_times)
    decode_tps = decode_tokens / decode_total if decode_total > 0 else 0.0
    attn = chat.attention_cache.stats(root)
    cache = chat.cache_stats(root)

    print(f"prefill_seconds={prefill_time:.6f}")
    print(f"prefill_tok_s={len(prompt_ids) / prefill_time:.4f}")
    print(f"decode_tokens={decode_tokens}")
    print(f"decode_seconds={decode_total:.6f}")
    print(f"decode_tok_s={decode_tps:.4f}")
    if decode_times:
        print(f"decode_ms_avg={decode_total / decode_tokens * 1000.0:.3f}")
        print(f"decode_ms_min={min(decode_times) * 1000.0:.3f}")
        print(f"decode_ms_max={max(decode_times) * 1000.0:.3f}")
    print(f"cache_hit_rate={cache.get('hit_rate', 0.0):.3f}")
    print(f"ram_hit_rate={cache.get('ram_hit_rate', 0.0):.3f}")
    print(f"vram_hit_rate={cache.get('vram_hit_rate', 0.0):.3f}")
    print(f"expert_vram_hit_rate={cache.get('vram_expert_hit_rate', 0.0):.3f}")
    print(f"stream_hit_rate={cache.get('vram_stream_hit_rate', 0.0):.3f}")
    print(f"ram_bytes={cache.get('ram_bytes', 0)}")
    print(f"vram_bytes={cache.get('vram_bytes', 0)}")
    print(f"attention_bytes={attn.get('bytes', 0)}")
    print(f"generated={tokenizer.decode(generated, skip_special_tokens=True)!r}")


if __name__ == "__main__":
    main()
