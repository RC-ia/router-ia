from __future__ import annotations

"""Runtime optimizations for the Qwen3.6 token loop.

These patches keep allocator churn out of the hot path, cache the hybrid
attention layout, bound the fused expert cache, and enable real autoregressive
state: full-attention K/V cache plus persistent Gated DeltaNet state.
"""

import gc
import sys
from pathlib import Path
from time import perf_counter
from typing import Callable, TypeVar

import torch
import torch.nn.functional as F

from . import qwen36_40layer_loop as base
from . import qwen36_attention_cache as attention_state
from . import qwen36_chat_batch as chat
from . import qwen36_cached_loop as cached
from . import qwen36_expert_cache as expert_cache
from .qwen36_mini_chat import sample_next

_T = TypeVar("_T")


_ORIGINAL_ATTENTION_TYPE = base.attention_type
_ATTENTION_TYPES: dict[Path, tuple[str, ...]] = {}


def _cached_attention_type(root: Path, layer: int) -> str:
    key = root.resolve()
    cached_types = _ATTENTION_TYPES.get(key)
    if cached_types is None:
        detected = tuple(_ORIGINAL_ATTENTION_TYPE(key, index) for index in range(base.DEFAULT_LAYERS))
        _ATTENTION_TYPES[key] = detected
        cached_types = detected
    return cached_types[int(layer)]


def _without_allocator_flush(fn: Callable[..., _T]) -> Callable[..., _T]:
    """Run a hot-path function without forced Python/CUDA cache flushing."""
    def wrapped(*args, **kwargs):
        old_collect = gc.collect
        old_empty_cache = torch.cuda.empty_cache
        gc.collect = lambda: 0
        torch.cuda.empty_cache = lambda: None
        try:
            return fn(*args, **kwargs)
        finally:
            gc.collect = old_collect
            torch.cuda.empty_cache = old_empty_cache

    return wrapped


def _configure_fused_cache_budget() -> None:
    """Keep the fused runner's cache pools within roughly 3 GiB total."""
    invocation = " ".join(str(arg) for arg in sys.argv)
    if "qwen36_chat_batch_fused" not in invocation:
        return

    expert_budget = 1 * 1024**3
    stream_budget = max(
        cached.VRAM_CACHE_BUDGET_BYTES
        - cached.RESIDENT_VRAM_BUDGET_BYTES
        - expert_budget,
        0,
    )
    cached.STREAM_BUDGET_BYTES = stream_budget
    cached.STREAM_GB = stream_budget / 1024**3

    if getattr(expert_cache.RoutedExpertCache, "_router_ia_budget_patch", False):
        return

    original_init = expert_cache.RoutedExpertCache.__init__

    def bounded_init(self, budget_bytes: int, layers: int = expert_cache.MODEL_LAYERS) -> None:
        requested = max(int(budget_bytes), 0)
        effective = max(requested, expert_budget) if requested else 0
        original_init(self, effective, layers)
        if effective:
            # Q4 is host-RAM backing and therefore must not consume VRAM slots.
            self.q4_slots = expert_cache.Q4_SLOTS_PER_LAYER
            self.slots_per_layer = self.fp8_slots + self.q4_slots
            self.total_slots = self.slots_per_layer * self.layers

    expert_cache.RoutedExpertCache.__init__ = bounded_init
    expert_cache.RoutedExpertCache._router_ia_budget_patch = True


def _state(root: Path, device: str) -> attention_state.AttentionState:
    return attention_state.state_for(root, device)


def _stateful_attention_step(root: Path, layer: int, x0: torch.Tensor, device: str) -> torch.Tensor:
    return attention_state.step_attention(root, layer, x0, device)


def _run_token_hidden(root: Path, token_id: int, device: str) -> torch.Tensor:
    """Consume one token through all layers and update persistent attention state."""
    attention_state.activate(root, _state(root, device))
    x = base.load_embedding_row(root, int(token_id)).reshape(1, base.HIDDEN).to(device).float()
    try:
        for layer in range(base.DEFAULT_LAYERS):
            residual = _stateful_attention_step(root, layer, x, device)
            x, *_ = chat.batched_moe_step(root, layer, residual, top_k=8, device=device)
            del residual
        return x
    except Exception:
        del x
        raise


def _run_generated_token_stateful(
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
    attention_state.activate(root, _state(root, device))
    try:
        result = _ORIGINAL_CHAT_RUN_GENERATED_TOKEN(
            root,
            token_id,
            final_norm,
            lm_head,
            final_norm_name,
            lm_head_name,
            device,
            sampling_top_k,
            temperature,
        )
        return result
    finally:
        attention_state.deactivate(root)


def _generate_response_stateful(
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
    """Prefill the prompt once, then decode with persistent attention state."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        print("IA> [nenhum token produzido pelo tokenizer]")
        return

    state = attention_state.reset(root)
    state.bind(device)
    attention_state.activate(root, state)
    print(f"\nVocê> {prompt}")
    print(f"  prompt tokens={len(prompt_ids)} | seed_token={int(prompt_ids[-1])}")
    turn_start = perf_counter()

    try:
        hidden: torch.Tensor | None = None
        for token_id in prompt_ids:
            if hidden is not None:
                del hidden
            hidden = _run_token_hidden(root, int(token_id), device)

        assert hidden is not None
        if device == "cuda":
            final_norm_runtime = cached.cached_runtime_tensor(root, final_norm_name, device, dtype=torch.float32)
            lm_head_runtime = cached.cached_runtime_tensor(root, lm_head_name, device, dtype=torch.float16)
        else:
            final_norm_runtime = final_norm
            lm_head_runtime = lm_head

        hidden = base.rmsnorm(hidden, final_norm_runtime)
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = F.linear(hidden.to(lm_head_runtime.dtype), lm_head_runtime)
        else:
            logits = F.linear(hidden, lm_head_runtime)
        current_id = int(sample_next(logits, temperature, sampling_top_k))
        if device == "cuda":
            torch.cuda.synchronize()
        del hidden, logits
        if device == "cuda":
            del final_norm_runtime, lm_head_runtime

        eos_id = getattr(tokenizer, "eos_token_id", None)
        generated: list[int] = []
        print("IA> ", end="", flush=True)

        if max_new_tokens > 0:
            generated.append(current_id)
            print(tokenizer.decode([current_id], skip_special_tokens=True), end="", flush=True)

        if max_new_tokens > 0 and (eos_id is None or current_id != int(eos_id)):
            for step in range(2, max_new_tokens + 1):
                before = chat.cache_stats(root)
                next_id, elapsed, peak = _run_generated_token_stateful(
                    root,
                    current_id,
                    final_norm,
                    lm_head,
                    final_norm_name,
                    lm_head_name,
                    device,
                    sampling_top_k,
                    temperature,
                )
                after = chat.cache_stats(root)
                delta_hits = int(after.get("hits", 0)) - int(before.get("hits", 0))
                delta_misses = int(after.get("misses", 0)) - int(before.get("misses", 0))
                step_hit_rate = delta_hits / max(delta_hits + delta_misses, 1) * 100.0
                generated.append(next_id)
                current_id = next_id
                print(tokenizer.decode([next_id], skip_special_tokens=True), end="", flush=True)
                print(
                    f"\n  [step {step:02d}] token={next_id} | time={elapsed:.3f}s | "
                    f"step_hit_rate={step_hit_rate:.1f}% | global_hit_rate={after.get('hit_rate', 0.0):.2f}% | "
                    f"ram_hit={after.get('ram_hit_rate', 0.0):.2f}% | vram_hit={after.get('vram_hit_rate', 0.0):.2f}% | "
                    f"expert_vram_hit={after.get('vram_expert_hit_rate', 0.0):.2f}% | stream_hit={after.get('vram_stream_hit_rate', 0.0):.2f}% | "
                    f"hits+{delta_hits} misses+{delta_misses} | peak_logit={peak:.4f}",
                    flush=True,
                )
                if eos_id is not None and next_id == int(eos_id):
                    break

        attention = attention_state.stats(root)
        print()
        print(f"  resposta: {len(generated)} tokens | wall={perf_counter() - turn_start:.3f}s")
        print(
            f"  attention_state: tokens_seen={attention['tokens_seen']} | "
            f"full_layers={attention['full_layers_cached']} | full_kv={attention['full_bytes'] / 1024**2:.1f} MiB | "
            f"linear_layers={attention['linear_layers_cached']} | linear_state={attention['linear_bytes'] / 1024**2:.1f} MiB"
        )
        chat.print_cache(root, "after turn")
    finally:
        attention_state.deactivate(root)


_configure_fused_cache_budget()

base.attention_type = _cached_attention_type

_ORIGINAL_CHAT_RUN_GENERATED_TOKEN = chat.run_generated_token
base.linear_attention_step = _without_allocator_flush(_stateful_attention_step)
base.full_attention_step = _without_allocator_flush(_stateful_attention_step)
chat.run_generated_token = _without_allocator_flush(_run_generated_token_stateful)
chat.generate_response = _generate_response_stateful
