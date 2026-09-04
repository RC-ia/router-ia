# router-ia

Experimental MoE inference harness for Qwen3.6-35B-A3B, written from
scratch in PyTorch without `llama.cpp`.

The project started as byte-level GGUF expert indexing, then evolved
into a 40-layer hybrid runtime and an end-to-end generation path that
runs on CPU or CUDA with hierarchical RAM/VRAM caches, expert streaming,
and bounded async prefetch. This README tracks what is actually in the
tree right now, not what was planned a month ago.

## Target model

Qwen3.6-35B-A3B:

- 40 transformer blocks in a 3:1 hybrid layout: 30 Gated DeltaNet
  (linear-attention) layers + 10 full-attention layers
- 256 routed experts per MoE layer
- top-8 routed experts per token
- 1 shared expert per layer
- 2048 hidden size
- ~35B total parameters / ~3B active per token

The hardware target remains **4 GB VRAM + 8 GB RAM**.

## Current state

The project is now divided into three functional layers:

### 1. GGUF indexing (frozen — superseded by the FP8 path)

| Module | Status |
|---|---|
| `gguf_inspect.py` | done — generic GGUF metadata + tensor enumeration |
| `mapper.py` | done — packed GGUF expert byte ranges for Qwen3.6 |
| `expert_cache.py` | done — 2-tier RAM/VRAM LRU cache over packed GGUF bytes; VRAM currently holds raw quantized bytes |
| `expert_runner.py` | done — single-expert GGUF execution via `ggml.dll` ctypes (IQ3_XXS / IQ4_XS) |

This path is kept as a correctness probe and residency simulator, but it
is not on the active inference path. The GGUF single-expert runner still
dequantizes on CPU.

### 2. FP8 Safetensors and expert-cache layer

| Module | Status |
|---|---|
| `safetensors_mapper.py`, `qwen36_probe.py` | done — header-only inspection of the FP8 checkpoint |
| `qwen36_layer1_structure_probe.py`, `qwen36_load_trace.py` | done — crash-resistant Safetensors load tracer and one-layer structure probe |
| `fp8_expert_cache.py`, `fp8_expert_cache_v2.py` | done — RAM/VRAM LRU cache for FP8 routed experts; v2 sidesteps `safe_open.get_dtype()` |
| `fp8_expert_runner.py` | done — single-expert FP8 execution (E4M3 + 128×128 inverse scales) |
| `qwen36_dequant.py` | done — memory-conscious FP8 blockwise dequant + batched triplet path |
| `qwen36_router.py` | done — top-k router via `mlp.gate.weight` |
| `qwen36_lru_benchmark.py` | done — repeated single-token RAM LRU benchmark |
| `qwen36_cached_loop.py` | done — hierarchical RAM/VRAM tensor cache wrapped around the reference runtime; configurable cache budgets and expert streaming |

### 3. Validated runtime + generation path

The mathematical runtime now covers the complete hybrid Qwen3.6 block:

- 30 Gated DeltaNet / linear-attention layers + 10 full-attention layers.
- Top-8 MoE routing over 256 experts plus the shared expert on every layer.
- FP8 expert weights with blockwise dequantization for execution.
- Gated RMSNorm with the validated computation order.
- Full-attention RoPE and KV-cache implementation validated independently.
- Gated DeltaNet Q/K normalization and recurrent update validated against
  the reference recurrence.
- Hierarchical RAM/VRAM caching and routed-expert streaming for the
  low-memory hardware target.

The 40-layer runner remains the reference executor, while the chat path
is now the place where these pieces are assembled into text generation.

| Module | What it does |
|---|---|
| `qwen36_op_probe.py` | Building blocks: tensor loading, projection loading, embedding rows, RMSNorm, FP8 dequant |
| `qwen36_gated_norm_probe.py` | Gated RMSNorm validation |
| `qwen36_out_proj_probe.py` | Isolated out-projection validation |
| `qwen36_residual_probe.py` | Layer-0 residual validation |
| `qwen36_shared_expert_probe.py` | Shared-expert execution validation |
| `qwen36_moe8_probe.py` | Router-selected top-k experts + aggregation validation |
| `qwen36_expert_probe.py`, `qwen36_to_router_probe.py` | Expert/router isolation |
| `qwen36_chain_probe.py` | Consecutive-layer execution |
| `qwen36_delta_sequence_probe.py` | Gated Delta Rule recurrence across tokens |
| `qwen36_linear_attention_hf.py` | Reference-aligned Gated DeltaNet implementation |
| `qwen36_linear_attention_hf_probe.py` | HF/reference comparison probe |
| `qwen36_40layer_loop.py` | Canonical sequential 40-layer reference executor; detects linear-vs-full attention and runs the full MoE block |
| `qwen36_cached_loop.py` | Cache infrastructure around the canonical executor |
| `qwen36_chat_batch.py` | **Official generator**: end-to-end tokenizer → 40-layer runtime → final norm → `lm_head` → token sampling/decoding |
| `qwen36_chat_batch_fused.py` | Experimental optimized generator path with fused routed-expert FP8 staging |
| `qwen36_mini_chat.py` | Earlier mini-chat smoke path; retained as a smaller diagnostic |
| `qwen36_cuda_loop.py` | CUDA-first runtime diagnostics |
| `qwen36_profile.py`, `qwen36_profile_chat.py` | Profiling tools for the runtime and generator |

## Official generator: `qwen36_chat_batch.py`

`qwen36_chat_batch.py` is the **official generation entry point** for
this project.

It is responsible for the end-to-end path:

```text
prompt
  ↓
tokenizer
  ↓
embedding
  ↓
40 hybrid transformer layers
  ├─ Gated DeltaNet / linear attention
  ├─ full attention
  └─ top-8 MoE + shared expert
  ↓
final RMSNorm
  ↓
lm_head
  ↓
logits / sampling
  ↓
next token
  ↓
decode
```

The generator already exercises the real tokenizer, full 40-layer
forward path, final normalization, LM head, and autoregressive token
loop. Its cache path is backed by `qwen36_cached_loop.py`, including the
RAM/VRAM hierarchy and routed expert staging.

The current functional gate before calling the generator fully faithful
is **persistent state integration**: generated tokens must reuse the
Gated DeltaNet recurrent states and the full-attention KV caches instead
of recomputing each token from only the previous token. Those two pieces
are already validated independently; the remaining work is to wire them
into this official path and verify multi-token output against a reference
implementation.

`qwen36_chat_batch_fused.py` is deliberately kept as an optimization
variant. It should not become a second official implementation.

## Validation status

The following building blocks have already been validated independently:

| Area | Status |
|---|---|
| 40-layer hybrid dispatch | **PASS** — all 40 layers execute with the expected 30/10 attention split |
| MoE router / top-8 selection | **PASS** — runtime routing agrees with `mlp.gate.weight` reference routing |
| FP8 expert MLP | **PASS** — routed expert outputs match the direct reference path |
| Shared expert + aggregation | **PASS** |
| Gated RMSNorm | **PASS** |
| Gated DeltaNet recurrence | **PASS** — reference-aligned state update validated across multiple tokens |
| Full-attention KV cache | **PASS** — K/V, lengths, reset and detach behavior validated |
| Hierarchical RAM/VRAM cache | **PASS** — cache hits, misses, streaming and eviction paths exercised |
| End-to-end tokenizer → runtime → `lm_head` | **RUNNING** — generator path exists; final stateful integration is the remaining correctness gate |

## What remains before optimization

The project is deliberately close to the optimization phase.

### Final correctness steps

1. Wire persistent Gated DeltaNet recurrent state into
   `qwen36_chat_batch.py`.
2. Wire the persistent full-attention KV cache into
   `qwen36_chat_batch.py`.
3. Run multi-token correctness tests against the reference implementation.
4. Confirm that the generated text is stable and valid across several
   short prompts and generation lengths.

### Then: optimization only

Once the stateful generator passes correctness checks, the remaining
work is primarily performance and memory engineering:

- reduce RAM ↔ VRAM transfers;
- improve routed-expert cache hit rate;
- overlap prefetch with computation;
- fuse/accelerate FP8 dequantization;
- reduce Python and allocator overhead;
- improve batching and token throughput;
- evaluate `torch.compile`, Triton and specialized CUDA kernels;
- tune VRAM/RAM/SSD residency policies for the target hardware.

SSD cold storage and the older GGUF path are optional memory/compatibility
work, not blockers for the official generator.

## CLI entry points

Only one script is installed as a console entry:

```bash
router-inspect path/to/model.gguf
```

Everything else is reachable as a module. The main runtime tools are:

```bash
# Canonical 40-layer reference executor
python -m router_ia.qwen36_40layer_loop /path/to/safetensors/dir --device cuda

# Hierarchical cache wrapper around the reference executor
python -m router_ia.qwen36_cached_loop /path/to/safetensors/dir --device cuda

# OFFICIAL GENERATOR
python -m router_ia.qwen36_chat_batch /path/to/safetensors/dir \
  --device cuda --max-new-tokens 16

# Experimental fused optimization variant
python -m router_ia.qwen36_chat_batch_fused /path/to/safetensors/dir

# Smaller diagnostic chat path
python -m router_ia.qwen36_mini_chat /path/to/safetensors/dir

# Validation probes
python -m router_ia.qwen36_layer0_executor /path/to/safetensors/dir
python -m router_ia.qwen36_layer_executor /path/to/safetensors/dir --layer 0
python -m router_ia.qwen36_moe_validate /path/to/safetensors/dir
python -m router_ia.qwen36_lru_benchmark /path/to/safetensors/dir
```

## Constraints

- Do not modify `llama.cpp`.
- Keep the first implementation small and inspectable.
- Correctness first; performance comes later.
- `qwen36_chat_batch.py` is the single official generation path; other
  runners exist for reference, diagnostics, validation, or optimization
  experiments.

## Roadmap

1. **Official generator correctness:** finish persistent DeltaNet state
   + full-attention KV-cache integration in `qwen36_chat_batch.py`.
2. **Generation validation:** compare multi-token outputs against a
   reference and confirm valid text.
3. **Optimization pass:** cache policy, memory movement, prefetch,
   dequantization, batching, kernels/compiler, and throughput.
4. **Optional packaging/serving:** only after the runtime itself is
   stable and optimized.
