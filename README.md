# router-ia

Experimental MoE inference harness for Qwen3.6-35B-A3B, written from
scratch in PyTorch without `llama.cpp`.

The project started as byte-level GGUF expert indexing, then evolved
into a full 40-layer single-token runtime that runs on CPU or CUDA
with hierarchical RAM/VRAM caches and bounded async expert prefetch.
This README tracks what is actually in the tree right now, not what
was planned a month ago.

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

The repo is split into three rough layers:

### 1. GGUF indexing (frozen — superseded by the FP8 path)

| Module | Status |
|---|---|
| `gguf_inspect.py` | done — generic GGUF metadata + tensor enumeration |
| `mapper.py` | done — packed GGUF expert byte ranges for Qwen3.6 |
| `expert_cache.py` | done — 2-tier RAM/VRAM LRU cache over packed GGUF bytes; VRAM currently holds raw quantized bytes |
| `expert_runner.py` | done — single-expert GGUF execution via `ggml.dll` ctypes (IQ3_XXS / IQ4_XS) |

This path is kept around as a correctness probe and a residency
simulator, but it is not on the active inference path. The GGUF
single-expert runner still dequantizes on CPU. A future revision will
either port it to GPU dequantization or drop it.

### 2. FP8 Safetensors cache layer

| Module | Status |
|---|---|
| `safetensors_mapper.py`, `qwen36_probe.py` | done — header-only inspection of the FP8 checkpoint |
| `qwen36_layer1_structure_probe.py`, `qwen36_load_trace.py` | done — crash-resistant Safetensors load tracer and one-layer structure probe |
| `fp8_expert_cache.py`, `fp8_expert_cache_v2.py` | done — RAM/VRAM LRU cache for FP8 routed experts; v2 sidesteps `safe_open.get_dtype()` |
| `fp8_expert_runner.py` | done — single-expert FP8 execution (E4M3 + 128×128 inverse scales) |
| `qwen36_dequant.py` | done — memory-conscious FP8 blockwise dequant + a batched triplet path used by the routed-expert streamer |
| `qwen36_router.py` | done — top-k router via `mlp.gate.weight` |
| `qwen36_lru_benchmark.py` | done — repeated single-token RAM LRU benchmark |

### 3. Single-token runtime (the active path)

The repo no longer ends at "single expert probe". There is now a
real 40-layer single-token forward path with hierarchical caching,
async prefetch, and a fused routed-expert FP8 staging path.

| Module | What it does |
|---|---|
| `qwen36_op_probe.py` | Building block: `load_tensor`, `load_projection`, `load_embedding_row`, `rmsnorm`, FP8 blockwise dequant |
| `qwen36_gated_norm_probe.py` | Gated RMSNorm (Qwen3.6 RMSNorm with the gating term) |
| `qwen36_out_proj_probe.py` | Isolated out-projection (the linear after the value heads) |
| `qwen36_residual_probe.py` | Layer-0 residual probe |
| `qwen36_shared_expert_probe.py` | Shared-expert execution in isolation |
| `qwen36_moe8_probe.py` | Router-selected top-k experts + aggregation (Layer 0 reference) |
| `qwen36_expert_probe.py`, `qwen36_to_router_probe.py` | One-expert and one-router isolation |
| `qwen36_chain_probe.py` | Consecutive layers for one token |
| `qwen36_delta_sequence_probe.py` | Gated Delta Rule recurrence across two tokens (state persistence) |
| `qwen36_layer0.py`, `qwen36_layer0_executor.py` | Validated single-token Layer-0 forward (DeltaNet + MoE) |
| `qwen36_layer_executor.py` | Generic single-token executor for any layer |
| `qwen36_40layer_loop.py` | Sequential 40-layer single-token loop; detects linear-vs-full attention per layer, runs MoE + shared expert; CPU and CUDA |
| `qwen36_cached_loop.py` | Same 40-layer loop wrapped in a hierarchical cache layer with env-tunable budgets (`QWEN36_CACHE_GB`, `QWEN36_VRAM_GB`, `QWEN36_VRAM_CACHE_GB`, `QWEN36_RESIDENT_VRAM_RATIO`, `QWEN36_VRAM_STREAM_GB`, `QWEN36_EXPERT_BONUS`) |
| `qwen36_cuda_loop.py` | CUDA-first runner with preflight and load diagnostics |
| `qwen36_cuda_prefetch_test.py`, `qwen36_cuda_tokens_test.py` | GPU-resident layer prefetch cache + benchmark for token-pass cache reuse |
| `qwen36_moe_weight_prefetch_loop.py` | Bounded async prefetch of guaranteed MoE weights before the layer needs them |
| `qwen36_planned_loop.py` | Planned expert prefetch (router-aware schedule) |
| `qwen36_mini_chat.py` | Mini chat smoke test that does a real `token → router → hidden → lm_head` cycle per generated token; **not yet a stateful decoder** (see below) |
| `qwen36_chat_batch.py` | Batch mini-chat smoke test over the cached loop |
| `qwen36_chat_batch_fused.py` | Same but with fused routed-expert FP8 staging: one H2D copy per stacked expert triplet + one vectorized GPU dequant |
| `qwen36_profile.py`, `qwen36_profile_chat.py` | Profilers around the existing reference loops |

The `qwen36_moe_probe.py` / `qwen36_moe_validate.py` pair at the top
level are now a partial MoE-only probe and a runtime-vs-checkpoint
validator — both still work but are below the 40-layer loop in the
hierarchy.

### What is implemented but limited

- **Single-token only.** Each generated token is decoded from the
  previous token alone. The 40-layer runner does not preserve the
  recurrent DeltaNet state or the full-attention KV cache across
  generated tokens. `qwen36_mini_chat` documents this explicitly:
  "router/runtime smoke-test, not yet a faithful autoregressive
  decoder."
- **Stateful decoding is partially started.** `qwen36_delta_sequence_probe.py`
  shows Gated Delta Rule recurrence working across two tokens; that
  is the seed for the real autoregressive loop but is not yet wired
  into the chat path.
- **Two expert-cache flavours** (GGUF + FP8) are kept side-by-side.
  They share the LRU shape but not the API. Unification is deferred.

### What is intentionally not implemented yet

- Stateful autoregressive decoder across multiple generated tokens.
- Real KV-cache for the 10 full-attention layers.
- SSD cold tier (was the original milestone 6).
- GPU-side dequantization of GGUF experts (the GGUF branch is frozen).
- `torch.compile`, FlashAttention, or any Triton kernel.
- Any HTTP / OpenAI-compatible serving layer.
- Tests as a separate suite (`tests/` does not exist); validation is
  done through the `*_validate.py` and `*_probe.py` scripts.

## CLI entry points

Only one script is installed as a console entry:

```bash
router-inspect path/to/model.gguf
```

Everything else is reachable as a module. The most useful for
end-to-end smoke tests:

```bash
# CPU/CUDA 40-layer single-token loop (the canonical reference path)
python -m router_ia.qwen36_40layer_loop /path/to/safetensors/dir --device cpu

# Same loop, but routed through the hierarchical RAM/VRAM caches
python -m router_ia.qwen36_cached_loop /path/to/safetensors/dir --device cuda

# CUDA-first runner with load diagnostics
python -m router_ia.qwen36_cuda_loop /path/to/safetensors/dir

# Mini chat (router + lm_head, no stateful decoder yet)
python -m router_ia.qwen36_mini_chat /path/to/safetensors/dir

# Batched chat over the cached loop, with fused expert FP8 staging
python -m router_ia.qwen36_chat_batch_fused /path/to/safetensors/dir

# FP8 expert / router / cache probes
python -m router_ia.qwen36_layer0_executor /path/to/safetensors/dir
python -m router_ia.qwen36_layer_executor  /path/to/safetensors/dir --layer 0
python -m router_ia.qwen36_moe_validate    /path/to/safetensors/dir
python -m router_ia.qwen36_lru_benchmark   /path/to/safetensors/dir
```

The `qwen36_moe_probe.py` file at the repo root is a compatibility
launcher for `python qwen36_moe_probe.py ...`.

## Constraints

- Do not modify `llama.cpp`.
- Keep the first implementation small and inspectable.
- Correctness first; performance comes later.

## Roadmap

1. Wire the persistent DeltaNet state and KV-cache into the
   `qwen36_mini_chat` path so it becomes a real autoregressive
   decoder (the Gated Delta Rule recurrence already works across two
   tokens in `qwen36_delta_sequence_probe`).
2. Unify the GGUF and FP8 expert caches into one
   `MoEExpertCache` over both formats.
3. SSD cold tier (original milestone 6).
4. GPU dequantization of GGUF experts or, more likely, drop the GGUF
   runner entirely and keep the FP8 path only.
5. Decide whether to lift this into a serving layer or hand it to
   `llama.cpp` / `vLLM` and keep this repo as the measurement
   harness.
