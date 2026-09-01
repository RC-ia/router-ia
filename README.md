# router-ia

Experimental MoE inference research harness for Qwen3.6-35B-A3B.

This is **not a finished inference runtime**. It is a set of probes,
indexers, and a hierarchical expert cache used to study how a hybrid
MoE model behaves when its experts are spread across disk, RAM and
CUDA VRAM. Most modules are single-expert or single-token correctness
harnesses, not production executors.

The intended long-term goal is still a minimal GGUF runtime that keeps
hot experts in VRAM and warm experts in RAM, but the GGUF decode path
is currently paused in favour of the FP8 Safetensors path, which is
where the latest results live.

## Target model

Qwen3.6-35B-A3B is a hybrid MoE model with:

- 40 transformer blocks
- 256 routed experts per MoE layer
- top-8 routed experts per token
- 1 shared expert per layer
- 2048 hidden size
- ~35B total parameters / ~3B active parameters per token
- 30 Gated DeltaNet layers + 10 full-attention layers

Hardware target declared in the original milestones: **4 GB VRAM + 8 GB RAM**.

## Current state (what is actually in the repo)

| Area | Module | Status |
|---|---|---|
| GGUF metadata + tensor enumeration | `gguf_inspect.py`, `mapper.py` | **done** — exposes packed expert byte ranges without decoding quantization |
| Hierarchical expert cache (RAM ⇄ VRAM) | `expert_cache.py` | **done for GGUF** — but the VRAM tier currently stores raw quantized bytes; dequantization still happens on the CPU in `expert_runner.py`. Treat it as a residency simulator, not a hot path |
| Single-expert GGUF execution (CPU/CUDA) | `expert_runner.py` | **done** — IQ3_XXS / IQ4_XS dequantization via `ggml.dll` ctypes, runs `silu(xW_gate) * (xW_up) @ W_down` for one expert |
| Safetensors FP8 inspector | `safetensors_mapper.py`, `qwen36_probe.py` | **done** — reads only Safetensors headers, no payload load |
| FP8 expert cache (v1 + v2 compat fix) | `fp8_expert_cache.py`, `fp8_expert_cache_v2.py` | **done** — same LRU 2-tier shape as the GGUF cache; v2 sidesteps `safe_open.get_dtype()` |
| Single-expert FP8 execution | `fp8_expert_runner.py` | **done** — E4M3 + 128×128 inverse scales, dequant + matmul on CPU/CUDA |
| Qwen3.6 router (top-k selector) | `qwen36_router.py` | **done** — loads only `mlp.gate.weight` per layer and returns top-8 with softmax weights |
| Partial MoE forward probe | `qwen36_moe_probe.py` | **partial** — token id → embedding → router → FP8 expert MLPs → weighted aggregation. Attention / DeltaNet, normalization, shared expert, residuals and the LM head are explicitly out of scope |
| Runtime vs checkpoint validation | `qwen36_moe_validate.py` | **done** — compares the cached runtime path against direct checkpoint math for the same embedding row |

### What is intentionally not implemented yet

- Full transformer block (no attention, no Gated DeltaNet, no normalization, no residuals, no shared expert gating, no LM head).
- Token-level batching, KV-cache, generation loop.
- Async prefetch of next-token experts.
- SSD cold tier (was milestone 6 — deferred).
- A unified cache abstraction across GGUF and FP8 paths (the two `ExpertCache` flavours are kept side-by-side for now).
- VRAM caching of **dequantized** FP16 weights (the current VRAM cache holds raw bytes; the runner still dequantizes on CPU).
- Any kind of HTTP / OpenAI-compatible serving layer.

### GGUF status

The GGUF branch (inspect + map + cache + single-expert runner) is
**frozen as-is** for now. It works as a correctness probe and a
byte-level residency simulator, but the dequantize-on-CPU path makes
it unsuitable as an actual inference engine. A future revision will
fold GGUF execution into the same FP8 forward path (dequantize on the
GPU, keep FP16 weights hot in VRAM) — this is the planned direction
but no code is committed yet.

## CLI entry points

The package installs one console script:

```bash
router-inspect path/to/model.gguf     # generic GGUF inspector
```

The other tools are reachable as Python modules / launchers:

```bash
# one-expert GGUF execution (needs ggml.dll for IQ3_XXS/IQ4_XS dequant)
python -m router_ia.expert_runner path/to/model.gguf --layer 0 --expert 0

# Safetensors FP8 inspector (header only)
python -m router_ia.qwen36_probe /path/to/safetensors/dir

# FP8 single-expert runner
python -m router_ia.fp8_expert_runner /path/to/safetensors/dir --layer 0 --expert 0

# Qwen3.6 router
python -m router_ia.qwen36_router /path/to/safetensors/dir --layer 0

# Partial MoE forward probe (token id -> experts, no attention)
python -m router_ia.qwen36_moe_probe /path/to/safetensors/dir --token-id 0

# Compare runtime vs checkpoint math
python -m router_ia.qwen36_moe_validate /path/to/safetensors/dir --layer 0
```

The `qwen36_moe_probe.py` file at the repo root is a compatibility
launcher for `python qwen36_moe_probe.py ...`.

## Constraints

- Do not modify `llama.cpp`.
- Keep the first implementation small and inspectable.
- Correctness first; performance comes later.
- The two expert-cache flavours (GGUF and FP8) will be unified in a later pass.

## Roadmap

1. GPU-side dequantization of GGUF experts (planned; blocked on picking the kernel path — `torch._weight_int4pack_mm`, Triton, or a small CUDA extension).
2. Async prefetch of the next token's top-k experts while the current token decodes.
3. Unified `MoEExpertCache` over both GGUF and Safetensors FP8 formats.
4. Shared expert tier (always-hot in VRAM, much smaller than routed experts).
5. SSD cold tier (original milestone 6).
6. Decide whether to lift this into a real token loop or hand the MoE path over to `llama.cpp` / `vLLM` and keep this repo as the measurement harness.
