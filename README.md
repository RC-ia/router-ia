# router-ia

Experimental, inspectable inference runtime for **Qwen3.6-35B-A3B**, written
from scratch around PyTorch and Safetensors, with a low-memory target of
roughly **4 GB VRAM + 8 GB RAM**.

The project began as a GGUF byte-range/indexing experiment and evolved into
the **V2 stateful runtime**: a 40-layer hybrid executor, persistent attention
state, routed MoE execution, hierarchical Q4 expert residency, batched expert
loads, and optional native CUDA memory primitives.

This README describes the state of the repository on `main`, including the
V2 memory/runtime work.

## Target model

**Qwen3.6-35B-A3B**

- 40 transformer layers
- 30 Gated DeltaNet / linear-attention layers
- 10 full-attention layers
- 256 routed experts per MoE layer
- top-8 routed experts per token
- 1 shared expert per layer
- hidden size 2048
- roughly 35B total parameters
- roughly 3B active parameters per token

A single decode token can therefore route through up to **320 expert
selections** (8 experts × 40 layers).

## V2 architecture

The current runtime is organized around four cooperating pieces:

```text
Safetensors checkpoint
        │
        ├── tokenizer
        │
        ├── fixed model tensors
        │      ├── attention projections
        │      ├── norms / routing tensors
        │      └── shared expert
        │
        └── routed experts
               │
               ▼
        ┌───────────────────┐
        │ Q4 hierarchy      │
        │                   │
        │ VRAM  → RAM → SSD │
        └───────────────────┘
               │
               ▼
       batched MoE execution
               │
               ▼
       persistent attention
       ├── DeltaNet state
       ├── linear conv state
       └── full-attention KV
               │
               ▼
        final RMSNorm
               │
               ▼
          lm_head chunks
               │
               ▼
          token sampling
```

### Stateful attention

The official generator now maintains attention state across tokens.

For the 30 linear-attention layers it keeps the recurrent DeltaNet matrix
state and the rolling causal-convolution state. For the 10 full-attention
layers it keeps the K/V history required for autoregressive decoding.

The state is reset per generation turn and tracks:

- tokens seen;
- full-attention KV layers/tokens/bytes;
- DeltaNet recurrent-state memory;
- linear convolution-state memory.

`src/router_ia/qwen36_attention_cache.py` contains the stateful implementation.

## Official V2 generation path

The canonical entry point is:

```bash
python -m router_ia.qwen36_chat_batch_runner /path/to/qwen36 \
  --device cuda \
  --max-new-tokens 16
```

`qwen36_chat_batch_runner.py` is the **V2 integration entry point**. It loads
the normal chat implementation and then installs the runtime optimizations and
memory policies used by the current hot path.

The underlying generator is:

```text
src/router_ia/qwen36_chat_batch.py
```

Its autoregressive path is:

```text
prompt
  ↓
tokenizer
  ↓
stateful prefill, token by token
  ↓
40-layer hybrid runtime
  ├─ Gated DeltaNet / linear attention
  ├─ full attention + KV cache
  └─ top-8 MoE + shared expert
  ↓
final RMSNorm
  ↓
lm_head
  ↓
sampling
  ↓
next token
  ↓
repeat with persistent state
```

The older `qwen36_mini_chat.py` remains a diagnostic/stateless smoke path and
is **not** the canonical V2 generator.

## V2 memory system

The main V2 change is that routed experts are no longer treated as ordinary
large tensors.

### Q4 expert hierarchy

`qwen36_expert_q4_hierarchy_fixed.py` owns the active expert residency model:

```text
source Safetensors FP8
        │
        │ first use
        ▼
      Q4 RAM
        │
        ▼
      Q4 VRAM
        │
        └── eviction ──► Q4 RAM ──► temporary SSD
```

The default budgets in the current implementation are approximately:

| Tier | Default budget |
|---|---:|
| Generic resident VRAM | 0.75 GiB |
| Q4 expert VRAM | 1.50 GiB |
| Q4 expert RAM | 1.50 GiB |
| Generic rotating VRAM stream | 0.75 GiB |
| Q4 SSD backing | `.router_q4_cache/` |

These are configurable through environment variables and are intentionally
small enough for the project's low-memory target.

The hierarchy records hits, misses, evictions, promotions/demotions and SSD
traffic so the runtime can distinguish a warm decode from a cold one.

### GPU Q4 dequantization

`qwen36_gpu_q4.py` keeps the packed Q4 representation in host RAM and moves
the raw packed bytes/scales to CUDA before unpacking and dequantizing.

The CUDA path therefore avoids doing the full Q4 arithmetic on the CPU.

### Batched MoE execution

`qwen36_expert_batch_plan_v2.py` and the later V3 refinements replace repeated
per-expert work with a layer-level plan:

- resolve the complete top-k route first;
- deduplicate expert IDs;
- group projection transfers;
- batch expert GEMMs;
- avoid the old per-expert Python thread-pool execution path.

The V3 layer adds direct routed-expert loading and avoids synchronous
Q4→FP8 promotion inside the token hot path.

### Async scheduling

The repository also contains an asynchronous expert scheduler capable of
prefetching likely next-layer experts on a dedicated CUDA stream.

The final Q4 hierarchy deliberately owns expert materialization on the
official hot path, so speculative async materialization is disabled there.
The scheduler remains useful as a prediction/diagnostic component and its
state/benchmarks remain in the tree.

## Memory policy V2

`qwen36_memory_policy_v2.py` changes how large non-expert tensors are handled:

- layer projections use the rotating VRAM stream instead of filling the
  resident pool;
- the embedding matrix stays on the host and only the required row is read
  for each token;
- `lm_head` stays on the host and is projected in CUDA chunks instead of
  requiring the full vocabulary matrix to be resident;
- small control tensors can remain in the resident VRAM budget.

The current `lm_head` chunk size is 16,384 vocabulary rows.

This is important because the full vocabulary projection is large enough to
compete directly with the memory budget needed by the routed-expert working
set.

## Native CUDA memory prototype

The repository contains an optional native memory subsystem under
`native_memory/`.

It provides:

- fixed-size VRAM slots backed by `cudaMalloc`;
- pinned host-RAM slots backed by `cudaHostAlloc`;
- configurable CUDA streams;
- synchronous and asynchronous H2D/D2H transfers;
- LRU-style block residency metadata;
- a C ABI;
- a Python `ctypes` wrapper.

The async API adds:

```text
router_mm_acquire_async()
router_mm_wait_acquire()
router_mm_is_loading()
```

The important semantic difference is that `acquire_async()` can issue the H2D
transfer without synchronizing immediately, allowing transfer work to overlap
with compute.

Build and smoke test:

```bash
python native_memory/build_and_smoke.py
python native_memory/qwen36_native_memory_smoke.py
```

Or with CMake:

```bash
cmake -S native_memory -B native_memory/build
cmake --build native_memory/build --config Release
```

Set `ROUTER_IA_NATIVE_MEMORY_LIB` when the native library is stored outside the
default build locations.

The native memory subsystem is **optional**. The Python router remains the
owner of model routing and expert metadata.

## Native Q4 work

`native_runtime/` contains host-side experiments aimed at reducing Python
overhead in the long-term runtime.

### Residency planner

`residency_plan.cpp` models the memory budget for the 35B model and estimates
whether a warm decode can be physically supplied by the available VRAM/RAM
bandwidth.

The planner's current working assumption is that the important bottleneck is
often **data delivery**, not the raw matrix multiply throughput: the model's
full routed MoE working set cannot fit in VRAM, so the decode path has to move
only the required expert representation efficiently.

One planner scenario reports roughly **128 ms/token** for a warm Q4 decode
under its modeled assumptions. This is a planning estimate, not a measured
end-to-end V2 benchmark.

### Native FP8 → Q4 conversion

`expert_loader.cpp` is a C++ prototype for converting the checkpoint's FP8
expert matrices into the same packed Q4 representation used by the Python
runtime.

The deterministic self-test is validated bit-for-bit against the Python
reference by:

```bash
g++ -O2 -std=c++17 native_runtime/expert_loader.cpp -o expert_loader
./expert_loader selftest > native_runtime/q4_ref.txt
python native_runtime/validate_q4_against_python.py
```

The validation checks both the packed Q4 bytes and the FP16 scalar scale.

## GGUF path

The original GGUF tools are still present, but they are no longer the active
inference path.

| Component | Role |
|---|---|
| `gguf_inspect.py` | GGUF metadata/tensor inspection |
| `mapper.py` | packed expert byte-range mapping |
| `expert_cache.py` | RAM/VRAM cache over packed GGUF bytes |
| `expert_runner.py` | single-expert GGUF execution via `ggml.dll` |

This code is retained as a correctness probe and residency experiment.

It does not define the current V2 generator.

## Validation status

The repository has independent validation for the major mathematical pieces:

| Area | Status |
|---|---|
| 40-layer hybrid dispatch | **PASS** |
| 30/10 linear-vs-full attention split | **PASS** |
| MoE top-8 routing | **PASS** |
| FP8 routed expert execution | **PASS** |
| Shared expert + aggregation | **PASS** |
| Gated RMSNorm | **PASS** |
| DeltaNet recurrent update | **PASS** |
| Full-attention KV cache behavior | **PASS** |
| Hierarchical expert residency | **PASS** |
| Native memory smoke/stress paths | **PASS** |
| Native C++ Q4 vs Python reference | **PASS** |
| Stateful generator path | **IMPLEMENTED** |
| End-to-end V2 output quality/fidelity | **ACTIVE VALIDATION** |

The last line is intentionally separate from mathematical component tests:
having a complete stateful runtime does not by itself prove parity with an
independent reference implementation across arbitrary prompts and generation
lengths.

## Repository layout

```text
router-ia/
├── src/router_ia/
│   ├── qwen36_chat_batch.py
│   ├── qwen36_chat_batch_runner.py
│   ├── qwen36_attention_cache.py
│   ├── qwen36_40layer_loop.py
│   ├── qwen36_cached_loop.py
│   ├── qwen36_expert_cache.py
│   ├── qwen36_expert_q4_hierarchy_fixed.py
│   ├── qwen36_expert_batch_plan_v2.py
│   ├── qwen36_expert_batch_plan_v3.py
│   ├── qwen36_memory_policy_v2.py
│   ├── qwen36_gpu_q4.py
│   ├── qwen36_native_memory.py
│   └── ...
│
├── native_memory/
│   ├── router_memory.cu
│   ├── router_memory.h
│   ├── build_and_smoke.py
│   ├── qwen36_native_memory_smoke.py
│   └── ...
│
├── native_runtime/
│   ├── residency_plan.cpp
│   ├── expert_loader.cpp
│   └── validate_q4_against_python.py
│
└── experimental/
    └── async_memory/
```

## Installation

The package metadata currently provides the base Python package and the GGUF
dependency. The V2 runtime additionally requires the libraries imported by the
runtime modules:

```bash
pip install -e .
pip install torch safetensors transformers
```

Use the PyTorch installation appropriate for the CUDA version available on the
machine.

The model directory must be local because the tokenizer is loaded with
`local_files_only=True`.

The expected checkpoint layout is the normal Safetensors/indexed form, including
`model.safetensors.index.json`, model shards, and the tokenizer files required
by `transformers`.

## Main commands

### Canonical V2 runner

```bash
python -m router_ia.qwen36_chat_batch_runner /path/to/qwen36 \
  --device cuda \
  --max-new-tokens 16
```

### Direct stateful generator

```bash
python -m router_ia.qwen36_chat_batch /path/to/qwen36 \
  --device cuda \
  --max-new-tokens 16
```

The direct generator is useful for isolating the core generation implementation.
The runner is the canonical V2 path because it installs the active policy and
optimization modules.

### Reference 40-layer executor

```bash
python -m router_ia.qwen36_40layer_loop /path/to/qwen36 --device cuda
```

### Hierarchical cache / diagnostics

```bash
python -m router_ia.qwen36_cached_loop /path/to/qwen36 --device cuda
python -m router_ia.qwen36_lru_benchmark /path/to/qwen36
```

### Older mini chat smoke path

```bash
python -m router_ia.qwen36_mini_chat /path/to/qwen36 --device cuda
```

## Environment knobs

The current runtime exposes several environment variables for memory and
profiling experiments, including:

```text
QWEN36_Q4_VRAM_GB
QWEN36_Q4_RAM_GB
QWEN36_Q4_RAM_SLOTS_PER_LAYER
QWEN36_Q4_SSD_DIRNAME
QWEN36_Q4_RESIDENT_GB
QWEN36_Q4_SSD_LOAD_WORKERS
QWEN36_EXPERT_LOAD_WORKERS
QWEN36_PROFILE
QWEN36_WINDOWS_MMAP_GUARD
QWEN36_MEMORY_LOAD_TRACE
ROUTER_IA_NATIVE_MEMORY_LIB
```

Additional V2 optimization modules expose their own `QWEN36_*` controls; see
the corresponding module docstrings before changing production defaults.

## Constraints

- Do not modify `llama.cpp`.
- Keep the runtime inspectable and modular.
- Preserve the reference math while optimizing storage and execution.
- Keep the Q4 hierarchy as the owner of expert residency on the canonical
  hot path.
- Treat optimization measurements as hypotheses until they are reproduced by
  the project's validation/benchmark scripts.

## Roadmap

1. **Reference fidelity:** expand stateful multi-token comparisons against an
   independent Qwen3.6 reference implementation.
2. **Decode optimization:** reduce RAM/VRAM movement, improve Q4 residency,
   overlap transfers with useful compute, and reduce Python/allocator overhead.
3. **Native hot path:** progressively replace the highest-cost Python memory
   operations with the native primitives already prototyped.
4. **Kernel work:** evaluate specialized CUDA/Triton/compiled kernels only after
   the memory/data-delivery path is measured and stable.
5. **Serving/packaging:** add a stable serving interface only after the runtime
   itself is sufficiently validated.
