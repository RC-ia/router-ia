# router-ia V2 runtime

Curated runtime-only copy of the code path used by the current Qwen3.6 model generator.

## Included

The V2 package contains only the generator and its runtime dependency chain:

- end-to-end chat/generation entry point;
- fused CUDA expert path;
- hierarchical RAM/VRAM cache;
- stateful DeltaNet and full-attention KV state;
- routed expert cache;
- FP8 blockwise dequantization;
- router selection;
- model tensor loading and runtime math helpers.

The original tree is untouched. V2 lives under `v2/src/router_ia/`.

## Run

From the repository root:

```bash
PYTHONPATH=v2/src python -m router_ia.qwen36_chat_batch /path/to/model \
  --device cuda --max-new-tokens 16
```

The model directory must contain the Qwen3.6 Safetensors checkpoint and its index.

## Excluded

V2 intentionally does not copy standalone probes, benchmarks, profilers, experimental proposals, native-memory experiments, old Q4 hierarchy experiments, or other files outside the inference dependency path.

This V2 is a curated structural snapshot of the existing runtime; it does not change the model architecture or inference behavior.

## Hybrid RAM/VRAM expert execution

When a routed expert is resident as compressed Q4 in system RAM, V2 can execute that expert on the CPU instead of promoting its matrices to VRAM. Only the small activation vector and final 2048-value result cross between CPU and GPU.

The active policy is:

- FP8 expert in VRAM -> GPU execution.
- Q4 expert in RAM -> CPU execution.
- cold expert not cached -> existing GPU load path, then it becomes an FP8 VRAM entry.

The CPU worker count is controlled by `QWEN36_CPU_EXPERT_WORKERS` and defaults to `2`.
## Expert transition prefetch

V2 also learns transitions between routed experts across consecutive tokens. For each layer, the observed expert set becomes the context for predicting the next likely experts.

The transition cache uses three states:

1. **VRAM / FP8:** actively used expert.
2. **RAM / predicted FP8:** speculative next expert. It is loaded into host RAM first and does not consume VRAM.
3. **RAM / Q4:** cold expert retained for CPU execution.

When a predicted FP8 expert is actually selected by the router, it is promoted from RAM to the VRAM expert cache. If the prediction is wrong, it stays only in the bounded predictive RAM window until eviction.

The predictive window is four experts per layer. This keeps the speculative RAM footprint bounded.

## RAM budget

The general runtime RAM cache has a fixed 8 GiB ceiling by default and evicts lower-priority cached tensors when full. Override it with `QWEN36_CACHE_GB`, for example `QWEN36_CACHE_GB=6` for a 6 GiB ceiling.