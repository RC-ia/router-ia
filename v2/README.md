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

Install the V2 package from the repository root:

```bash
python -m pip install ./v2
```

Then use its installed command:

```bash
router-ia-v2 /path/to/model \
  --device cuda --max-new-tokens 16
```

The model directory must contain the Qwen3.6 Safetensors checkpoint and its index.

### CUDA environment (optional)

CUDA is optional: the runtime defaults to CPU execution when `--device` is not
specified. To run with `--device cuda`, install a CUDA-enabled PyTorch build
that matches the NVIDIA driver and CUDA environment on the host. PyTorch CUDA
wheels are selected separately according to the platform; CUDA is not an
implicit dependency of this package.

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
## Expert cross-layer prefetch

V2 predicts the experts for the current layer from experts selected in earlier layers of the same token. Nearby layers receive more weight, while all earlier layers can contribute evidence.

The predictor uses three states:

1. **VRAM / FP8:** actively used expert.
2. **RAM / predicted FP8:** speculative next expert. It is loaded into host RAM first and does not consume VRAM.
3. **RAM / Q4:** cold expert retained for CPU execution.

Predicted experts are prefetched into host RAM before the current layer resolves its routed experts, allowing the RAM load to overlap the router and SSD path. When a predicted FP8 expert is selected, it can be reused from RAM and promoted into the VRAM expert cache.

The predictive window is two experts per layer by default.

### Persistent router learning

The cross-layer predictor persists its learned transition counts between runs. By default the state is written to a model-specific JSON file in the current working directory, using the checkpoint index hash so different model checkpoints do not share learned routing data.

Set `QWEN36_ROUTER_STATE` to choose another persistent location. The learned state is loaded when the model is first accessed and saved after each chat turn and again at process exit.

The state keeps only the strongest observations for each routing context, controlled by `QWEN36_ROUTER_STATE_CONTEXT_LIMIT` (default 16), so the router remains compact.

## RAM budget

The general runtime RAM cache has a fixed 8 GiB ceiling by default and evicts lower-priority cached tensors when full. Override it with `QWEN36_CACHE_GB`, for example `QWEN36_CACHE_GB=6` for a 6 GiB ceiling.
