from __future__ import annotations

"""Route routed-expert tensors around the generic RAM cache.

Experts have their own FP8/Q4 cache. A routed expert tensor requested from the
expert path must therefore not be inserted into the generic 5 GiB RAM tensor
cache, otherwise the two cache systems duplicate the same workload.

This patch only changes cache placement. Model math is untouched.
"""

from . import qwen36_cached_loop as cached

_ORIGINAL_LOAD = cached._ShardStore.load


def _expert_aware_load(self, name: str, device: str):
    if not cached._is_expert_tensor(name):
        return _ORIGINAL_LOAD(self, name, device)

    if device == "cuda":
        self.target_device = "cuda"
        cached_tensor = self.vram_cache.get(name)
        if cached_tensor is not None:
            return cached_tensor

    # Cold routed experts bypass the generic RAM cache. The caller owns the
    # expert-specific FP8/Q4 retention decision; this tensor is only a host
    # staging buffer on the way from the safetensor shard to the GPU.
    tensor = self._load_ssd(name)
    if device == "cuda":
        return tensor.to(device="cuda")
    return tensor


cached._ShardStore.load = _expert_aware_load
print("expert_tier_policy=ram-bypass|expert-cache-owned")
