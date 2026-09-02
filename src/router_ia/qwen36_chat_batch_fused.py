from __future__ import annotations

"""Qwen3.6 chat runner with fused routed-expert FP8 staging.

The normal chat runner asks for gate/up/down independently. This wrapper
replaces the expert streaming method so the first projection request for an
expert loads all three matrices together, performs one H2D copy per stacked
weight/scale tensor, and dequantizes the triplet in one vectorized GPU pass.
The returned matrices are split into the existing stream cache, so the rest of
the application remains unchanged.
"""

from pathlib import Path

import torch

from . import qwen36_cached_loop as cached
from . import qwen36_dequant as dequant
from . import qwen36_chat_batch as chat


_ORIGINAL_STREAM_PROJECTION = cached._ShardStore.stream_projection


def _expert_triplet_prefix(prefix: str) -> tuple[str, str] | None:
    for suffix in (".gate_proj", ".up_proj", ".down_proj"):
        if prefix.endswith(suffix):
            expert_prefix = prefix[: -len(suffix)]
            return expert_prefix, suffix[1:]
    return None


def _fused_stream_projection(self, prefix: str) -> torch.Tensor:
    """Stage gate/up/down together for one routed expert."""
    if self.target_device != "cuda":
        return _ORIGINAL_STREAM_PROJECTION(self, prefix)

    parsed = _expert_triplet_prefix(prefix)
    if parsed is None or ".mlp.experts." not in prefix:
        return _ORIGINAL_STREAM_PROJECTION(self, prefix)

    expert_prefix, requested = parsed
    names = (
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    # Fast cache path: every requested projection is already in the stream.
    requested_key = prefix + ".__stream__"
    cached_requested = self.vram_cache.get_stream(requested_key)
    if cached_requested is not None:
        return cached_requested

    keys = [expert_prefix + "." + name + ".__stream__" for name in names]
    existing = {key: self.vram_cache.get_stream(key) for key in keys}
    if all(tensor is not None for tensor in existing.values()):
        return existing[requested_key]

    # Load the raw FP8 weights and scales from the RAM cache first. This also
    # keeps SSD accesses out of the GPU staging critical path.
    raw_weights = []
    raw_scales = []
    for name in names:
        proj = expert_prefix + "." + name
        raw_weights.append(self.load(proj + ".weight", device="cpu"))
        raw_scales.append(self.load(proj + ".weight_scale_inv", device="cpu"))

    if not all(weight.dtype == torch.float8_e4m3fn for weight in raw_weights):
        # Preserve correctness for non-FP8 tensors.
        outputs = [weight.to(device="cuda", dtype=torch.float16) for weight in raw_weights]
    else:
        # One contiguous H2D transfer for the three weights and one for scales,
        # followed by one batched FP8 -> FP16 dequantization kernel sequence.
        weight_batch = torch.stack(raw_weights, dim=0).to(device="cuda")
        scale_batch = torch.stack(raw_scales, dim=0).to(device="cuda")
        output_batch = dequant.dequantize_fp8_blockwise_batch(weight_batch, scale_batch)
        output_batch = output_batch.to(dtype=torch.float16)
        outputs = list(output_batch.unbind(dim=0))
        del weight_batch, scale_batch, output_batch

    for name, tensor in zip(names, outputs):
        key = expert_prefix + "." + name + ".__stream__"
        # Each complete expert triplet is ~6 MiB in FP16, so the existing
        # 600 MiB stream budget can retain many experts across layers.
        self.vram_cache.put_stream(key, tensor)

    requested_key = prefix + ".__stream__"
    result = self.vram_cache.get_stream(requested_key)
    if result is None:
        raise RuntimeError(f"Fused expert staging failed for {prefix}")
    return result


cached._ShardStore.stream_projection = _fused_stream_projection


def main() -> None:
    print("fused_expert_staging=gate+up+down")
    print("fused_dequant=3-matrix-batch")
    print("fused_h2d=stacked-weight-and-scale")
    print("reduction=24->8 routed-expert-dequant-passes-per-layer")
    chat.main()


if __name__ == "__main__":
    main()
