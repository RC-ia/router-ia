from __future__ import annotations

"""Qwen3.6 loop with optimized Safetensors I/O, FP8 dequantization and MoE.

This wrapper keeps the reference math in qwen36_40layer_loop.py unchanged while
replacing the tensor-loading backend, FP8 dequantizer and routed-MoE executor.
Safetensors shards are opened lazily and kept open for the lifetime of the run.
FP8 scales are broadcast over quantization blocks without materializing a
full-size scale matrix. Routed experts are processed in micro-batches of four
experts to reduce small GEMM/Python-call overhead while keeping RAM bounded.
"""

import atexit
import gc
import json
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from . import qwen36_40layer_loop as base
from .qwen36_dequant import dequantize_fp8_blockwise

MOE_EXPERT_BATCH = 4


class _ShardStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.stack = ExitStack()
        self.weight_map: dict[str, str] = {}
        self.handles: dict[Path, object] = {}
        self.handle_opens = 0
        self.handle_hits = 0

        index_path = self.root / "model.safetensors.index.json"
        if index_path.is_file():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            self.weight_map = dict(payload.get("weight_map", {}))

    def _handle(self, shard: Path):
        handle = self.handles.get(shard)
        if handle is not None:
            self.handle_hits += 1
            return handle

        handle = self.stack.enter_context(
            safe_open(str(shard), framework="pt", device="cpu")
        )
        self.handles[shard] = handle
        self.handle_opens += 1
        return handle

    def load(self, name: str, device: str):
        shard_name = self.weight_map.get(name)
        if shard_name:
            shards = [self.root / shard_name]
        else:
            shards = sorted(self.root.glob("*.safetensors"))

        for shard in shards:
            if not shard.is_file():
                continue
            handle = self._handle(shard)
            if name in handle.keys():
                return handle.get_tensor(name).to(device=device)

        raise KeyError(f"Tensor not found: {name}")

    def close(self) -> None:
        self.stack.close()
        self.handles.clear()


_stores: dict[Path, _ShardStore] = {}


def _store(root: Path) -> _ShardStore:
    key = root.resolve()
    store = _stores.get(key)
    if store is None:
        store = _ShardStore(key)
        _stores[key] = store
    return store


def _cached_load_tensor(root: Path, name: str, device: str = "cpu"):
    return _store(root).load(name, device)


def _load_moe_projection(root: Path, layer: int, expert: int, kind: str, device: str) -> torch.Tensor:
    prefix = f"{base.layer_prefix(layer)}mlp.experts.{expert}.{kind}"
    weight = base.load_tensor(root, prefix + ".weight", device="cpu")
    if weight.dtype == torch.float8_e4m3fn:
        scale = base.load_tensor(root, prefix + ".weight_scale_inv", device="cpu")
        out = base.dequantize_fp8_blockwise(weight, scale).to(device)
        del scale
    else:
        out = weight.float().to(device)
    del weight
    return out


def _load_shared_projection(root: Path, layer: int, kind: str, device: str) -> torch.Tensor:
    prefix = f"{base.layer_prefix(layer)}mlp.shared_expert.{kind}"
    weight = base.load_tensor(root, prefix + ".weight", device="cpu")
    if weight.dtype == torch.float8_e4m3fn:
        scale = base.load_tensor(root, prefix + ".weight_scale_inv", device="cpu")
        out = base.dequantize_fp8_blockwise(weight, scale).to(device)
        del scale
    else:
        out = weight.float().to(device)
    del weight
    return out


def _run_expert_batch(
    root: Path,
    layer: int,
    expert_ids: list[int],
    weights: list[float],
    x: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Run up to four routed experts as one CPU/GPU micro-batch.

    Each expert still lives only for the duration of this micro-batch. Three
    FP32 matrices per expert are held simultaneously, about 12 MiB/expert for
    the current Qwen3.6 shapes, so four experts are about 48 MiB of weights.
    """
    gate_w = torch.stack(
        [_load_moe_projection(root, layer, expert, "gate_proj", device) for expert in expert_ids], dim=0
    )
    up_w = torch.stack(
        [_load_moe_projection(root, layer, expert, "up_proj", device) for expert in expert_ids], dim=0
    )
    down_w = torch.stack(
        [_load_moe_projection(root, layer, expert, "down_proj", device) for expert in expert_ids], dim=0
    )

    batch = len(expert_ids)
    x_batch = x.expand(batch, -1).unsqueeze(1)
    gate = torch.bmm(x_batch, gate_w.transpose(1, 2)).squeeze(1)
    up = torch.bmm(x_batch, up_w.transpose(1, 2)).squeeze(1)
    hidden = F.silu(gate) * up
    expert_out = torch.bmm(hidden.unsqueeze(1), down_w.transpose(1, 2)).squeeze(1)

    weight_tensor = torch.tensor(weights, dtype=expert_out.dtype, device=expert_out.device).unsqueeze(1)
    combined = (expert_out * weight_tensor).sum(dim=0, keepdim=True)

    del gate_w, up_w, down_w, x_batch, gate, up, hidden, expert_out, weight_tensor
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return combined


def _run_shared_expert(root: Path, layer: int, x: torch.Tensor, device: str) -> tuple[torch.Tensor, float]:
    gate_w = _load_shared_projection(root, layer, "gate_proj", device)
    up_w = _load_shared_projection(root, layer, "up_proj", device)
    down_w = _load_shared_projection(root, layer, "down_proj", device)
    shared_gate_w = base.load_tensor(
        root, base.layer_prefix(layer) + "mlp.shared_expert_gate.weight", device=device
    ).float()

    gate = torch.sigmoid(F.linear(x, shared_gate_w))
    hidden_gate = F.linear(x, gate_w)
    up = F.linear(x, up_w)
    hidden = F.silu(hidden_gate) * up
    raw = F.linear(hidden, down_w)
    out = raw * gate
    gate_value = float(gate.item())

    del gate_w, up_w, down_w, shared_gate_w, gate, hidden_gate, up, hidden, raw
    return out, gate_value


def _batched_moe_step(
    root: Path,
    layer: int,
    residual: torch.Tensor,
    top_k: int,
    device: str,
) -> tuple[torch.Tensor, list[int], list[float], float, float]:
    post_norm = base.load_layer_weight(root, layer, "post_attention_layernorm.weight", device)
    moe_in = base.rmsnorm(residual, post_norm).reshape(1, base.HIDDEN).float()
    router_w = base.load_layer_weight(root, layer, "mlp.gate.weight", device).float()
    routed = base.route(moe_in.reshape(-1), router_w, top_k=top_k)
    expert_ids = [int(v) for v in routed.expert_ids.detach().cpu().tolist()]
    weights = [float(v) for v in routed.weights.detach().cpu().tolist()]

    routed_sum = torch.zeros_like(moe_in)
    for start in range(0, len(expert_ids), MOE_EXPERT_BATCH):
        ids = expert_ids[start : start + MOE_EXPERT_BATCH]
        ws = weights[start : start + MOE_EXPERT_BATCH]
        out = _run_expert_batch(root, layer, ids, ws, moe_in, device)
        routed_sum.add_(out)
        del out, ids, ws

    shared_out, shared_gate = _run_shared_expert(root, layer, moe_in, device)
    moe_out = routed_sum + shared_out
    layer_out = residual + moe_out
    moe_input_norm = float(torch.linalg.vector_norm(moe_in).item())

    del post_norm, moe_in, router_w, routed, routed_sum, shared_out, moe_out
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return layer_out, expert_ids, weights, shared_gate, moe_input_norm


# Patch only the loader, dequantizer and routed-MoE implementation. Attention
# and all other reference math remain in qwen36_40layer_loop.py.
base.load_tensor = _cached_load_tensor
base.dequantize_fp8_blockwise = dequantize_fp8_blockwise
base.moe_step = _batched_moe_step


@atexit.register
def _close_stores() -> None:
    for store in _stores.values():
        store.close()


def main() -> None:
    base.main()

    for root, store in _stores.items():
        print(
            f"cached reader: root={root} | "
            f"shards opened={store.handle_opens} | "
            f"cached handle hits={store.handle_hits} | "
            f"moe expert batch={MOE_EXPERT_BATCH}"
        )


if __name__ == "__main__":
    main()
