from __future__ import annotations

"""Qwen3.6 loop with planned expert prefetch.

Keeps the optimized Safetensors reader and FP8 dequantization, and adds a
bounded prefetch planner for routed MoE experts. After the router chooses the
8 experts of a layer, raw FP8 weights for a few upcoming experts are loaded in
background threads while the current expert is computed. Dequantization stays
on the foreground path and only the raw FP8 payloads are prefetched, keeping
memory bounded.
"""

import atexit
import json
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from . import qwen36_40layer_loop as base
from .qwen36_dequant import dequantize_fp8_blockwise

PREFETCH_WINDOW = 4


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

    def load(self, name: str, device: str = "cpu") -> torch.Tensor:
        shard_name = self.weight_map.get(name)
        shards = [self.root / shard_name] if shard_name else sorted(self.root.glob("*.safetensors"))
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
_executor = ThreadPoolExecutor(max_workers=PREFETCH_WINDOW, thread_name_prefix="qwen-prefetch")


def _store(root: Path) -> _ShardStore:
    key = root.resolve()
    store = _stores.get(key)
    if store is None:
        store = _ShardStore(key)
        _stores[key] = store
    return store


def _cached_load_tensor(root: Path, name: str, device: str = "cpu") -> torch.Tensor:
    return _store(root).load(name, device)


def _raw_expert(root: Path, layer: int, expert: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prefix = f"{base.layer_prefix(layer)}mlp.experts.{expert}"
    gw = _cached_load_tensor(root, prefix + ".gate_proj.weight", "cpu")
    gs = _cached_load_tensor(root, prefix + ".gate_proj.weight_scale_inv", "cpu")
    uw = _cached_load_tensor(root, prefix + ".up_proj.weight", "cpu")
    us = _cached_load_tensor(root, prefix + ".up_proj.weight_scale_inv", "cpu")
    dw = _cached_load_tensor(root, prefix + ".down_proj.weight", "cpu")
    ds = _cached_load_tensor(root, prefix + ".down_proj.weight_scale_inv", "cpu")
    return gw, gs, uw, us, dw, ds


def _dequant_or_float(weight: torch.Tensor, scale: torch.Tensor | None, device: str) -> torch.Tensor:
    if weight.dtype == torch.float8_e4m3fn:
        if scale is None:
            raise ValueError("FP8 expert weight is missing its scale tensor")
        return dequantize_fp8_blockwise(weight, scale).to(device)
    return weight.float().to(device)


def _run_expert_raw(raw, x: torch.Tensor, weight: float, device: str) -> torch.Tensor:
    gw, gs, uw, us, dw, ds = raw
    gate_w = _dequant_or_float(gw, gs, device)
    up_w = _dequant_or_float(uw, us, device)
    down_w = _dequant_or_float(dw, ds, device)

    gate = F.linear(x, gate_w)
    up = F.linear(x, up_w)
    hidden = F.silu(gate) * up
    out = F.linear(hidden, down_w)
    out.mul_(weight)

    del gate_w, up_w, down_w, gate, up, hidden
    del gw, gs, uw, us, dw, ds
    return out


def _load_shared_projection(root: Path, layer: int, kind: str, device: str) -> torch.Tensor:
    prefix = f"{base.layer_prefix(layer)}mlp.shared_expert.{kind}"
    weight = _cached_load_tensor(root, prefix + ".weight", "cpu")
    scale = _cached_load_tensor(root, prefix + ".weight_scale_inv", "cpu") if weight.dtype == torch.float8_e4m3fn else None
    out = _dequant_or_float(weight, scale, device)
    del weight
    if scale is not None:
        del scale
    return out


def _run_shared_expert(root: Path, layer: int, x: torch.Tensor, device: str):
    gate_w = _load_shared_projection(root, layer, "gate_proj", device)
    up_w = _load_shared_projection(root, layer, "up_proj", device)
    down_w = _load_shared_projection(root, layer, "down_proj", device)
    shared_gate_w = _cached_load_tensor(root, base.layer_prefix(layer) + "mlp.shared_expert_gate.weight", device=device).float()
    gate = torch.sigmoid(F.linear(x, shared_gate_w))
    hidden_gate = F.linear(x, gate_w)
    up = F.linear(x, up_w)
    hidden = F.silu(hidden_gate) * up
    raw = F.linear(hidden, down_w)
    out = raw * gate
    gate_value = float(gate.item())
    del gate_w, up_w, down_w, shared_gate_w, gate, hidden_gate, up, hidden, raw
    return out, gate_value


def _planned_moe_step(root: Path, layer: int, residual: torch.Tensor, top_k: int, device: str):
    post_norm = base.load_layer_weight(root, layer, "post_attention_layernorm.weight", device)
    moe_in = base.rmsnorm(residual, post_norm).reshape(1, base.HIDDEN).float()
    router_w = base.load_layer_weight(root, layer, "mlp.gate.weight", device).float()
    routed = base.route(moe_in.reshape(-1), router_w, top_k=top_k)
    expert_ids = [int(v) for v in routed.expert_ids.detach().cpu().tolist()]
    weights = [float(v) for v in routed.weights.detach().cpu().tolist()]

    futures: dict[int, Future] = {}
    next_to_submit = 0

    def submit_until_window() -> None:
        nonlocal next_to_submit
        while next_to_submit < len(expert_ids) and len(futures) < PREFETCH_WINDOW:
            expert = expert_ids[next_to_submit]
            futures[next_to_submit] = _executor.submit(_raw_expert, root, layer, expert)
            next_to_submit += 1

    submit_until_window()
    routed_sum = torch.zeros_like(moe_in)

    for idx, weight in enumerate(weights):
        raw = futures.pop(idx).result()
        submit_until_window()
        out = _run_expert_raw(raw, moe_in, weight, device)
        routed_sum.add_(out)
        del out, raw

    shared_out, shared_gate = _run_shared_expert(root, layer, moe_in, device)
    moe_out = routed_sum + shared_out
    layer_out = residual + moe_out
    moe_input_norm = float(torch.linalg.vector_norm(moe_in).item())

    del post_norm, moe_in, router_w, routed, routed_sum, shared_out, moe_out, futures
    return layer_out, expert_ids, weights, shared_gate, moe_input_norm


base.load_tensor = _cached_load_tensor
base.dequantize_fp8_blockwise = dequantize_fp8_blockwise
base.moe_step = _planned_moe_step


@atexit.register
def _close() -> None:
    _executor.shutdown(wait=True, cancel_futures=False)
    for store in _stores.values():
        store.close()


def main() -> None:
    base.main()
    print(
        f"planned reader: shards opened={sum(s.handle_opens for s in _stores.values())} | "
        f"cached handle hits={sum(s.handle_hits for s in _stores.values())} | "
        f"moe prefetch window={PREFETCH_WINDOW}"
    )


if __name__ == "__main__":
    main()
