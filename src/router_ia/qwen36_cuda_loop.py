from __future__ import annotations

"""CUDA-first single-token Qwen3.6 reference runner.

The existing qwen36_40layer_loop.py remains the CPU/reference implementation.
This variant keeps FP8 weights compact while transferring them to the target
GPU, then performs the blockwise dequantization on the target device. It also
supports a conservative LRU cache for already-used MoE/shared-expert weights
so repeated tokens can reuse GPU-resident weights without prefetching whole
layers.

Full-attention KV cache is still intentionally limited to position 0.
"""

import argparse
import gc
from collections import OrderedDict
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from .qwen36_gated_norm_probe import gated_rmsnorm
from .qwen36_op_probe import (
    BLOCK,
    HEAD_DIM,
    dequantize_fp8_blockwise,
    load_embedding_row,
    load_tensor,
    rmsnorm,
)
from .qwen36_router import route

HIDDEN = 2048
LINEAR_NUM_K_HEADS = 16
LINEAR_NUM_V_HEADS = 32
LINEAR_KEY_DIM = LINEAR_NUM_K_HEADS * 128
LINEAR_VALUE_DIM = LINEAR_NUM_V_HEADS * 128
FULL_NUM_HEADS = 16
FULL_NUM_KV_HEADS = 2
FULL_HEAD_DIM = 256
FULL_Q_DIM = FULL_NUM_HEADS * FULL_HEAD_DIM
FULL_NUM_KV_GROUPS = FULL_NUM_HEADS // FULL_NUM_KV_HEADS
EPS = 1e-6
DEFAULT_LAYERS = 40
DEFAULT_CACHE_MIB = 512.0


class GPUWeightCache:
    """Conservative LRU cache for dequantized MoE weights on the GPU."""

    def __init__(self, budget_mib: float) -> None:
        if budget_mib < 0:
            raise ValueError("cache budget must be non-negative")
        self.budget_bytes = int(budget_mib * 1024**2)
        self.used_bytes = 0
        self.items: OrderedDict[str, tuple[torch.Tensor, int]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    @staticmethod
    def nbytes(tensor: torch.Tensor) -> int:
        return int(tensor.numel() * tensor.element_size())

    def get(self, key: str) -> torch.Tensor | None:
        item = self.items.get(key)
        if item is None:
            self.misses += 1
            return None
        tensor, _ = item
        self.items.move_to_end(key)
        self.hits += 1
        return tensor

    def put(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        size = self.nbytes(tensor)
        if self.budget_bytes == 0 or size > self.budget_bytes:
            return tensor

        old = self.items.pop(key, None)
        if old is not None:
            self.used_bytes -= old[1]

        while self.items and self.used_bytes + size > self.budget_bytes:
            _, (evicted, evicted_size) = self.items.popitem(last=False)
            self.used_bytes -= evicted_size
            self.evictions += 1
            del evicted

        self.items[key] = (tensor, size)
        self.used_bytes += size
        return tensor

    def clear(self) -> None:
        self.items.clear()
        self.used_bytes = 0


_GPU_CACHE: GPUWeightCache | None = None


def set_gpu_cache(cache: GPUWeightCache | None) -> None:
    global _GPU_CACHE
    _GPU_CACHE = cache


def should_cache_projection(prefix: str) -> bool:
    return ".mlp.experts." in prefix or ".mlp.shared_expert." in prefix


def layer_prefix(layer: int) -> str:
    return f"model.language_model.layers.{layer}."


def load_layer_weight(root: Path, layer: int, suffix: str, device: str) -> torch.Tensor:
    return load_tensor(root, layer_prefix(layer) + suffix, device=device)


def load_optional_tensor(root: Path, name: str, device: str) -> torch.Tensor | None:
    try:
        return load_tensor(root, name, device=device)
    except KeyError:
        return None


def attention_type(root: Path, layer: int) -> str:
    prefix = layer_prefix(layer)
    linear = load_optional_tensor(root, prefix + "linear_attn.in_proj_qkv.weight", "cpu")
    if linear is not None:
        del linear
        return "linear_attention"
    full = load_optional_tensor(root, prefix + "self_attn.q_proj.weight", "cpu")
    if full is not None:
        del full
        return "full_attention"
    raise KeyError(f"Could not identify attention type for layer {layer}")


def dequantize_fp8_target(weight: torch.Tensor, scale_inv: torch.Tensor, device: str) -> torch.Tensor:
    if device != "cuda":
        return dequantize_fp8_blockwise(weight, scale_inv)

    w = weight.to(device=device, non_blocking=True)
    s = scale_inv.to(device=device, non_blocking=True).float()
    expanded = s.repeat_interleave(BLOCK, dim=0).repeat_interleave(BLOCK, dim=1)
    out = w.float() * expanded[: w.shape[0], : w.shape[1]]
    del w, s, expanded
    return out


def load_fp8_projection_target(root: Path, prefix: str, device: str) -> torch.Tensor:
    cache_key = prefix if should_cache_projection(prefix) and device == "cuda" else None
    if cache_key is not None and _GPU_CACHE is not None:
        cached = _GPU_CACHE.get(cache_key)
        if cached is not None:
            return cached

    weight = load_tensor(root, prefix + ".weight", device="cpu")
    if weight.dtype == torch.float8_e4m3fn:
        scale = load_tensor(root, prefix + ".weight_scale_inv", device="cpu")
        out = dequantize_fp8_target(weight, scale, device)
        del scale
    else:
        out = weight.float().to(device)
    del weight

    if cache_key is not None and _GPU_CACHE is not None:
        out = _GPU_CACHE.put(cache_key, out)
    return out


def load_moe_projection_target(root: Path, layer: int, expert: int, kind: str, device: str) -> torch.Tensor:
    prefix = f"{layer_prefix(layer)}mlp.experts.{expert}.{kind}"
    return load_fp8_projection_target(root, prefix, device)


def load_shared_projection_target(root: Path, layer: int, kind: str, device: str) -> torch.Tensor:
    prefix = f"{layer_prefix(layer)}mlp.shared_expert.{kind}"
    return load_fp8_projection_target(root, prefix, device)


def linear_attention_step(root: Path, layer: int, x0: torch.Tensor, device: str) -> torch.Tensor:
    prefix = layer_prefix(layer)
    input_norm = load_layer_weight(root, layer, "input_layernorm.weight", device)
    h = rmsnorm(x0, input_norm)

    qkv_w = load_tensor(root, prefix + "linear_attn.in_proj_qkv.weight", device="cpu")
    qkv_scale = load_tensor(root, prefix + "linear_attn.in_proj_qkv.weight_scale_inv", device="cpu")
    qkv_w = dequantize_fp8_target(qkv_w, qkv_scale, device)
    mixed = F.linear(h.float(), qkv_w.float()).reshape(1, LINEAR_KEY_DIM * 2 + LINEAR_VALUE_DIM)

    conv_w = load_layer_weight(root, layer, "linear_attn.conv1d.weight", device).float()
    mixed = F.silu(mixed * conv_w[:, 0, -1].reshape(1, -1))
    q, k, v = torch.split(mixed, [LINEAR_KEY_DIM, LINEAR_KEY_DIM, LINEAR_VALUE_DIM], dim=-1)
    q = q.reshape(1, LINEAR_NUM_K_HEADS, 128).repeat_interleave(2, dim=1)
    k = k.reshape(1, LINEAR_NUM_K_HEADS, 128).repeat_interleave(2, dim=1)
    v = v.reshape(1, LINEAR_NUM_V_HEADS, 128)

    a_w = load_fp8_projection_target(root, prefix + "linear_attn.in_proj_a", device)
    b_w = load_fp8_projection_target(root, prefix + "linear_attn.in_proj_b", device)
    a_log = load_layer_weight(root, layer, "linear_attn.A_log", device).float().reshape(1, LINEAR_NUM_V_HEADS)
    dt_bias = load_layer_weight(root, layer, "linear_attn.dt_bias", device).float().reshape(1, LINEAR_NUM_V_HEADS)
    a_raw = F.linear(h.float(), a_w.float()).reshape(1, LINEAR_NUM_V_HEADS)
    b_raw = F.linear(h.float(), b_w.float()).reshape(1, LINEAR_NUM_V_HEADS)
    beta = torch.sigmoid(b_raw)
    g = -torch.exp(a_log) * F.softplus(a_raw + dt_bias)
    decay = torch.exp(g)

    qn = F.normalize(q.float(), dim=-1, eps=EPS) * (128 ** -0.5)
    kn = F.normalize(k.float(), dim=-1, eps=EPS)
    state = torch.zeros(1, LINEAR_NUM_V_HEADS, 128, 128, device=device, dtype=torch.float32)
    state = state * decay.unsqueeze(-1).unsqueeze(-1)
    retrieved = torch.einsum("bhkd,bhk->bhd", state, kn)
    delta = (v.float() - retrieved) * beta.unsqueeze(-1)
    state = state + kn.unsqueeze(-1) * delta.unsqueeze(-2)
    attn = torch.einsum("bhkd,bhk->bhd", state, qn)

    z_w = load_fp8_projection_target(root, prefix + "linear_attn.in_proj_z", device)
    z = F.linear(h.float(), z_w.float()).reshape(1, LINEAR_NUM_V_HEADS, 128)
    norm_w = load_layer_weight(root, layer, "linear_attn.norm.weight", device)
    gated, _, _ = gated_rmsnorm(attn, z, norm_w)

    out_w = load_fp8_projection_target(root, prefix + "linear_attn.out_proj", device)
    attn_projected = F.linear(gated.reshape(1, LINEAR_VALUE_DIM), out_w.float())
    residual = x0.reshape(1, HIDDEN) + attn_projected

    del input_norm, h, qkv_w, qkv_scale, mixed, conv_w, q, k, v
    del a_w, b_w, a_log, dt_bias, a_raw, b_raw, beta, g, decay, qn, kn
    del state, retrieved, delta, attn, z_w, z, norm_w, gated, out_w, attn_projected
    return residual


def full_attention_step(root: Path, layer: int, x0: torch.Tensor, device: str) -> torch.Tensor:
    prefix = layer_prefix(layer)
    input_norm = load_layer_weight(root, layer, "input_layernorm.weight", device)
    h = rmsnorm(x0, input_norm)

    q_w = load_tensor(root, prefix + "self_attn.q_proj.weight", device="cpu")
    q_scale = load_tensor(root, prefix + "self_attn.q_proj.weight_scale_inv", device="cpu")
    q_w = dequantize_fp8_target(q_w, q_scale, device)
    k_w = load_tensor(root, prefix + "self_attn.k_proj.weight", device="cpu")
    k_scale = load_tensor(root, prefix + "self_attn.k_proj.weight_scale_inv", device="cpu")
    k_w = dequantize_fp8_target(k_w, k_scale, device)
    v_w = load_tensor(root, prefix + "self_attn.v_proj.weight", device="cpu")
    v_scale = load_tensor(root, prefix + "self_attn.v_proj.weight_scale_inv", device="cpu")
    v_w = dequantize_fp8_target(v_w, v_scale, device)

    q_gate = F.linear(h.float(), q_w.float()).reshape(1, FULL_NUM_HEADS, FULL_HEAD_DIM * 2)
    q, gate = torch.chunk(q_gate, 2, dim=-1)
    k = F.linear(h.float(), k_w.float()).reshape(1, FULL_NUM_KV_HEADS, FULL_HEAD_DIM)
    v = F.linear(h.float(), v_w.float()).reshape(1, FULL_NUM_KV_HEADS, FULL_HEAD_DIM)

    q_norm_w = load_layer_weight(root, layer, "self_attn.q_norm.weight", device)
    k_norm_w = load_layer_weight(root, layer, "self_attn.k_norm.weight", device)
    q = rmsnorm(q, q_norm_w).float()
    k = rmsnorm(k, k_norm_w).float()
    k = k.repeat_interleave(FULL_NUM_KV_GROUPS, dim=1)
    v = v.repeat_interleave(FULL_NUM_KV_GROUPS, dim=1)

    scores = torch.matmul(q.unsqueeze(2), k.transpose(-1, -2)).squeeze(-2) * (FULL_HEAD_DIM ** -0.5)
    attn_weights = torch.softmax(scores.float(), dim=-1)
    attn = torch.matmul(attn_weights.unsqueeze(-2), v).squeeze(-2)
    attn = attn * torch.sigmoid(gate)
    attn_flat = attn.reshape(1, FULL_Q_DIM)

    out_w = load_tensor(root, prefix + "self_attn.o_proj.weight", device="cpu")
    out_scale = load_tensor(root, prefix + "self_attn.o_proj.weight_scale_inv", device="cpu")
    out_w = dequantize_fp8_target(out_w, out_scale, device)
    attn_projected = F.linear(attn_flat, out_w.float())
    residual = x0.reshape(1, HIDDEN) + attn_projected

    del input_norm, h, q_w, q_scale, k_w, k_scale, v_w, v_scale
    del q_gate, q, gate, k, v, q_norm_w, k_norm_w, scores, attn_weights, attn
    del attn_flat, out_w, out_scale, attn_projected
    return residual


def run_routed_expert(root: Path, layer: int, expert: int, x: torch.Tensor, device: str) -> torch.Tensor:
    gate_w = load_moe_projection_target(root, layer, expert, "gate_proj", device)
    up_w = load_moe_projection_target(root, layer, expert, "up_proj", device)
    down_w = load_moe_projection_target(root, layer, expert, "down_proj", device)
    gate = F.linear(x, gate_w.float())
    up = F.linear(x, up_w.float())
    hidden = F.silu(gate) * up
    out = F.linear(hidden, down_w.float())
    del gate_w, up_w, down_w, gate, up, hidden
    return out


def run_shared_expert(root: Path, layer: int, x: torch.Tensor, device: str) -> tuple[torch.Tensor, float]:
    gate_w = load_shared_projection_target(root, layer, "gate_proj", device)
    up_w = load_shared_projection_target(root, layer, "up_proj", device)
    down_w = load_shared_projection_target(root, layer, "down_proj", device)
    shared_gate_w = load_layer_weight(root, layer, "mlp.shared_expert_gate.weight", device).float()

    gate = torch.sigmoid(F.linear(x, shared_gate_w))
    hidden_gate = F.linear(x, gate_w.float())
    up = F.linear(x, up_w.float())
    hidden = F.silu(hidden_gate) * up
    raw = F.linear(hidden, down_w.float())
    out = raw * gate
    gate_value = float(gate.item())
    del gate_w, up_w, down_w, shared_gate_w, gate, hidden_gate, up, hidden, raw
    return out, gate_value


def moe_step(root: Path, layer: int, residual: torch.Tensor, top_k: int, device: str) -> tuple[torch.Tensor, list[int], list[float], float, float]:
    post_norm = load_layer_weight(root, layer, "post_attention_layernorm.weight", device)
    moe_in = rmsnorm(residual, post_norm).reshape(1, HIDDEN).float()
    router_w = load_layer_weight(root, layer, "mlp.gate.weight", device).float()
    routed = route(moe_in.reshape(-1), router_w, top_k=top_k)
    expert_ids = [int(v) for v in routed.expert_ids.detach().cpu().tolist()]
    weights = [float(v) for v in routed.weights.detach().cpu().tolist()]

    routed_sum = torch.zeros_like(moe_in)
    for expert_id, weight in zip(expert_ids, weights):
        out = run_routed_expert(root, layer, expert_id, moe_in, device)
        routed_sum.add_(out, alpha=weight)
        del out

    shared_out, shared_gate = run_shared_expert(root, layer, moe_in, device)
    moe_out = routed_sum + shared_out
    layer_out = residual + moe_out
    moe_input_norm = float(torch.linalg.vector_norm(moe_in).item())

    del post_norm, moe_in, router_w, routed, routed_sum, shared_out, moe_out
    gc.collect()
    return layer_out, expert_ids, weights, shared_gate, moe_input_norm


def main() -> None:
    parser = argparse.ArgumentParser(description="CUDA-first Qwen3.6 single-token loop")
    parser.add_argument("root", type=Path)
    parser.add_argument("--token-id", type=int, default=0)
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--end-layer", type=int, default=DEFAULT_LAYERS - 1)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--cache-mib",
        type=float,
        default=DEFAULT_CACHE_MIB,
        help="GPU LRU cache budget for used MoE/shared-expert weights; 0 disables it",
    )
    if False:
        parser.add_argument("--unused", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    if not 0 <= args.start_layer <= args.end_layer < DEFAULT_LAYERS:
        raise SystemExit(f"layer range must be inside 0..{DEFAULT_LAYERS - 1}")
    if args.cache_mib < 0:
        raise SystemExit("cache-mib must be non-negative")

    root = args.root.resolve()
    device = "cuda"
    cache = GPUWeightCache(args.cache_mib)
    set_gpu_cache(cache)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch CUDA: {torch.version.cuda}")
    print(f"GPU cache budget: {args.cache_mib:.1f} MiB")

    x = load_embedding_row(root, args.token_id).reshape(1, HIDDEN).to(device).float()
    torch.cuda.synchronize()
    start_total = perf_counter()

    print("op=cuda_loop")
    print(f"token id: {args.token_id}")
    print(f"layers: {args.start_layer}..{args.end_layer}")
    print(f"input norm: {torch.linalg.vector_norm(x).item():.8f}")

    for layer in range(args.start_layer, args.end_layer + 1):
        start_layer = perf_counter()
        kind = attention_type(root, layer)
        if kind == "linear_attention":
            residual = linear_attention_step(root, layer, x, device)
        else:
            residual = full_attention_step(root, layer, x, device)
        x, expert_ids, weights, shared_gate, moe_input_norm = moe_step(
            root, layer, residual, args.top_k, device
        )
        torch.cuda.synchronize()
        layer_ms = (perf_counter() - start_layer) * 1000.0

        used_bytes, entries, hits, misses = cache.stats()
        print(f"layer {layer} ({kind}):")
        print(f"  router top-{args.top_k}: {expert_ids}")
        print(f"  router weights: {[round(v, 8) for v in weights]}")
        print(f"  shared gate: {shared_gate:.8f}")
        print(f"  moe input norm: {moe_input_norm:.8f}")
        print(f"  output shape: {tuple(x.shape)}")
        print(f"  output norm: {torch.linalg.vector_norm(x).item():.8f}")
        print(f"  output mean: {x.mean().item():.8f}")
        print(f"  VRAM allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MiB")
        print(f"  VRAM reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MiB")
        print(f"  cache: {used_bytes / 1024**2:.1f} MiB, entries={entries}, hits={hits}, misses={misses}, evictions={cache.evictions}")
        print(f"  time: {layer_ms:.3f} ms")

        del residual
        gc.collect()
        torch.cuda.empty_cache()

    torch.cuda.synchronize()
    total_ms = (perf_counter() - start_total) * 1000.0
    print(f"final output shape: {tuple(x.shape)}")
    print(f"final output norm: {torch.linalg.vector_norm(x).item():.8f}")
    print(f"final output mean: {x.mean().item():.8f}")
    print(f"final output min: {x.min().item():.8f}")
    print(f"final output max: {x.max().item():.8f}")
    print(f"total time: {total_ms:.3f} ms")
    used_bytes, entries, hits, misses = cache.stats()
    print(f"cache summary: {used_bytes / 1024**2:.1f} MiB / {args.cache_mib:.1f} MiB")
    print(f"cache summary: entries={entries}, hits={hits}, misses={misses}, evictions={cache.evictions}")

    del x
    cache.clear()
    set_gpu_cache(None)
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
