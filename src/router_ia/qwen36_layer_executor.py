from __future__ import annotations

"""Generic single-token executor for Qwen3.6 linear-attention + routed MoE layers."""

import argparse
import gc
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from .qwen36_gated_norm_probe import gated_rmsnorm
from .qwen36_op_probe import (
    HEAD_DIM,
    NUM_V_HEADS,
    dequantize_fp8_blockwise,
    load_embedding_row,
    load_projection,
    load_tensor,
    rmsnorm,
)
from .qwen36_router import route

HIDDEN = 2048
NUM_K_HEADS = 16
KEY_DIM = NUM_K_HEADS * HEAD_DIM
VALUE_DIM = NUM_V_HEADS * HEAD_DIM
EPS = 1e-6


def prefix(layer: int, suffix: str) -> str:
    return f"model.language_model.layers.{layer}.{suffix}"


def layer_tensor(root: Path, layer: int, suffix: str, device: str = "cpu") -> torch.Tensor:
    return load_tensor(root, prefix(layer, suffix), device=device)


def build_layer_attention(root: Path, layer: int, x0: torch.Tensor, device: str) -> torch.Tensor:
    """Single-token linear-attention path with a zero causal-conv history."""
    input_norm = layer_tensor(root, layer, "input_layernorm.weight", device=device)
    h = rmsnorm(x0, input_norm)

    qkv_w = layer_tensor(root, layer, "linear_attn.in_proj_qkv.weight", device="cpu")
    qkv_scale = layer_tensor(root, layer, "linear_attn.in_proj_qkv.weight_scale_inv", device="cpu")
    qkv_w = dequantize_fp8_blockwise(qkv_w, qkv_scale).to(device)
    mixed = F.linear(h.float(), qkv_w.float()).reshape(1, KEY_DIM * 2 + VALUE_DIM)

    conv_w = layer_tensor(root, layer, "linear_attn.conv1d.weight", device=device).float()
    mixed = F.silu(mixed * conv_w[:, 0, -1].reshape(1, -1))
    q, k, v = torch.split(mixed, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
    q = q.reshape(1, NUM_K_HEADS, HEAD_DIM).repeat_interleave(2, dim=1)
    k = k.reshape(1, NUM_K_HEADS, HEAD_DIM).repeat_interleave(2, dim=1)
    v = v.reshape(1, NUM_V_HEADS, HEAD_DIM)

    a_w = layer_tensor(root, layer, "linear_attn.in_proj_a.weight", device="cpu")
    b_w = layer_tensor(root, layer, "linear_attn.in_proj_b.weight", device="cpu")
    if a_w.dtype == torch.float8_e4m3fn:
        a_scale = layer_tensor(root, layer, "linear_attn.in_proj_a.weight_scale_inv", device="cpu")
        a_w = dequantize_fp8_blockwise(a_w, a_scale).to(device)
        del a_scale
    else:
        a_w = a_w.float().to(device)
    if b_w.dtype == torch.float8_e4m3fn:
        b_scale = layer_tensor(root, layer, "linear_attn.in_proj_b.weight_scale_inv", device="cpu")
        b_w = dequantize_fp8_blockwise(b_w, b_scale).to(device)
        del b_scale
    else:
        b_w = b_w.float().to(device)

    a_log = layer_tensor(root, layer, "linear_attn.A_log", device=device).float().reshape(1, NUM_V_HEADS)
    dt_bias = layer_tensor(root, layer, "linear_attn.dt_bias", device=device).float().reshape(1, NUM_V_HEADS)
    a_raw = F.linear(h.float(), a_w.float()).reshape(1, NUM_V_HEADS)
    b_raw = F.linear(h.float(), b_w.float()).reshape(1, NUM_V_HEADS)
    beta = torch.sigmoid(b_raw)
    g = -torch.exp(a_log) * F.softplus(a_raw + dt_bias)
    decay = torch.exp(g)

    qn = F.normalize(q.float(), dim=-1, eps=EPS) * (HEAD_DIM ** -0.5)
    kn = F.normalize(k.float(), dim=-1, eps=EPS)
    state = torch.zeros(1, NUM_V_HEADS, HEAD_DIM, HEAD_DIM, device=device, dtype=torch.float32)
    state = state * decay.unsqueeze(-1).unsqueeze(-1)
    retrieved = torch.einsum("bhkd,bhk->bhd", state, kn)
    delta = (v.float() - retrieved) * beta.unsqueeze(-1)
    state = state + kn.unsqueeze(-1) * delta.unsqueeze(-2)
    attn = torch.einsum("bhkd,bhk->bhd", state, qn)

    z_w = layer_tensor(root, layer, "linear_attn.in_proj_z.weight", device="cpu")
    if z_w.dtype == torch.float8_e4m3fn:
        z_scale = layer_tensor(root, layer, "linear_attn.in_proj_z.weight_scale_inv", device="cpu")
        z_w = dequantize_fp8_blockwise(z_w, z_scale).to(device)
        del z_scale
    else:
        z_w = z_w.float().to(device)
    z = F.linear(h.float(), z_w.float()).reshape(1, NUM_V_HEADS, HEAD_DIM)

    norm_w = layer_tensor(root, layer, "linear_attn.norm.weight", device=device)
    gated, _, _ = gated_rmsnorm(attn, z, norm_w)

    out_w = layer_tensor(root, layer, "linear_attn.out_proj.weight", device="cpu")
    if out_w.dtype == torch.float8_e4m3fn:
        out_scale = layer_tensor(root, layer, "linear_attn.out_proj.weight_scale_inv", device="cpu")
        out_w = dequantize_fp8_blockwise(out_w, out_scale).to(device)
        del out_scale
    else:
        out_w = out_w.float().to(device)
    attn_projected = F.linear(gated.reshape(1, VALUE_DIM), out_w.float())
    residual = x0.reshape(1, HIDDEN) + attn_projected

    del input_norm, h, qkv_w, qkv_scale, mixed, conv_w, q, k, v
    del a_w, b_w, a_log, dt_bias, a_raw, b_raw, beta, g, decay, qn, kn
    del state, retrieved, delta, attn, z_w, z, norm_w, gated, out_w, attn_projected
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return residual


def load_expert_projection(root: Path, layer: int, expert: int, kind: str, device: str) -> torch.Tensor:
    p = prefix(layer, f"mlp.experts.{expert}.{kind}.weight")
    weight = load_tensor(root, p, device="cpu")
    if weight.dtype == torch.float8_e4m3fn:
        scale = load_tensor(root, prefix(layer, f"mlp.experts.{expert}.{kind}.weight_scale_inv"), device="cpu")
        out = dequantize_fp8_blockwise(weight, scale).to(device)
        del scale
    else:
        out = weight.float().to(device)
    del weight
    return out


def run_expert(root: Path, layer: int, expert: int, x: torch.Tensor, device: str) -> torch.Tensor:
    gate_w = load_expert_projection(root, layer, expert, "gate_proj", device)
    up_w = load_expert_projection(root, layer, expert, "up_proj", device)
    down_w = load_expert_projection(root, layer, expert, "down_proj", device)
    gate = F.linear(x, gate_w.float())
    up = F.linear(x, up_w.float())
    hidden = F.silu(gate) * up
    out = F.linear(hidden, down_w.float())
    del gate_w, up_w, down_w, gate, up, hidden
    return out


def load_shared_projection(root: Path, layer: int, kind: str, device: str) -> torch.Tensor:
    p = prefix(layer, f"mlp.shared_expert.{kind}.weight")
    weight = load_tensor(root, p, device="cpu")
    if weight.dtype == torch.float8_e4m3fn:
        scale = load_tensor(root, prefix(layer, f"mlp.shared_expert.{kind}.weight_scale_inv"), device="cpu")
        out = dequantize_fp8_blockwise(weight, scale).to(device)
        del scale
    else:
        out = weight.float().to(device)
    del weight
    return out


def run_shared_expert(root: Path, layer: int, x: torch.Tensor, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    gate_w = load_shared_projection(root, layer, "gate_proj", device)
    up_w = load_shared_projection(root, layer, "up_proj", device)
    down_w = load_shared_projection(root, layer, "down_proj", device)
    gate_vector = layer_tensor(root, layer, "mlp.shared_expert_gate.weight", device=device).float()
    gate = torch.sigmoid(F.linear(x, gate_vector))
    hidden = F.silu(F.linear(x, gate_w.float())) * F.linear(x, up_w.float())
    raw = F.linear(hidden, down_w.float())
    out = raw * gate
    del gate_w, up_w, down_w, gate_vector, hidden, raw
    return out, gate


def execute_layer(root: Path, layer: int, x: torch.Tensor, device: str, top_k: int = 8) -> tuple[torch.Tensor, dict[str, object]]:
    x = x.reshape(1, HIDDEN).float().to(device)
    residual = build_layer_attention(root, layer, x, device)
    post_norm = layer_tensor(root, layer, "post_attention_layernorm.weight", device=device)
    moe_in = rmsnorm(residual, post_norm).reshape(1, HIDDEN).float()
    router_w = layer_tensor(root, layer, "mlp.gate.weight", device=device).float()
    routed = route(moe_in.reshape(-1), router_w, top_k=top_k)
    ids = [int(v) for v in routed.expert_ids.detach().cpu().tolist()]
    weights = [float(v) for v in routed.weights.detach().cpu().tolist()]
    routed_sum = torch.zeros_like(moe_in)
    for expert_id, weight in zip(ids, weights):
        routed_sum.add_(run_expert(root, layer, expert_id, moe_in, device), alpha=weight)
    shared_out, shared_gate = run_shared_expert(root, layer, moe_in, device)
    moe_out = routed_sum + shared_out
    layer_out = residual + moe_out
    info = {"expert_ids": ids, "router_weights": weights, "shared_gate": float(shared_gate.item()), "moe_input_norm": float(torch.linalg.vector_norm(moe_in).item())}
    del residual, post_norm, moe_in, router_w, routed, routed_sum, shared_out, shared_gate, moe_out
    gc.collect()
    return layer_out, info


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute one Qwen3.6 layer using a 2048-d hidden state")
    parser.add_argument("root", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--token-id", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")

    root = args.root.resolve()
    x = load_embedding_row(root, args.token_id).reshape(1, HIDDEN).to(args.device)
    start = perf_counter()
    out, info = execute_layer(root, args.layer, x, args.device, args.top_k)
    if args.device == "cuda":
        torch.cuda.synchronize()
    total_ms = (perf_counter() - start) * 1000.0
    print(f"op=layer{args.layer}")
    print(f"layer: {args.layer}")
    print(f"token id: {args.token_id}")
    print(f"router top-{args.top_k} ids: {info['expert_ids']}")
    print(f"router weights: {[round(w, 8) for w in info['router_weights']]}")
    print(f"shared gate value: {info['shared_gate']:.8f}")
    print(f"moe input norm: {info['moe_input_norm']:.8f}")
    print(f"layer output shape: {tuple(out.shape)}")
    print(f"layer output norm: {torch.linalg.vector_norm(out).item():.8f}")
    print(f"layer output mean: {out.mean().item():.8f}")
    print(f"layer output min: {out.min().item():.8f}")
    print(f"layer output max: {out.max().item():.8f}")
    print(f"total time: {total_ms:.3f} ms")

    del x, out
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
