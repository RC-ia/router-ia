from __future__ import annotations

"""Execute the router-selected Top-K Qwen3.6 Layer-0 experts and aggregate them."""

import argparse
import gc
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from .qwen36_op_probe import (
    HEAD_DIM,
    LAYER_PREFIX,
    NUM_V_HEADS,
    dequantize_fp8_blockwise,
    load_embedding_row,
    load_projection,
    load_tensor,
    rmsnorm,
)
from .qwen36_gated_norm_probe import gated_rmsnorm
from .qwen36_router import route

HIDDEN = 2048
NUM_K_HEADS = 16
KEY_DIM = NUM_K_HEADS * HEAD_DIM
VALUE_DIM = NUM_V_HEADS * HEAD_DIM
EPS = 1e-6


def build_moe_input(root: Path, token_id: int, device: str) -> torch.Tensor:
    x0 = load_embedding_row(root, token_id).to(device)
    input_norm = load_tensor(root, LAYER_PREFIX + "input_layernorm.weight", device=device)
    h = rmsnorm(x0, input_norm)

    qkv_w = load_tensor(root, LAYER_PREFIX + "linear_attn.in_proj_qkv.weight", device="cpu")
    qkv_scale = load_tensor(root, LAYER_PREFIX + "linear_attn.in_proj_qkv.weight_scale_inv", device="cpu")
    qkv_w = dequantize_fp8_blockwise(qkv_w, qkv_scale).to(device)
    mixed = F.linear(h.float(), qkv_w.float()).reshape(1, KEY_DIM * 2 + VALUE_DIM)

    conv_w = load_tensor(root, LAYER_PREFIX + "linear_attn.conv1d.weight", device=device).float()
    mixed = F.silu(mixed * conv_w[:, 0, -1].reshape(1, -1))
    q, k, v = torch.split(mixed, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
    q = q.reshape(1, NUM_K_HEADS, HEAD_DIM).repeat_interleave(2, dim=1)
    k = k.reshape(1, NUM_K_HEADS, HEAD_DIM).repeat_interleave(2, dim=1)
    v = v.reshape(1, NUM_V_HEADS, HEAD_DIM)

    a_w = load_projection(root, LAYER_PREFIX + "linear_attn.in_proj_a", device)
    b_w = load_projection(root, LAYER_PREFIX + "linear_attn.in_proj_b", device)
    a_log = load_tensor(root, LAYER_PREFIX + "linear_attn.A_log", device=device).float().reshape(1, NUM_V_HEADS)
    dt_bias = load_tensor(root, LAYER_PREFIX + "linear_attn.dt_bias", device=device).float().reshape(1, NUM_V_HEADS)
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

    z_w = load_projection(root, LAYER_PREFIX + "linear_attn.in_proj_z", device)
    z = F.linear(h.float(), z_w.float()).reshape(1, NUM_V_HEADS, HEAD_DIM)
    norm_w = load_tensor(root, LAYER_PREFIX + "linear_attn.norm.weight", device=device)
    gated, _, _ = gated_rmsnorm(attn, z, norm_w)

    out_w = load_projection(root, LAYER_PREFIX + "linear_attn.out_proj", device)
    attn_projected = F.linear(gated.reshape(1, VALUE_DIM), out_w.float())
    residual = x0.reshape(1, HIDDEN) + attn_projected

    post_norm = load_tensor(root, LAYER_PREFIX + "post_attention_layernorm.weight", device=device)
    moe_in = rmsnorm(residual, post_norm)

    del x0, input_norm, h, qkv_w, qkv_scale, mixed, conv_w, q, k, v
    del a_w, b_w, a_log, dt_bias, a_raw, b_raw, beta, g, decay, qn, kn
    del state, retrieved, delta, attn, z_w, z, norm_w, gated, out_w, attn_projected, residual, post_norm
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return moe_in


def load_expert_projection(root: Path, layer: int, expert: int, kind: str, device: str) -> torch.Tensor:
    prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{kind}"
    weight = load_tensor(root, prefix + ".weight", device="cpu")
    if weight.ndim != 2:
        raise ValueError(f"Unexpected {kind} weight shape: {tuple(weight.shape)}")
    if weight.dtype == torch.float8_e4m3fn:
        scale = load_tensor(root, prefix + ".weight_scale_inv", device="cpu")
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
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the routed Top-K Qwen3.6 Layer-0 experts")
    parser.add_argument("root", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=None, help="Override routed experts with a comma-separated list")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--token-id", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")

    root = args.root.resolve()
    start_total = perf_counter()
    moe_in = build_moe_input(root, args.token_id, args.device)

    router_w = load_tensor(root, f"model.language_model.layers.{args.layer}.mlp.gate.weight", device=args.device).float()
    routed = route(moe_in.reshape(-1), router_w, top_k=args.top_k)
    router_ids = [int(x) for x in routed.expert_ids.detach().cpu().tolist()]
    router_weights = routed.weights.detach().cpu().tolist()

    if args.expert is None:
        expert_ids = router_ids
    else:
        expert_ids = [int(x.strip()) for x in str(args.expert).split(",") if x.strip()]
        if len(expert_ids) != args.top_k:
            raise SystemExit(f"--expert exige exatamente {args.top_k} IDs separados por vírgula")
        router_weights = [float(x) for x in routed.weights.detach().cpu().tolist()]

    x = moe_in.reshape(1, HIDDEN).float()
    weighted_sum = torch.zeros_like(x)
    expert_norms: list[float] = []

    start_experts = perf_counter()
    for idx, expert_id in enumerate(expert_ids):
        out = run_expert(root, args.layer, expert_id, x, args.device)
        expert_norms.append(float(torch.linalg.vector_norm(out).item()))
        weighted_sum.add_(out, alpha=float(router_weights[idx]))
        del out
    if args.device == "cuda":
        torch.cuda.synchronize()
    experts_ms = (perf_counter() - start_experts) * 1000.0
    total_ms = (perf_counter() - start_total) * 1000.0

    print("op=moe8")
    print(f"layer: {args.layer}")
    print(f"token id: {args.token_id}")
    print(f"moe input shape: {tuple(moe_in.shape)}")
    print(f"router top-{args.top_k} ids: {router_ids}")
    print(f"executed expert ids: {expert_ids}")
    print(f"router weights: {[round(float(w), 8) for w in router_weights]}")
    print(f"router weight sum: {sum(float(w) for w in router_weights):.8f}")
    print(f"expert output norms: {[round(x, 8) for x in expert_norms]}")
    print(f"aggregated MoE output shape: {tuple(weighted_sum.shape)}")
    print(f"moe input norm: {torch.linalg.vector_norm(x).item():.8f}")
    print(f"aggregated MoE output norm: {torch.linalg.vector_norm(weighted_sum).item():.8f}")
    print(f"aggregated MoE output mean: {weighted_sum.mean().item():.8f}")
    print(f"aggregated MoE output min: {weighted_sum.min().item():.8f}")
    print(f"aggregated MoE output max: {weighted_sum.max().item():.8f}")
    print(f"experts time: {experts_ms:.3f} ms")
    print(f"total time: {total_ms:.3f} ms")

    del moe_in, router_w, routed, x, weighted_sum
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
