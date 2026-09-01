from __future__ import annotations

"""Execute the validated Qwen3.6 Layer-0 linear-attention + MoE block."""

import argparse
import gc
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from .qwen36_gated_norm_probe import gated_rmsnorm
from .qwen36_moe8_probe import build_moe_input, load_expert_projection, run_expert
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
from .qwen36_router import route

HIDDEN = 2048
NUM_K_HEADS = 16
KEY_DIM = NUM_K_HEADS * HEAD_DIM
VALUE_DIM = NUM_V_HEADS * HEAD_DIM
EPS = 1e-6


def build_attention_residual(root: Path, token_id: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (original hidden residual, post-attention residual) for Layer 0."""
    x0 = load_embedding_row(root, token_id).to(device)
    input_norm = load_tensor(root, LAYER_PREFIX + "input_layernorm.weight", device=device)
    h = rmsnorm(x0, input_norm)

    qkv_w = load_tensor(root, LAYER_PREFIX + "linear_attn.in_proj_qkv.weight", device="cpu")
    qkv_scale = load_tensor(root, LAYER_PREFIX + "linear_attn.in_proj_qkv.weight_scale_inv", device="cpu")
    qkv_w = dequantize_fp8_blockwise(qkv_w, qkv_scale).to(device)
    mixed = F.linear(h.float(), qkv_w.float()).reshape(1, KEY_DIM * 2 + VALUE_DIM)

    conv_w = load_tensor(root, LAYER_PREFIX + "linear_attn.conv1d.weight", device=device).float()
    # Single-token decode: causal conv history is zero, so only the newest tap survives.
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
    post_attention_residual = x0.reshape(1, HIDDEN) + attn_projected

    del x0, input_norm, h, qkv_w, qkv_scale, mixed, conv_w, q, k, v
    del a_w, b_w, a_log, dt_bias, a_raw, b_raw, beta, g, decay, qn, kn
    del state, retrieved, delta, attn, z_w, z, norm_w, gated, out_w, attn_projected
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return load_embedding_row(root, token_id).to(device), post_attention_residual


def load_shared_projection(root: Path, layer: int, kind: str, device: str) -> torch.Tensor:
    prefix = f"model.language_model.layers.{layer}.mlp.shared_expert.{kind}"
    weight = load_tensor(root, prefix + ".weight", device="cpu")
    if weight.ndim != 2:
        raise ValueError(f"Unexpected shared {kind} shape: {tuple(weight.shape)}")
    if weight.dtype == torch.float8_e4m3fn:
        scale = load_tensor(root, prefix + ".weight_scale_inv", device="cpu")
        out = dequantize_fp8_blockwise(weight, scale).to(device)
        del scale
    else:
        out = weight.float().to(device)
    del weight
    return out


def run_shared_expert(root: Path, layer: int, x: torch.Tensor, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gate_w = load_shared_projection(root, layer, "gate_proj", device)
    up_w = load_shared_projection(root, layer, "up_proj", device)
    down_w = load_shared_projection(root, layer, "down_proj", device)
    shared_gate_w = load_tensor(
        root,
        f"model.language_model.layers.{layer}.mlp.shared_expert_gate.weight",
        device=device,
    ).float()

    shared_gate = torch.sigmoid(F.linear(x, shared_gate_w))
    gate = F.linear(x, gate_w)
    up = F.linear(x, up_w)
    hidden = F.silu(gate) * up
    raw = F.linear(hidden, down_w)
    out = raw * shared_gate

    del gate_w, up_w, down_w, shared_gate_w, gate, up, hidden, raw
    return out, shared_gate, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute validated Qwen3.6 Layer-0 block")
    parser.add_argument("root", type=Path)
    parser.add_argument("--token-id", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")

    root = args.root.resolve()
    device = args.device
    start_total = perf_counter()

    _, post_attention_residual = build_attention_residual(root, args.token_id, device)
    post_norm = load_tensor(root, LAYER_PREFIX + "post_attention_layernorm.weight", device=device)
    moe_in = rmsnorm(post_attention_residual, post_norm).reshape(1, HIDDEN).float()

    router_w = load_tensor(root, LAYER_PREFIX + "mlp.gate.weight", device=device).float()
    routed = route(moe_in.reshape(-1), router_w, top_k=args.top_k)
    expert_ids = [int(x) for x in routed.expert_ids.detach().cpu().tolist()]
    router_weights = [float(x) for x in routed.weights.detach().cpu().tolist()]

    routed_sum = torch.zeros_like(moe_in)
    expert_norms: list[float] = []
    start_experts = perf_counter()
    for expert_id, weight in zip(expert_ids, router_weights):
        out = run_expert(root, 0, expert_id, moe_in, device)
        expert_norms.append(float(torch.linalg.vector_norm(out).item()))
        routed_sum.add_(out, alpha=weight)
        del out
    if device == "cuda":
        torch.cuda.synchronize()
    experts_ms = (perf_counter() - start_experts) * 1000.0

    shared_out, shared_gate, _ = run_shared_expert(root, 0, moe_in, device)
    moe_out = routed_sum + shared_out
    layer_out = post_attention_residual + moe_out

    if device == "cuda":
        torch.cuda.synchronize()
    total_ms = (perf_counter() - start_total) * 1000.0

    print("op=layer0")
    print(f"layer: 0")
    print(f"token id: {args.token_id}")
    print(f"post-attention residual shape: {tuple(post_attention_residual.shape)}")
    print(f"moe input shape: {tuple(moe_in.shape)}")
    print(f"router top-{args.top_k} ids: {expert_ids}")
    print(f"router weights: {[round(x, 8) for x in router_weights]}")
    print(f"router weight sum: {sum(router_weights):.8f}")
    print(f"expert output norms: {[round(x, 8) for x in expert_norms]}")
    print(f"routed MoE output norm: {torch.linalg.vector_norm(routed_sum).item():.8f}")
    print(f"shared gate value: {shared_gate.item():.8f}")
    print(f"shared output norm: {torch.linalg.vector_norm(shared_out).item():.8f}")
    print(f"complete MoE output norm: {torch.linalg.vector_norm(moe_out).item():.8f}")
    print(f"complete layer output shape: {tuple(layer_out.shape)}")
    print(f"complete layer output norm: {torch.linalg.vector_norm(layer_out).item():.8f}")
    print(f"complete layer output mean: {layer_out.mean().item():.8f}")
    print(f"complete layer output min: {layer_out.min().item():.8f}")
    print(f"complete layer output max: {layer_out.max().item():.8f}")
    print(f"experts time: {experts_ms:.3f} ms")
    print(f"total time: {total_ms:.3f} ms")

    del post_attention_residual, post_norm, moe_in, router_w, routed, routed_sum, shared_out, shared_gate, moe_out, layer_out
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
