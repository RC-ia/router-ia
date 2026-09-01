from __future__ import annotations

"""Run the validated Qwen3.6 Layer-0 path through the MoE router only."""

import argparse
import gc
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from .qwen36_delta_sequence_probe import token_params
from .qwen36_gated_norm_probe import gated_rmsnorm
from .qwen36_op_probe import (
    HEAD_DIM,
    LAYER_PREFIX,
    NUM_V_HEADS,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.6 Layer-0 combined probe through router")
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

    # Input and first normalization.
    x0 = load_embedding_row(root, args.token_id).to(device)
    input_norm = load_tensor(root, LAYER_PREFIX + "input_layernorm.weight", device=device)
    h = rmsnorm(x0, input_norm)

    # Q/K/V path: projection -> causal conv (zero history) -> split -> GQA expansion.
    qkv_w = load_tensor(root, LAYER_PREFIX + "linear_attn.in_proj_qkv.weight", device="cpu")
    qkv_scale = load_tensor(root, LAYER_PREFIX + "linear_attn.in_proj_qkv.weight_scale_inv", device="cpu")
    from .qwen36_op_probe import dequantize_fp8_blockwise
    qkv_w = dequantize_fp8_blockwise(qkv_w, qkv_scale).to(device)
    mixed = F.linear(h.float(), qkv_w.float()).reshape(1, KEY_DIM * 2 + VALUE_DIM)
    conv_w = load_tensor(root, LAYER_PREFIX + "linear_attn.conv1d.weight", device=device).float()
    mixed = F.silu(mixed * conv_w[:, 0, -1].reshape(1, -1))
    q, k, v = torch.split(mixed, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
    q = q.reshape(1, NUM_K_HEADS, HEAD_DIM).repeat_interleave(2, dim=1)
    k = k.reshape(1, NUM_K_HEADS, HEAD_DIM).repeat_interleave(2, dim=1)
    v = v.reshape(1, NUM_V_HEADS, HEAD_DIM)

    # Gated Delta Rule, first token, zero recurrent state.
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

    # Gated RMSNorm with z projection.
    z_w = load_projection(root, LAYER_PREFIX + "linear_attn.in_proj_z", device)
    z = F.linear(h.float(), z_w.float()).reshape(1, NUM_V_HEADS, HEAD_DIM)
    norm_w = load_tensor(root, LAYER_PREFIX + "linear_attn.norm.weight", device=device)
    gated, _, _ = gated_rmsnorm(attn, z, norm_w)

    # Output projection and residual.
    out_w = load_projection(root, LAYER_PREFIX + "linear_attn.out_proj", device)
    attn_projected = F.linear(gated.reshape(1, VALUE_DIM), out_w.float())
    residual = x0.reshape(1, HIDDEN) + attn_projected

    # Post-attention norm and router; do not execute any experts.
    post_norm = load_tensor(root, LAYER_PREFIX + "post_attention_layernorm.weight", device=device)
    moe_in = rmsnorm(residual, post_norm)
    router_w = load_tensor(root, LAYER_PREFIX + "mlp.gate.weight", device=device).float()
    routed = route(moe_in.reshape(-1), router_w, top_k=args.top_k)
    ids = routed.expert_ids.detach().cpu().tolist()
    weights = routed.weights.detach().cpu().tolist()

    if device == "cuda":
        torch.cuda.synchronize()
    total_ms = (perf_counter() - start_total) * 1000.0

    print("op=to_router")
    print(f"token id: {args.token_id}")
    print(f"input hidden shape: {tuple(x0.shape)}")
    print(f"attention projected shape: {tuple(attn_projected.shape)}")
    print(f"residual shape: {tuple(residual.shape)}")
    print(f"post-attention norm shape: {tuple(moe_in.shape)}")
    print(f"router weight shape: {tuple(router_w.shape)}")
    print(f"input hidden norm: {torch.linalg.vector_norm(x0).item():.8f}")
    print(f"attention projected norm: {torch.linalg.vector_norm(attn_projected).item():.8f}")
    print(f"residual norm: {torch.linalg.vector_norm(residual).item():.8f}")
    print(f"moe input norm: {torch.linalg.vector_norm(moe_in).item():.8f}")
    print(f"router top-{args.top_k} ids: {ids}")
    print(f"router weights: {[round(float(w), 8) for w in weights]}")
    print(f"router weight sum: {sum(float(w) for w in weights):.8f}")
    print(f"total time: {total_ms:.3f} ms")
    print("experts executed: NO")

    del x0, input_norm, h, qkv_w, qkv_scale, mixed, conv_w, q, k, v, a_w, b_w, a_log, dt_bias, a_raw, b_raw, beta, g, decay, qn, kn, state, retrieved, delta, attn, z_w, z, norm_w, gated, out_w, attn_projected, residual, post_norm, moe_in, router_w, routed
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
