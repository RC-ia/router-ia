from __future__ import annotations

"""Run a sequential Qwen3.6 single-token inference loop.

Qwen3.6 uses a 3:1 hybrid backbone: three Gated DeltaNet/linear-attention
layers followed by one full-attention layer. This runner detects that layout
per layer and executes the corresponding attention path, then runs the shared
MoE block.

This remains a conservative single-token reference runner. Full-attention
KV-cache across a multi-token sequence is intentionally not implemented yet.
"""

import argparse
import gc
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from .qwen36_gated_norm_probe import gated_rmsnorm
from .qwen36_op_probe import (
    HEAD_DIM,
    dequantize_fp8_blockwise,
    load_embedding_row,
    load_projection,
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
FULL_KV_DIM = FULL_NUM_KV_HEADS * FULL_HEAD_DIM
FULL_Q_GATE_DIM = FULL_Q_DIM * 2
FULL_ROPE_DIM = int(FULL_HEAD_DIM * 0.25)
FULL_NUM_KV_GROUPS = FULL_NUM_HEADS // FULL_NUM_KV_HEADS
EPS = 1e-6
DEFAULT_LAYERS = 40


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


def linear_attention_step(root: Path, layer: int, x0: torch.Tensor, device: str) -> torch.Tensor:
    prefix = layer_prefix(layer)
    input_norm = load_layer_weight(root, layer, "input_layernorm.weight", device)
    h = rmsnorm(x0, input_norm)

    qkv_w = load_layer_weight(root, layer, "linear_attn.in_proj_qkv.weight", "cpu")
    qkv_scale = load_layer_weight(root, layer, "linear_attn.in_proj_qkv.weight_scale_inv", "cpu")
    qkv_w = dequantize_fp8_blockwise(qkv_w, qkv_scale).to(device)
    mixed = F.linear(h.float(), qkv_w.float()).reshape(1, LINEAR_KEY_DIM * 2 + LINEAR_VALUE_DIM)

    conv_w = load_layer_weight(root, layer, "linear_attn.conv1d.weight", device).float()
    mixed = F.silu(mixed * conv_w[:, 0, -1].reshape(1, -1))
    q, k, v = torch.split(mixed, [LINEAR_KEY_DIM, LINEAR_KEY_DIM, LINEAR_VALUE_DIM], dim=-1)
    q = q.reshape(1, LINEAR_NUM_K_HEADS, 128).repeat_interleave(2, dim=1)
    k = k.reshape(1, LINEAR_NUM_K_HEADS, 128).repeat_interleave(2, dim=1)
    v = v.reshape(1, LINEAR_NUM_V_HEADS, 128)

    a_w = load_projection(root, prefix + "linear_attn.in_proj_a", device)
    b_w = load_projection(root, prefix + "linear_attn.in_proj_b", device)
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

    z_w = load_projection(root, prefix + "linear_attn.in_proj_z", device)
    z = F.linear(h.float(), z_w.float()).reshape(1, LINEAR_NUM_V_HEADS, 128)
    norm_w = load_layer_weight(root, layer, "linear_attn.norm.weight", device)
    gated, _, _ = gated_rmsnorm(attn, z, norm_w)

    out_w = load_projection(root, prefix + "linear_attn.out_proj", device)
    attn_projected = F.linear(gated.reshape(1, LINEAR_VALUE_DIM), out_w.float())
    residual = x0.reshape(1, HIDDEN) + attn_projected

    del input_norm, h, qkv_w, qkv_scale, mixed, conv_w, q, k, v
    del a_w, b_w, a_log, dt_bias, a_raw, b_raw, beta, g, decay, qn, kn
    del state, retrieved, delta, attn, z_w, z, norm_w, gated, out_w, attn_projected
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return residual


def full_attention_step(root: Path, layer: int, x0: torch.Tensor, device: str) -> torch.Tensor:
    """Execute one-token full attention for the periodic Qwen3.6 full-attn layers.

    With one token at position 0 there is no past KV cache and RoPE is an exact
    identity (cos=1, sin=0). We nevertheless keep the head normalization,
    grouped-query expansion, causal softmax and output gate explicit.
    """
    prefix = layer_prefix(layer)
    input_norm = load_layer_weight(root, layer, "input_layernorm.weight", device)
    h = rmsnorm(x0, input_norm)

    q_w = load_layer_weight(root, layer, "self_attn.q_proj.weight", "cpu")
    q_scale = load_layer_weight(root, layer, "self_attn.q_proj.weight_scale_inv", "cpu")
    q_w = dequantize_fp8_blockwise(q_w, q_scale).to(device)
    k_w = load_layer_weight(root, layer, "self_attn.k_proj.weight", "cpu")
    k_scale = load_layer_weight(root, layer, "self_attn.k_proj.weight_scale_inv", "cpu")
    k_w = dequantize_fp8_blockwise(k_w, k_scale).to(device)
    v_w = load_layer_weight(root, layer, "self_attn.v_proj.weight", "cpu")
    v_scale = load_layer_weight(root, layer, "self_attn.v_proj.weight_scale_inv", "cpu")
    v_w = dequantize_fp8_blockwise(v_w, v_scale).to(device)

    q_gate = F.linear(h.float(), q_w.float()).reshape(1, FULL_NUM_HEADS, FULL_HEAD_DIM * 2)
    q, gate = torch.chunk(q_gate, 2, dim=-1)
    k = F.linear(h.float(), k_w.float()).reshape(1, FULL_NUM_KV_HEADS, FULL_HEAD_DIM)
    v = F.linear(h.float(), v_w.float()).reshape(1, FULL_NUM_KV_HEADS, FULL_HEAD_DIM)

    q_norm_w = load_layer_weight(root, layer, "self_attn.q_norm.weight", device)
    k_norm_w = load_layer_weight(root, layer, "self_attn.k_norm.weight", device)
    q = rmsnorm(q, q_norm_w).float()
    k = rmsnorm(k, k_norm_w).float()

    # Position 0: applying partial RoPE would be exactly the identity.
    k = k.repeat_interleave(FULL_NUM_KV_GROUPS, dim=1)
    v = v.repeat_interleave(FULL_NUM_KV_GROUPS, dim=1)

    scores = torch.matmul(q.unsqueeze(2), k.unsqueeze(-1)).squeeze(-1) * (FULL_HEAD_DIM ** -0.5)
    attn_weights = torch.softmax(scores.float(), dim=-1)
    attn = torch.matmul(attn_weights.unsqueeze(2), v.unsqueeze(-1)).squeeze(-1)
    attn = attn * torch.sigmoid(gate)
    attn_flat = attn.reshape(1, FULL_Q_DIM)

    out_w = load_layer_weight(root, layer, "self_attn.o_proj.weight", "cpu")
    out_scale = load_layer_weight(root, layer, "self_attn.o_proj.weight_scale_inv", "cpu")
    out_w = dequantize_fp8_blockwise(out_w, out_scale).to(device)
    attn_projected = F.linear(attn_flat, out_w.float())
    residual = x0.reshape(1, HIDDEN) + attn_projected

    del input_norm, h, q_w, q_scale, k_w, k_scale, v_w, v_scale
    del q_gate, q, gate, k, v, q_norm_w, k_norm_w, scores, attn_weights, attn
    del attn_flat, out_w, out_scale, attn_projected
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return residual


def load_moe_projection(root: Path, layer: int, expert: int, kind: str, device: str) -> torch.Tensor:
    prefix = f"{layer_prefix(layer)}mlp.experts.{expert}.{kind}"
    weight = load_tensor(root, prefix + ".weight", device="cpu")
    if weight.dtype == torch.float8_e4m3fn:
        scale = load_tensor(root, prefix + ".weight_scale_inv", device="cpu")
        out = dequantize_fp8_blockwise(weight, scale).to(device)
        del scale
    else:
        out = weight.float().to(device)
    del weight
    return out


def run_routed_expert(root: Path, layer: int, expert: int, x: torch.Tensor, device: str) -> torch.Tensor:
    gate_w = load_moe_projection(root, layer, expert, "gate_proj", device)
    up_w = load_moe_projection(root, layer, expert, "up_proj", device)
    down_w = load_moe_projection(root, layer, expert, "down_proj", device)
    gate = F.linear(x, gate_w)
    up = F.linear(x, up_w)
    hidden = F.silu(gate) * up
    out = F.linear(hidden, down_w)
    del gate_w, up_w, down_w, gate, up, hidden
    return out


def load_shared_projection(root: Path, layer: int, kind: str, device: str) -> torch.Tensor:
    prefix = f"{layer_prefix(layer)}mlp.shared_expert.{kind}"
    weight = load_tensor(root, prefix + ".weight", device="cpu")
    if weight.dtype == torch.float8_e4m3fn:
        scale = load_tensor(root, prefix + ".weight_scale_inv", device="cpu")
        out = dequantize_fp8_blockwise(weight, scale).to(device)
        del scale
    else:
        out = weight.float().to(device)
    del weight
    return out


def run_shared_expert(root: Path, layer: int, x: torch.Tensor, device: str) -> tuple[torch.Tensor, float]:
    gate_w = load_shared_projection(root, layer, "gate_proj", device)
    up_w = load_shared_projection(root, layer, "up_proj", device)
    down_w = load_shared_projection(root, layer, "down_proj", device)
    shared_gate_w = load_layer_weight(root, layer, "mlp.shared_expert_gate.weight", device).float()

    gate = torch.sigmoid(F.linear(x, shared_gate_w))
    hidden_gate = F.linear(x, gate_w)
    up = F.linear(x, up_w)
    hidden = F.silu(hidden_gate) * up
    raw = F.linear(hidden, down_w)
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
    if device == "cuda":
        torch.cuda.empty_cache()
    return layer_out, expert_ids, weights, shared_gate, moe_input_norm


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen3.6 through all layers sequentially")
    parser.add_argument("root", type=Path)
    parser.add_argument("--token-id", type=int, default=0)
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--end-layer", type=int, default=DEFAULT_LAYERS - 1)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--quiet", action="store_true", help="Only print final result")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    if not 0 <= args.start_layer <= args.end_layer < DEFAULT_LAYERS:
        raise SystemExit(f"layer range must be inside 0..{DEFAULT_LAYERS - 1}")

    root = args.root.resolve()
    x = load_embedding_row(root, args.token_id).reshape(1, HIDDEN).to(args.device).float()
    start_total = perf_counter()

    if not args.quiet:
        print("op=loop")
        print(f"token id: {args.token_id}")
        print(f"layers: {args.start_layer}..{args.end_layer}")
        print(f"device: {args.device}")
        print(f"input shape: {tuple(x.shape)}")
        print(f"input norm: {torch.linalg.vector_norm(x).item():.8f}")

    for layer in range(args.start_layer, args.end_layer + 1):
        start_layer = perf_counter()
        x_before = x
        kind = attention_type(root, layer)
        if kind == "linear_attention":
            residual = linear_attention_step(root, layer, x_before, args.device)
        else:
            residual = full_attention_step(root, layer, x_before, args.device)

        x, expert_ids, weights, shared_gate, moe_input_norm = moe_step(
            root, layer, residual, args.top_k, args.device
        )
        if args.device == "cuda":
            torch.cuda.synchronize()
        layer_ms = (perf_counter() - start_layer) * 1000.0

        if not args.quiet:
            print(f"layer {layer} ({kind}):")
            print(f"  router top-{args.top_k}: {expert_ids}")
            print(f"  router weights: {[round(v, 8) for v in weights]}")
            print(f"  shared gate: {shared_gate:.8f}")
            print(f"  moe input norm: {moe_input_norm:.8f}")
            print(f"  output shape: {tuple(x.shape)}")
            print(f"  output norm: {torch.linalg.vector_norm(x).item():.8f}")
            print(f"  output mean: {x.mean().item():.8f}")
            print(f"  time: {layer_ms:.3f} ms")
        del x_before, residual

    if args.device == "cuda":
        torch.cuda.synchronize()
    total_ms = (perf_counter() - start_total) * 1000.0
    print(f"final output shape: {tuple(x.shape)}")
    print(f"final output norm: {torch.linalg.vector_norm(x).item():.8f}")
    print(f"final output mean: {x.mean().item():.8f}")
    print(f"final output min: {x.min().item():.8f}")
    print(f"final output max: {x.max().item():.8f}")
    print(f"total time: {total_ms:.3f} ms")

    del x
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
