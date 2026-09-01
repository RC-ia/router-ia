from __future__ import annotations

"""Execute one Qwen3.6 routed MoE expert in isolation."""

import argparse
import gc
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from safetensors import safe_open

from .qwen36_op_probe import (
    KEY_DIM,
    VALUE_DIM,
    LAYER_PREFIX,
    dequantize_fp8_blockwise,
    load_embedding_row,
    load_projection,
    load_tensor,
    rmsnorm,
)

HIDDEN = 2048
NUM_K_HEADS = 16
NUM_V_HEADS = 32
HEAD_DIM = 128
DEFAULT_LAYER = 0
DEFAULT_EXPERT = 112
EPS = 1e-6


def load_expert_projection(root: Path, layer: int, expert: int, kind: str, device: str) -> torch.Tensor:
    """Load one per-expert projection, supporting BF16 and FP8 blockwise weights."""
    prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{kind}"
    weight = load_tensor(root, prefix + ".weight", device="cpu")
    if weight.ndim != 2:
        raise ValueError(f"Unexpected {kind} weight shape: {tuple(weight.shape)}")

    if weight.dtype == torch.float8_e4m3fn:
        scale_name = prefix + ".weight_scale_inv"
        scale = load_tensor(root, scale_name, device="cpu")
        out = dequantize_fp8_blockwise(weight, scale).to(device)
        del scale
    else:
        out = weight.float().to(device)

    del weight
    return out


def find_expert_shapes(root: Path, layer: int, expert: int) -> dict[str, tuple[int, ...]]:
    names = {
        kind: f"model.language_model.layers.{layer}.mlp.experts.{expert}.{kind}.weight"
        for kind in ("gate_proj", "up_proj", "down_proj")
    }
    found: dict[str, tuple[int, ...]] = {}
    missing = set(names.values())
    for shard in sorted(root.glob("*.safetensors")):
        if not missing:
            break
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            for kind, name in names.items():
                if name in keys:
                    found[kind] = tuple(int(x) for x in handle.get_slice(name).get_shape())
                    missing.discard(name)
    if missing:
        raise KeyError("Missing expert tensors: " + ", ".join(sorted(missing)))
    return found


def build_moe_input(root: Path, token_id: int, device: str, layer: int = 0) -> torch.Tensor:
    """Recreate the validated Layer-0 path and return the post-attention MoE input."""
    if layer != 0:
        raise ValueError("This isolated probe currently validates Layer 0 only")

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
    gated, _, _ = __import__("router_ia.qwen36_gated_norm_probe", fromlist=["gated_rmsnorm"]).gated_rmsnorm(attn, z, norm_w)

    out_w = load_projection(root, LAYER_PREFIX + "linear_attn.out_proj", device)
    attn_projected = F.linear(gated.reshape(1, VALUE_DIM), out_w.float())
    residual = x0.reshape(1, HIDDEN) + attn_projected

    post_norm = load_tensor(root, LAYER_PREFIX + "post_attention_layernorm.weight", device=device)
    moe_in = rmsnorm(residual, post_norm)

    return moe_in.detach()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Qwen3.6 Layer-0 MoE expert")
    parser.add_argument("root", type=Path)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--expert", type=int, default=DEFAULT_EXPERT)
    parser.add_argument("--token-id", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    root = args.root.resolve()
    device = args.device

    start_total = perf_counter()
    moe_in = build_moe_input(root, args.token_id, device, layer=args.layer)
    shape_info = find_expert_shapes(root, args.layer, args.expert)

    gate_w = load_expert_projection(root, args.layer, args.expert, "gate_proj", device)
    up_w = load_expert_projection(root, args.layer, args.expert, "up_proj", device)
    down_w = load_expert_projection(root, args.layer, args.expert, "down_proj", device)

    x = moe_in.reshape(1, HIDDEN).float()
    gate = F.linear(x, gate_w.float())
    up = F.linear(x, up_w.float())
    hidden = F.silu(gate) * up
    out = F.linear(hidden, down_w.float())

    if device == "cuda":
        torch.cuda.synchronize()
    total_ms = (perf_counter() - start_total) * 1000.0

    print("op=expert")
    print(f"layer: {args.layer}")
    print(f"expert: {args.expert}")
    print(f"token id: {args.token_id}")
    print(f"expert tensor shapes: {shape_info}")
    print(f"moe input shape: {tuple(moe_in.shape)}")
    print(f"gate shape: {tuple(gate.shape)}")
    print(f"up shape: {tuple(up.shape)}")
    print(f"down input shape: {tuple(hidden.shape)}")
    print(f"expert output shape: {tuple(out.shape)}")
    print(f"moe input norm: {torch.linalg.vector_norm(x).item():.8f}")
    print(f"gate norm: {torch.linalg.vector_norm(gate).item():.8f}")
    print(f"up norm: {torch.linalg.vector_norm(up).item():.8f}")
    print(f"gated hidden norm: {torch.linalg.vector_norm(hidden).item():.8f}")
    print(f"expert output norm: {torch.linalg.vector_norm(out).item():.8f}")
    print(f"expert output mean: {out.mean().item():.8f}")
    print(f"expert output min: {out.min().item():.8f}")
    print(f"expert output max: {out.max().item():.8f}")
    print(f"total time: {total_ms:.3f} ms")

    del moe_in, gate_w, up_w, down_w, x, gate, up, hidden, out
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
