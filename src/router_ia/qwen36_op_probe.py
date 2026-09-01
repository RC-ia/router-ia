from __future__ import annotations

"""Safely probe one Qwen3.6 Layer-0 operation at a time."""

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from safetensors import safe_open

HIDDEN = 2048
QKV_OUT = 8192
BLOCK = 128
CONV_KERNEL = 4
CONV_CHANNELS = 8192
NUM_K_HEADS = 16
NUM_V_HEADS = 32
HEAD_DIM = 128
KEY_DIM = NUM_K_HEADS * HEAD_DIM
VALUE_DIM = NUM_V_HEADS * HEAD_DIM
LAYER_PREFIX = "model.language_model.layers.0."
EMBEDDING_NAME = "model.language_model.embed_tokens.weight"
EPS = 1e-6


def load_tensor(root: Path, name: str, device: str = "cpu") -> torch.Tensor:
    index_path = root / "model.safetensors.index.json"
    shard_name = None
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        shard_name = payload.get("weight_map", {}).get(name)
    shards = [root / shard_name] if shard_name else sorted(root.glob("*.safetensors"))
    for shard in shards:
        if not shard.is_file():
            continue
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            if name in handle.keys():
                return handle.get_tensor(name).to(device=device)
    raise KeyError(f"Tensor not found: {name}")


def load_embedding_row(root: Path, token_id: int) -> torch.Tensor:
    emb = load_tensor(root, EMBEDDING_NAME, device="cpu")
    if emb.ndim != 2 or emb.shape[1] != HIDDEN:
        raise ValueError(f"Unexpected embedding shape: {tuple(emb.shape)}")
    if not 0 <= token_id < emb.shape[0]:
        raise ValueError(f"token_id {token_id} outside vocabulary")
    return emb[token_id].float()


def rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x = x.float()
    w = weight.float()
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS) * (1.0 + w)


def dequantize_fp8_blockwise(weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 2 or scale_inv.ndim != 2:
        raise ValueError(
            f"Expected 2-D weight/scale tensors, got {tuple(weight.shape)} and {tuple(scale_inv.shape)}"
        )
    out_features, in_features = map(int, weight.shape)
    expected = ((out_features + BLOCK - 1) // BLOCK, (in_features + BLOCK - 1) // BLOCK)
    if tuple(scale_inv.shape) != expected:
        raise ValueError(
            f"Scale shape {tuple(scale_inv.shape)} does not match weight {tuple(weight.shape)}; expected {expected}"
        )
    expanded = scale_inv.float().repeat_interleave(BLOCK, dim=0).repeat_interleave(BLOCK, dim=1)
    return weight.float() * expanded[:out_features, :in_features]


def dequantize_conv1d_weight(weight: torch.Tensor) -> torch.Tensor:
    if tuple(weight.shape) != (CONV_CHANNELS, 1, CONV_KERNEL):
        raise ValueError(f"Unexpected conv weight shape: {tuple(weight.shape)}")
    return weight.float()


def stats(name: str, x: torch.Tensor) -> None:
    y = x.detach().float().cpu()
    print(f"{name}: shape={tuple(x.shape)} dtype={x.dtype} norm={torch.linalg.vector_norm(y).item():.8f} mean={y.mean().item():.8f} std={y.std().item():.8f}")
    print(f"{name}: min={y.min().item():.8f} max={y.max().item():.8f}")


def compute_qkv(root: Path, token_id: int, device: str) -> torch.Tensor:
    x = load_embedding_row(root, token_id).to(device)
    norm_weight = load_tensor(root, LAYER_PREFIX + "input_layernorm.weight", device=device)
    qkv_weight = load_tensor(root, LAYER_PREFIX + "linear_attn.in_proj_qkv.weight", device="cpu")
    qkv_scale = load_tensor(root, LAYER_PREFIX + "linear_attn.in_proj_qkv.weight_scale_inv", device="cpu")
    h = rmsnorm(x, norm_weight)
    qkv_fp32 = dequantize_fp8_blockwise(qkv_weight, qkv_scale).to(device)
    y = F.linear(h.float(), qkv_fp32.float())
    if device == "cuda":
        torch.cuda.synchronize()
    del x, norm_weight, qkv_weight, qkv_scale, h, qkv_fp32
    return y


def compute_conv(root: Path, token_id: int, device: str) -> torch.Tensor:
    qkv = compute_qkv(root, token_id, device)
    conv_weight = load_tensor(root, LAYER_PREFIX + "linear_attn.conv1d.weight", device="cpu")
    if tuple(conv_weight.shape) != (CONV_CHANNELS, 1, CONV_KERNEL):
        raise ValueError(f"Unexpected conv weight shape: {tuple(conv_weight.shape)}")
    conv = conv_weight.float().to(device)
    tap = conv[:, 0, -1].reshape(1, CONV_CHANNELS)
    y = F.silu(qkv.reshape(1, CONV_CHANNELS) * tap).reshape(CONV_CHANNELS)
    if device == "cuda":
        torch.cuda.synchronize()
    del qkv, conv_weight, conv, tap
    return y


def split_qkv(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.numel() != QKV_OUT:
        raise ValueError(f"Expected {QKV_OUT} QKV values, got {x.numel()}")
    q, k, v = torch.split(x.reshape(-1), [KEY_DIM, KEY_DIM, VALUE_DIM], dim=0)
    q = q.reshape(1, NUM_K_HEADS, HEAD_DIM)
    k = k.reshape(1, NUM_K_HEADS, HEAD_DIM)
    v = v.reshape(1, NUM_V_HEADS, HEAD_DIM)
    q32 = q.repeat_interleave(NUM_V_HEADS // NUM_K_HEADS, dim=1)
    k32 = k.repeat_interleave(NUM_V_HEADS // NUM_K_HEADS, dim=1)
    return q, k, v, q32, k32


def load_projection(root: Path, prefix: str, device: str) -> torch.Tensor:
    """Load a linear-attention projection, whether BF16 or FP8."""
    weight = load_tensor(root, prefix + ".weight", device="cpu")
    if weight.dtype == torch.float8_e4m3fn:
        scale = load_tensor(root, prefix + ".weight_scale_inv", device="cpu")
        out = dequantize_fp8_blockwise(weight, scale).to(device)
        del scale
    else:
        out = weight.float().to(device)
    del weight
    return out


def compute_delta_rule(root: Path, token_id: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    conv = compute_conv(root, token_id, device)
    _, _, v, q, k = split_qkv(conv)

    a_weight = load_projection(root, LAYER_PREFIX + "linear_attn.in_proj_a", device)
    b_weight = load_projection(root, LAYER_PREFIX + "linear_attn.in_proj_b", device)
    norm_weight = load_tensor(root, LAYER_PREFIX + "input_layernorm.weight", device=device)

    h = rmsnorm(load_embedding_row(root, token_id).to(device), norm_weight)
    a_raw = F.linear(h.float(), a_weight.float()).reshape(1, NUM_V_HEADS)
    b_raw = F.linear(h.float(), b_weight.float()).reshape(1, NUM_V_HEADS)
    beta = torch.sigmoid(b_raw)
    A_log = load_tensor(root, LAYER_PREFIX + "linear_attn.A_log", device=device).float().reshape(1, NUM_V_HEADS)
    dt_bias = load_tensor(root, LAYER_PREFIX + "linear_attn.dt_bias", device=device).float().reshape(1, NUM_V_HEADS)
    g = -torch.exp(A_log) * F.softplus(a_raw + dt_bias)
    decay = torch.exp(g)

    q = F.normalize(q.float(), dim=-1, eps=EPS)
    k = F.normalize(k.float(), dim=-1, eps=EPS)
    q = q * (HEAD_DIM ** -0.5)

    state = torch.zeros(1, NUM_V_HEADS, HEAD_DIM, HEAD_DIM, device=device, dtype=torch.float32)
    state = state * decay.unsqueeze(-1).unsqueeze(-1)
    retrieved = torch.einsum("bhkd,bhk->bhd", state, k)
    delta = (v.float() - retrieved) * beta.unsqueeze(-1)
    state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
    out = torch.einsum("bhkd,bhk->bhd", state, q)

    del conv, a_weight, b_weight, norm_weight, a_raw, b_raw, A_log, dt_bias, q, k, state
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return beta, g, decay, retrieved, delta, out


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.6 isolated Layer-0 operation probe")
    parser.add_argument("root", type=Path)
    parser.add_argument("--token-id", type=int, default=0)
    parser.add_argument("--op", choices=("norm", "qkv", "conv", "split", "delta"), default="qkv")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    root = args.root.resolve()

    if args.op == "norm":
        x = load_embedding_row(root, args.token_id).to(args.device)
        weight = load_tensor(root, LAYER_PREFIX + "input_layernorm.weight", device=args.device)
        stats("embedding", x)
        start = perf_counter()
        y = rmsnorm(x, weight)
        if args.device == "cuda":
            torch.cuda.synchronize()
        print(f"op=norm time={(perf_counter()-start)*1000:.3f} ms")
        stats("norm output", y)
        return

    if args.op == "qkv":
        x = load_embedding_row(root, args.token_id).to(args.device)
        norm_weight = load_tensor(root, LAYER_PREFIX + "input_layernorm.weight", device=args.device)
        qkv_weight = load_tensor(root, LAYER_PREFIX + "linear_attn.in_proj_qkv.weight", device="cpu")
        qkv_scale = load_tensor(root, LAYER_PREFIX + "linear_attn.in_proj_qkv.weight_scale_inv", device="cpu")
        h = rmsnorm(x, norm_weight)
        stats("norm input", h)
        print(f"qkv weight shape={tuple(qkv_weight.shape)} dtype={qkv_weight.dtype}")
        print(f"qkv scale shape={tuple(qkv_scale.shape)} dtype={qkv_scale.dtype}")
        start = perf_counter()
        qkv_fp32 = dequantize_fp8_blockwise(qkv_weight, qkv_scale).to(args.device)
        if args.device == "cuda":
            torch.cuda.synchronize()
        dequant_ms = (perf_counter() - start) * 1000.0
        start = perf_counter()
        y = F.linear(h.float(), qkv_fp32.float())
        if args.device == "cuda":
            torch.cuda.synchronize()
        compute_ms = (perf_counter() - start) * 1000.0
        print(f"op=qkv dequant time={dequant_ms:.3f} ms")
        print(f"op=qkv compute time={compute_ms:.3f} ms")
        stats("qkv output", y)
        del x, norm_weight, qkv_weight, qkv_scale, h, qkv_fp32, y
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
        return

    if args.op == "conv":
        qkv = compute_qkv(root, args.token_id, args.device)
        conv_weight = load_tensor(root, LAYER_PREFIX + "linear_attn.conv1d.weight", device="cpu")
        if tuple(conv_weight.shape) != (CONV_CHANNELS, 1, CONV_KERNEL):
            raise ValueError(f"Unexpected conv weight shape: {tuple(conv_weight.shape)}")
        print(f"qkv input shape={tuple(qkv.shape)}")
        print(f"conv weight shape={tuple(conv_weight.shape)} dtype={conv_weight.dtype}")
        print("conv scale shape=None")

        conv_fp32 = dequantize_conv1d_weight(conv_weight).to(args.device)
        qkv_current = qkv.reshape(1, CONV_CHANNELS)
        tap = conv_fp32[:, 0, -1].reshape(1, CONV_CHANNELS)
        start = perf_counter()
        y_linear = qkv_current * tap
        y = F.silu(y_linear)
        if args.device == "cuda":
            torch.cuda.synchronize()
        compute_ms = (perf_counter() - start) * 1000.0

        print("op=conv causal history=zeros")
        print("op=conv dequant time=0.000 ms (BF16 direct)")
        print(f"op=conv compute time={compute_ms:.3f} ms")
        stats("conv pre-activation", y_linear.reshape(-1))
        stats("conv output", y.reshape(-1))

        del qkv, conv_weight, conv_fp32, qkv_current, tap, y_linear, y
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
        return

    if args.op == "split":
        conv = compute_conv(root, args.token_id, args.device)
        start = perf_counter()
        q, k, v, q32, k32 = split_qkv(conv)
        if args.device == "cuda":
            torch.cuda.synchronize()
        split_ms = (perf_counter() - start) * 1000.0

        print("op=split")
        print(f"input shape: {tuple(conv.shape)}")
        print(f"Q shape: {tuple(q.shape)}")
        print(f"K shape: {tuple(k.shape)}")
        print(f"V shape: {tuple(v.shape)}")
        print(f"Q expanded shape: {tuple(q32.shape)}")
        print(f"K expanded shape: {tuple(k32.shape)}")
        print(f"split/reshape time: {split_ms:.3f} ms")
        print(f"Q norm: {torch.linalg.vector_norm(q).item():.8f}")
        print(f"K norm: {torch.linalg.vector_norm(k).item():.8f}")
        print(f"V norm: {torch.linalg.vector_norm(v).item():.8f}")
        print(f"Q expanded norm: {torch.linalg.vector_norm(q32).item():.8f}")
        print(f"K expanded norm: {torch.linalg.vector_norm(k32).item():.8f}")
        print("head configuration: Q/K=16x128 -> expanded to 32 heads; V=32x128")

        del conv, q, k, v, q32, k32
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
        return

    start = perf_counter()
    beta, g, decay, retrieved, delta, out = compute_delta_rule(root, args.token_id, args.device)
    if args.device == "cuda":
        torch.cuda.synchronize()
    total_ms = (perf_counter() - start) * 1000.0
    print("op=delta")
    stats("beta", beta)
    stats("g", g)
    stats("decay", decay)
    stats("retrieved", retrieved)
    stats("delta", delta)
    stats("delta output", out)
    print(f"op=delta total time={total_ms:.3f} ms")

    del beta, g, decay, retrieved, delta, out
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
