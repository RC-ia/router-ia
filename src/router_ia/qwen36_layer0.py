from __future__ import annotations

"""Experimental single-token Qwen3.6 Layer 0 forward.

Layer 0 of Qwen3.6 is a linear-attention (Gated DeltaNet) block followed by
an MoE block. This implementation targets first-token/decode semantics with
zero initial recurrent and convolution state. It intentionally loads only
the tensors required by one layer and reuses the existing FP8 expert cache.
"""

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from safetensors import safe_open

from .fp8_expert_cache import FP8ExpertCache
from .fp8_expert_runner import _dequantize_blockwise
from .qwen36_moe_probe import discover_embedding_shard, load_embedding
from .qwen36_router import DEFAULT_HIDDEN_SIZE, route

EPS = 1e-6
HIDDEN = 2048
NUM_K_HEADS = 16
NUM_V_HEADS = 32
K_HEAD_DIM = 128
V_HEAD_DIM = 128
KEY_DIM = NUM_K_HEADS * K_HEAD_DIM
VALUE_DIM = NUM_V_HEADS * V_HEAD_DIM
CONV_DIM = KEY_DIM * 2 + VALUE_DIM
CONV_KERNEL = 4
MOE_INTERMEDIATE = 512
EMBEDDING_NAME = "model.language_model.embed_tokens.weight"
LAYER_PREFIX = "model.language_model.layers.0."


def _load_tensor(root: Path, name: str, *, device: str = "cpu") -> torch.Tensor:
    index_path = root / "model.safetensors.index.json"
    shard_name: str | None = None
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


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + EPS)
    return y * (1.0 + weight.float())


def _gated_rmsnorm(x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + EPS)
    y = y * weight.float()
    return y * F.silu(z.float())


def _linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(x, weight.float())


def _first_token_conv(mixed_qkv: torch.Tensor, conv_weight: torch.Tensor) -> torch.Tensor:
    """Causal depthwise conv for the first token with zero history."""
    if conv_weight.ndim != 3:
        raise ValueError(f"Unexpected conv1d weight shape: {tuple(conv_weight.shape)}")
    if conv_weight.shape[0] != CONV_DIM or conv_weight.shape[1] != 1 or conv_weight.shape[2] != CONV_KERNEL:
        raise ValueError(f"Unexpected conv1d weight shape: {tuple(conv_weight.shape)}")
    # With zero left context, only the final causal kernel tap sees token t=0.
    out = mixed_qkv * conv_weight[:, 0, -1].view(1, -1)
    return F.silu(out)


def _gated_delta_first_token(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Reference single-step recurrent gated-delta update with zero state."""
    q = F.normalize(query.float(), p=2.0, dim=-1, eps=EPS) / (K_HEAD_DIM**0.5)
    k = F.normalize(key.float(), p=2.0, dim=-1, eps=EPS)
    v = value.float()
    decay = torch.exp(g.float()).view(1, NUM_V_HEADS, 1, 1)
    b = beta.float().sigmoid().view(1, NUM_V_HEADS, 1)

    # First token starts from an all-zero recurrent state.
    state = torch.zeros(
        1, NUM_V_HEADS, K_HEAD_DIM, V_HEAD_DIM,
        dtype=torch.float32,
        device=value.device,
    )
    state = state * decay
    # delta = v - state @ k = v for zero state.
    delta = v - torch.zeros_like(v)
    delta = delta * b
    state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
    out = (state * q.unsqueeze(-1)).sum(dim=-2)
    return out


def _run_shared_expert(root: Path, x: torch.Tensor) -> torch.Tensor:
    names = {
        "gate": f"{LAYER_PREFIX}mlp.shared_expert.gate_proj.weight",
        "up": f"{LAYER_PREFIX}mlp.shared_expert.up_proj.weight",
        "down": f"{LAYER_PREFIX}mlp.shared_expert.down_proj.weight",
        "switch": f"{LAYER_PREFIX}mlp.shared_expert_gate.weight",
    }
    gate = _load_tensor(root, names["gate"], device=x.device)
    up = _load_tensor(root, names["up"], device=x.device)
    down = _load_tensor(root, names["down"], device=x.device)
    switch = _load_tensor(root, names["switch"], device=x.device)
    hidden = F.silu(_linear(x, gate)) * _linear(x, up)
    shared = _linear(hidden, down)
    gate_value = torch.sigmoid(_linear(x, switch))
    out = shared * gate_value
    del gate, up, down, switch, hidden, shared, gate_value
    return out


def _run_routed_experts(root: Path, x: torch.Tensor, cache: FP8ExpertCache, ids: list[int], weights: list[float]) -> torch.Tensor:
    aggregate = torch.zeros(HIDDEN, dtype=torch.float32, device=x.device)
    for expert_id, weight in zip(ids, weights):
        blob = cache.get(0, int(expert_id), tier="vram" if x.device.type == "cuda" else "ram")
        gate = _dequantize_blockwise(blob.weights["gate_proj"], blob.scales["gate_proj"]).to(x.device)
        up = _dequantize_blockwise(blob.weights["up_proj"], blob.scales["up_proj"]).to(x.device)
        down = _dequantize_blockwise(blob.weights["down_proj"], blob.scales["down_proj"]).to(x.device)
        expert_hidden = F.silu(F.linear(x, gate)) * F.linear(x, up)
        out = F.linear(expert_hidden, down)
        aggregate.add_(out.float(), alpha=float(weight))
        del gate, up, down, expert_hidden, out, blob
        gc.collect()
        if x.device.type == "cuda":
            torch.cuda.empty_cache()
    return aggregate


def run_layer0(root: Path, token_id: int, device: str, ram_gb: float, vram_gb: float) -> torch.Tensor:
    x0 = load_embedding(root, token_id, device)
    tensors = {}
    tensor_names = [
        "input_layernorm.weight",
        "linear_attn.in_proj_qkv.weight",
        "linear_attn.in_proj_z.weight",
        "linear_attn.in_proj_b.weight",
        "linear_attn.in_proj_a.weight",
        "linear_attn.conv1d.weight",
        "linear_attn.A_log",
        "linear_attn.dt_bias",
        "linear_attn.norm.weight",
        "linear_attn.out_proj.weight",
        "post_attention_layernorm.weight",
    ]
    for suffix in tensor_names:
        tensors[suffix] = _load_tensor(root, LAYER_PREFIX + suffix, device=device)

    h = _rmsnorm(x0, tensors["input_layernorm.weight"])

    mixed = _linear(h, tensors["linear_attn.in_proj_qkv.weight"]).view(1, CONV_DIM)
    mixed = _first_token_conv(mixed, tensors["linear_attn.conv1d.weight"])
    q, k, v = torch.split(mixed, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
    q = q.view(1, NUM_K_HEADS, K_HEAD_DIM)
    k = k.view(1, NUM_K_HEADS, K_HEAD_DIM)
    v = v.view(1, NUM_V_HEADS, V_HEAD_DIM)
    q = q.repeat_interleave(NUM_V_HEADS // NUM_K_HEADS, dim=1)
    k = k.repeat_interleave(NUM_V_HEADS // NUM_K_HEADS, dim=1)
    z = _linear(h, tensors["linear_attn.in_proj_z.weight"]).view(1, NUM_V_HEADS, V_HEAD_DIM)
    b = _linear(h, tensors["linear_attn.in_proj_b.weight"])
    a = _linear(h, tensors["linear_attn.in_proj_a.weight"])
    g = -tensors["linear_attn.A_log"].float().exp() * F.softplus(a.float() + tensors["linear_attn.dt_bias"].float())

    attn = _gated_delta_first_token(q, k, v, g, b)
    attn = _gated_rmsnorm(attn.reshape(1, VALUE_DIM), z.reshape(1, VALUE_DIM), tensors["linear_attn.norm.weight"])
    mixer_out = _linear(attn, tensors["linear_attn.out_proj.weight"]).reshape(HIDDEN)
    residual = x0 + mixer_out

    moe_in = _rmsnorm(residual, tensors["post_attention_layernorm.weight"])
    gate = _load_tensor(root, f"{LAYER_PREFIX}mlp.gate.weight", device=device).float()
    route_result = route(moe_in, gate, top_k=8)
    ids = route_result.expert_ids.cpu().tolist()
    weights = route_result.weights.cpu().tolist()

    cache = FP8ExpertCache(
        root,
        ram_limit_bytes=int(ram_gb * 1024**3),
        vram_limit_bytes=int(vram_gb * 1024**3),
        device=device,
    )
    routed = _run_routed_experts(root, moe_in, cache, ids, weights)
    shared = _run_shared_expert(root, moe_in)
    final = residual + routed + shared

    print(f"Token ID: {token_id}")
    print("Layer type: linear_attention")
    print("Router experts:", ids)
    print("Router weights:", [round(float(w), 8) for w in weights])
    print(f"Input norm: {torch.linalg.vector_norm(x0).item():.6f}")
    print(f"Attention output norm: {torch.linalg.vector_norm(mixer_out).item():.6f}")
    print(f"Routed MoE norm: {torch.linalg.vector_norm(routed).item():.6f}")
    print(f"Shared MoE norm: {torch.linalg.vector_norm(shared).item():.6f}")
    print(f"Layer 0 output norm: {torch.linalg.vector_norm(final).item():.6f}")
    print(f"Layer 0 output mean: {final.mean().item():.6f}")
    print(f"Layer 0 output std: {final.std().item():.6f}")
    print("Cache:", cache.snapshot())
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.6 Layer 0 single-token forward")
    parser.add_argument("root", type=Path)
    parser.add_argument("--token-id", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--ram-gb", type=float, default=2.0)
    parser.add_argument("--vram-gb", type=float, default=3.0)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    start = perf_counter()
    run_layer0(args.root.resolve(), args.token_id, args.device, args.ram_gb, args.vram_gb)
    if args.device == "cuda":
        torch.cuda.synchronize()
    print(f"Total time: {(perf_counter() - start) * 1000.0:.3f} ms")


if __name__ == "__main__":
    main()
