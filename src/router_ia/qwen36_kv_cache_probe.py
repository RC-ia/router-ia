from __future__ import annotations

"""Qwen3.6 full-attention KV-cache fidelity probe."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from . import qwen36_40layer_loop as base
from . import qwen36_attention_cache as cache
from .qwen36_op_probe import rmsnorm

TOLERANCE = 1e-3


def compare(name: str, ref: torch.Tensor, got: torch.Tensor, tolerance: float = TOLERANCE) -> bool:
    ref = ref.float()
    got = got.float()
    diff = (ref - got).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    ref_norm = float(torch.linalg.vector_norm(ref).item())
    got_norm = float(torch.linalg.vector_norm(got).item())
    rel = max_abs / max(ref_norm, 1e-12)
    cosine = float(F.cosine_similarity(ref.reshape(1, -1), got.reshape(1, -1), dim=-1).item())
    ok = max_abs <= tolerance
    print(
        f"{name:<24} {'PASS' if ok else 'FAIL'} "
        f"max_abs={max_abs:.8g} mean_abs={mean_abs:.8g} rel={rel:.8g} "
        f"cosine={cosine:.9f} ref_norm={ref_norm:.8g} got_norm={got_norm:.8g}"
    )
    return ok


def discover_full_attention_layers(root: Path) -> list[int]:
    layers = []
    for layer in range(base.DEFAULT_LAYERS):
        if base.attention_type(root, layer) == "full_attention":
            layers.append(layer)
    return layers


def reference_full_attention(root: Path, layer: int, hidden_states: list[torch.Tensor], device: str):
    prefix = base.layer_prefix(layer)
    input_norm = base.load_layer_weight(root, layer, "input_layernorm.weight", device)
    q_w = cache._projection(root, prefix + "self_attn.q_proj", device)
    k_w = cache._projection(root, prefix + "self_attn.k_proj", device)
    v_w = cache._projection(root, prefix + "self_attn.v_proj", device)
    q_norm_w = base.load_layer_weight(root, layer, "self_attn.q_norm.weight", device)
    k_norm_w = base.load_layer_weight(root, layer, "self_attn.k_norm.weight", device)
    out_w = cache._projection(root, prefix + "self_attn.o_proj", device)

    q_tokens, k_tokens, v_tokens, gates = [], [], [], []
    for position, x in enumerate(hidden_states):
        h = rmsnorm(x, input_norm)
        q_gate = F.linear(h.to(dtype=q_w.dtype), q_w).reshape(1, base.FULL_NUM_HEADS, base.FULL_HEAD_DIM * 2)
        q, gate = torch.chunk(q_gate, 2, dim=-1)
        k = F.linear(h.to(dtype=k_w.dtype), k_w).reshape(1, base.FULL_NUM_KV_HEADS, base.FULL_HEAD_DIM)
        v = F.linear(h.to(dtype=v_w.dtype), v_w).reshape(1, base.FULL_NUM_KV_HEADS, base.FULL_HEAD_DIM)
        q = rmsnorm(q, q_norm_w).float().unsqueeze(2)
        k = rmsnorm(k, k_norm_w).float().unsqueeze(2)
        v = v.float().unsqueeze(2)
        q, k = cache._apply_rope(q, k, position)
        q_tokens.append(q)
        k_tokens.append(k)
        v_tokens.append(v)
        gates.append(gate)

    full_k = torch.cat(k_tokens, dim=2)
    full_v = torch.cat(v_tokens, dim=2)
    outputs = []
    for position, q_token in enumerate(q_tokens):
        k_visible = full_k[:, :, :position + 1]
        v_visible = full_v[:, :, :position + 1]
        k_expanded = k_visible.repeat_interleave(base.FULL_NUM_KV_GROUPS, dim=1).float()
        v_expanded = v_visible.repeat_interleave(base.FULL_NUM_KV_GROUPS, dim=1).float()
        q_now = q_token.squeeze(2)
        scores = torch.einsum("bhd,bhld->bhl", q_now, k_expanded) * (base.FULL_HEAD_DIM ** -0.5)
        weights = torch.softmax(scores, dim=-1)
        attn = torch.einsum("bhl,bhld->bhd", weights, v_expanded)
        attn = attn * torch.sigmoid(gates[position].float())
        attn_flat = attn.reshape(1, base.FULL_Q_DIM)
        projected = F.linear(attn_flat.to(dtype=out_w.dtype), out_w).float()
        outputs.append(projected)

    return {"keys": full_k, "values": full_v, "outputs": outputs}


def run_probe(root: Path, layer: int, num_tokens: int, device: str, seed: int, tolerance: float) -> bool:
    torch.manual_seed(seed)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if base.attention_type(root, layer) != "full_attention":
        raise RuntimeError(f"Layer {layer} is not full_attention: {base.attention_type(root, layer)}")

    print("op=kv-cache-fidelity")
    print(f"layer={layer}")
    print(f"tokens={num_tokens}")
    print(f"device={device}")
    print(f"seed={seed}")
    print(f"tolerance={tolerance}")
    print()

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    hidden_states = [torch.randn(1, base.HIDDEN, device=device, dtype=dtype) for _ in range(num_tokens)]
    reference = reference_full_attention(root, layer, hidden_states, device)

    print("=== REFERENCE ===")
    print(f"reference K shape={tuple(reference['keys'].shape)} dtype={reference['keys'].dtype}")
    print(f"reference V shape={tuple(reference['values'].shape)} dtype={reference['values'].dtype}")
    print()

    state = cache.AttentionState()
    state.bind(device)
    cache.activate(root, state)
    all_pass = True
    try:
        print("=== SEQUENTIAL CACHE ===")
        for position, x in enumerate(hidden_states):
            print()
            print(f"--- token {position} ---")
            state.tokens_seen = position
            got = cache._full_stateful(root, layer, x, device)
            got_k = state.full_keys[layer]
            got_v = state.full_values[layer]
            ref_k = reference["keys"][:, :, :position + 1]
            ref_v = reference["values"][:, :, :position + 1]
            all_pass &= compare("cached K", ref_k, got_k, tolerance)
            all_pass &= compare("cached V", ref_v, got_v, tolerance)
            all_pass &= compare("attention output", reference["outputs"][position], got, tolerance)
            expected_tokens = position + 1
            actual_tokens = int(got_k.shape[-2])
            ok_tokens = actual_tokens == expected_tokens
            print(f"{'cache length':<24}{'PASS' if ok_tokens else 'FAIL'} expected={expected_tokens} got={actual_tokens}")
            all_pass &= ok_tokens
            detached = not got_k.requires_grad and not got_v.requires_grad
            print(f"{'cache detached':<24}{'PASS' if detached else 'FAIL'}")
            all_pass &= detached
    finally:
        cache.deactivate(root)

    print()
    print("=== RESET ===")
    state.reset()
    reset_ok = (
        not state.full_keys and not state.full_values and
        not state.linear_states and not state.linear_conv_states and
        state.tokens_seen == 0
    )
    print(f"{'state.reset()':<24}{'PASS' if reset_ok else 'FAIL'} tokens_seen={state.tokens_seen} full_layers={len(state.full_keys)}")
    all_pass &= reset_ok

    print()
    print("=== STATE STATS ===")
    stats = state.snapshot()
    print(f"tokens_seen={stats['tokens_seen']}")
    print(f"full_layers_cached={stats['full_layers_cached']}")
    print(f"full_tokens={stats['full_tokens']}")
    print(f"full_bytes={stats['full_bytes']}")
    print()
    print("=== RESULT ===")
    print(f"status={'PASS' if all_pass else 'FAIL'}")
    return all_pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen3.6 full-attention KV-cache fidelity probe")
    parser.add_argument("root", type=Path)
    parser.add_argument("--layer", type=int, default=None, help="full-attention layer; defaults to the first detected one")
    parser.add_argument("--all-full", action="store_true", help="test every detected full-attention layer")
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = parser.parse_args()

    root = args.root.resolve()
    full_layers = discover_full_attention_layers(root)
    print(f"full_attention_layers={full_layers}")
    if not full_layers:
        raise SystemExit("No full-attention layers detected")

    if args.layer is not None:
        if args.layer not in full_layers:
            raise SystemExit(f"Layer {args.layer} is not full_attention; detected={full_layers}")
        layers = [args.layer]
    elif args.all_full:
        layers = full_layers
    else:
        layers = [full_layers[0]]

    print(f"selected_layers={layers}")
    print()
    all_pass = True
    for layer in layers:
        all_pass &= run_probe(root, layer, args.tokens, args.device, args.seed, args.tolerance)
        print()
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
