from __future__ import annotations

"""Break Qwen3.6 full-attention residual fidelity into intermediate tensors."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from . import qwen36_40layer_loop as base
from . import qwen36_attention_cache as cache
from . import qwen36_cached_loop as cached
from .qwen36_op_probe import rmsnorm

TOLERANCE = 1e-3
ROPE_THETA = 10_000_000.0
ROPE_DIM = int(base.FULL_HEAD_DIM * 0.25)


def compare(name: str, ref: torch.Tensor, got: torch.Tensor, tolerance: float) -> bool:
    ref = ref.float()
    got = got.float()
    delta = (got - ref).abs()
    max_abs = float(delta.max().item()) if delta.numel() else 0.0
    mean_abs = float(delta.mean().item()) if delta.numel() else 0.0
    ref_norm = float(torch.linalg.vector_norm(ref).item())
    rel = float(torch.linalg.vector_norm(got - ref).item() / max(ref_norm, 1e-12))
    cosine = float(F.cosine_similarity(ref.reshape(-1), got.reshape(-1), dim=0).item()) if ref.numel() else 1.0
    ok = max_abs <= tolerance
    print(f"{name:<28} {'PASS' if ok else 'FAIL'} max_abs={max_abs:.8g} mean_abs={mean_abs:.8g} rel={rel:.8g} cosine={cosine:.9f}")
    return ok


def _rope(position: int, device: torch.device, dtype: torch.dtype):
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, ROPE_DIM, 2, device=device, dtype=torch.float32) / ROPE_DIM))
    angles = float(position) * inv_freq
    emb = torch.cat((angles, angles), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, position: int):
    cos, sin = _rope(position, q.device, q.dtype)
    cos = cos.view(1, 1, 1, ROPE_DIM)
    sin = sin.view(1, 1, 1, ROPE_DIM)
    q_rot, q_pass = q[..., :ROPE_DIM], q[..., ROPE_DIM:]
    k_rot, k_pass = k[..., :ROPE_DIM], k[..., ROPE_DIM:]
    q_rot = q_rot * cos + _rotate_half(q_rot) * sin
    k_rot = k_rot * cos + _rotate_half(k_rot) * sin
    return torch.cat((q_rot, q_pass), dim=-1), torch.cat((k_rot, k_pass), dim=-1)


def reference_token(root: Path, layer: int, x0: torch.Tensor, position: int, full_k: torch.Tensor, full_v: torch.Tensor, device: str):
    prefix = base.layer_prefix(layer)
    input_norm = base.load_layer_weight(root, layer, "input_layernorm.weight", device)
    h = rmsnorm(x0, input_norm)
    compute_dtype = torch.float16 if device == "cuda" else torch.float32
    h_compute = h.to(dtype=compute_dtype)
    q_w = cached._cached_load_projection(root, prefix + "self_attn.q_proj", device)
    k_w = cached._cached_load_projection(root, prefix + "self_attn.k_proj", device)
    v_w = cached._cached_load_projection(root, prefix + "self_attn.v_proj", device)
    o_w = cached._cached_load_projection(root, prefix + "self_attn.o_proj", device)

    q_gate_raw = F.linear(h_compute.to(dtype=q_w.dtype), q_w)
    q_gate = q_gate_raw.reshape(1, base.FULL_NUM_HEADS, base.FULL_HEAD_DIM * 2)
    q, gate = torch.chunk(q_gate, 2, dim=-1)
    k_raw = F.linear(h_compute.to(dtype=k_w.dtype), k_w)
    v_raw = F.linear(h_compute.to(dtype=v_w.dtype), v_w)
    k = k_raw.reshape(1, base.FULL_NUM_KV_HEADS, base.FULL_HEAD_DIM)
    v = v_raw.reshape(1, base.FULL_NUM_KV_HEADS, base.FULL_HEAD_DIM)

    q_norm_w = base.load_layer_weight(root, layer, "self_attn.q_norm.weight", device)
    k_norm_w = base.load_layer_weight(root, layer, "self_attn.k_norm.weight", device)
    q_norm = rmsnorm(q, q_norm_w).float().unsqueeze(2)
    k_norm = rmsnorm(k, k_norm_w).float().unsqueeze(2)
    q_rope, k_token = apply_rope(q_norm, k_norm, position)
    visible_k = full_k
    visible_v = full_v
    k_expanded = visible_k.repeat_interleave(base.FULL_NUM_KV_GROUPS, dim=1).float()
    v_expanded = visible_v.repeat_interleave(base.FULL_NUM_KV_GROUPS, dim=1).float()
    q_now = q_rope.squeeze(2)
    score_raw = torch.einsum("bhd,bhld->bhl", q_now, k_expanded)
    scores = score_raw * (base.FULL_HEAD_DIM ** -0.5)
    weights = torch.softmax(scores, dim=-1)
    attn_raw = torch.einsum("bhl,bhld->bhd", weights, v_expanded)
    attn_gated = attn_raw * torch.sigmoid(gate.float())
    attn_flat = attn_gated.reshape(1, base.FULL_Q_DIM).to(dtype=compute_dtype)
    projected = F.linear(attn_flat.to(dtype=o_w.dtype), o_w).float()
    residual = x0.reshape(1, base.HIDDEN).float() + projected
    return {
        "q_gate_raw": q_gate_raw, "k_raw": k_raw, "v_raw": v_raw, "projected": projected,
        "h": h, "q_norm": q_norm, "k_norm": k_norm, "q_rope": q_rope, "k_rope": k_token,
        "q": q, "gate": gate, "q_now": q_now, "k_expanded": k_expanded, "v_expanded": v_expanded,
        "score_raw": score_raw, "scores": scores, "weights": weights, "attn_raw": attn_raw,
        "attn_gated": attn_gated, "residual": residual,
    }


def capture_runtime(fn):
    captured = {"linear": [], "norm": [], "rope": [], "einsum": [], "softmax": []}
    orig_linear = cache.F.linear
    orig_norm = cache.rmsnorm
    orig_rope = cache._apply_rope
    orig_einsum = cache.torch.einsum
    orig_softmax = cache.torch.softmax

    def linear(input, weight, bias=None):
        out = orig_linear(input, weight, bias)
        captured["linear"].append(out.detach().clone())
        return out

    def norm(x, weight):
        out = orig_norm(x, weight)
        captured["norm"].append(out.detach().clone())
        return out

    def rope(q, k, position):
        out = orig_rope(q, k, position)
        captured["rope"].append((out[0].detach().clone(), out[1].detach().clone()))
        return out

    def einsum(equation, *operands, **kwargs):
        out = orig_einsum(equation, *operands, **kwargs)
        captured["einsum"].append({
            "equation": equation,
            "operands": [op.detach().clone() if torch.is_tensor(op) else op for op in operands],
            "out": out.detach().clone(),
        })
        return out

    def softmax(input, dim=None, *args, **kwargs):
        out = orig_softmax(input, dim=dim, *args, **kwargs)
        captured["softmax"].append({"input": input.detach().clone(), "out": out.detach().clone()})
        return out

    cache.F.linear = linear
    cache.rmsnorm = norm
    cache._apply_rope = rope
    cache.torch.einsum = einsum
    cache.torch.softmax = softmax
    try:
        return fn(), captured
    finally:
        cache.F.linear = orig_linear
        cache.rmsnorm = orig_norm
        cache._apply_rope = orig_rope
        cache.torch.einsum = orig_einsum
        cache.torch.softmax = orig_softmax


def find_einsum(captured, equation: str, occurrence: int = 0):
    matches = [item for item in captured["einsum"] if item["equation"] == equation]
    return matches[occurrence] if occurrence < len(matches) else None


def run(root: Path, layer: int, tokens: int, device: str, seed: int, tolerance: float) -> bool:
    torch.manual_seed(seed)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    hidden = [torch.randn(1, base.HIDDEN, device=device, dtype=dtype) for _ in range(tokens)]

    prefix = base.layer_prefix(layer)
    k_w = cached._cached_load_projection(root, prefix + "self_attn.k_proj", device)
    v_w = cached._cached_load_projection(root, prefix + "self_attn.v_proj", device)
    input_norm = base.load_layer_weight(root, layer, "input_layernorm.weight", device)
    k_norm_w = base.load_layer_weight(root, layer, "self_attn.k_norm.weight", device)
    ref_k, ref_v = [], []
    for x in hidden:
        h = rmsnorm(x, input_norm)
        k = F.linear(h.to(dtype=k_w.dtype), k_w).reshape(1, base.FULL_NUM_KV_HEADS, base.FULL_HEAD_DIM)
        v = F.linear(h.to(dtype=v_w.dtype), v_w).reshape(1, base.FULL_NUM_KV_HEADS, base.FULL_HEAD_DIM)
        k = rmsnorm(k, k_norm_w).float().unsqueeze(2)
        k = apply_rope(torch.zeros(1, base.FULL_NUM_HEADS, 1, base.FULL_HEAD_DIM, device=device), k, len(ref_k))[1]
        ref_k.append(k)
        ref_v.append(v.float().unsqueeze(2))
    full_k = torch.cat(ref_k, dim=2)
    full_v = torch.cat(ref_v, dim=2)

    state = cache.AttentionState()
    state.bind(device)
    cache.activate(root, state)
    all_pass = True
    try:
        for position, x in enumerate(hidden):
            print()
            print(f"=== TOKEN {position} ===")
            state.tokens_seen = position
            got, cap = capture_runtime(lambda: cache._full_stateful(root, layer, x, device))
            ref = reference_token(root, layer, x, position, full_k[:, :, :position + 1], full_v[:, :, :position + 1], device)

            print("--- projections ---")
            for name, ref_name, idx in (("q_gate", "q_gate_raw", 0), ("k", "k_raw", 1), ("v", "v_raw", 2), ("o_proj", "projected", 3)):
                all_pass &= compare(name, ref[ref_name], cap["linear"][idx], tolerance)

            print("--- norms / rope ---")
            for name, ref_name, idx in (("input_norm", "h", 0), ("q_norm", "q_norm", 1), ("k_norm", "k_norm", 2)):
                ref_value = ref[ref_name].squeeze(2) if name in ("q_norm", "k_norm") else ref[ref_name]
                all_pass &= compare(name, ref_value, cap["norm"][idx], tolerance)
            all_pass &= compare("q_after_rope", ref["q_rope"], cap["rope"][0][0], tolerance)
            all_pass &= compare("k_after_rope", ref["k_rope"], cap["rope"][0][1], tolerance)

            print("--- attention operands ---")
            score_op = find_einsum(cap, "bhd,bhld->bhl", 0)
            attn_op = find_einsum(cap, "bhl,bhld->bhd", 0)
            soft = cap["softmax"][0] if cap["softmax"] else None
            if score_op is None or attn_op is None or soft is None:
                print("attention intermediates       MISSING")
                all_pass = False
            else:
                runtime_q, runtime_k = score_op["operands"]
                runtime_score_raw = score_op["out"]
                runtime_weights_input = soft["input"]
                runtime_weights = soft["out"]
                runtime_attn_weights, runtime_v = attn_op["operands"]
                runtime_attn = attn_op["out"]
                scale = base.FULL_HEAD_DIM ** -0.5
                runtime_score = runtime_score_raw * scale

                all_pass &= compare("score_q_operand", ref["q_now"], runtime_q, tolerance)
                all_pass &= compare("score_k_operand", ref["k_expanded"], runtime_k, tolerance)
                all_pass &= compare("score_v_operand", ref["v_expanded"], runtime_v, tolerance)
                all_pass &= compare("score_einsum_raw", ref["score_raw"], runtime_score_raw, tolerance)
                reconstructed_raw = torch.einsum("bhd,bhld->bhl", runtime_q, runtime_k)
                all_pass &= compare("score_reconstructed_raw", runtime_score_raw, reconstructed_raw, tolerance)
                all_pass &= compare("score_scaled", ref["scores"], runtime_score, tolerance)
                all_pass &= compare("score_vs_softmax_input", runtime_score, runtime_weights_input, tolerance)
                all_pass &= compare("softmax", ref["weights"], runtime_weights, tolerance)
                all_pass &= compare("attn_weights_operand", ref["weights"], runtime_attn_weights, tolerance)
                all_pass &= compare("attn_v_operand", ref["v_expanded"], runtime_v, tolerance)
                all_pass &= compare("attn_raw", ref["attn_raw"], runtime_attn, tolerance)

            print("--- output / residual ---")
            q_gate = cap["linear"][0].reshape(1, base.FULL_NUM_HEADS, base.FULL_HEAD_DIM * 2)
            q_runtime, gate_runtime = torch.chunk(q_gate, 2, dim=-1)
            all_pass &= compare("q_pre_rope", ref["q"], q_runtime, tolerance)
            all_pass &= compare("gate", ref["gate"], gate_runtime, tolerance)
            if attn_op is not None:
                attn_runtime = attn_op["out"] * torch.sigmoid(gate_runtime.float())
                all_pass &= compare("attn_gated", ref["attn_gated"], attn_runtime, tolerance)
            all_pass &= compare("projected", ref["projected"], cap["linear"][3], tolerance)
            all_pass &= compare("residual", ref["residual"], got, tolerance)
    finally:
        cache.deactivate(root)

    print()
    print(f"RESULT status={'PASS' if all_pass else 'FAIL'}")
    return all_pass


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--tokens", type=int, default=4)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = p.parse_args()
    return 0 if run(args.root.resolve(), args.layer, args.tokens, args.device, args.seed, args.tolerance) else 1


if __name__ == "__main__":
    raise SystemExit(main())
