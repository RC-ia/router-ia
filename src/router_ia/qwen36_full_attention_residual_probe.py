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
    if ref.shape != got.shape:
        print(f"{name:<28} SHAPE_MISMATCH ref={tuple(ref.shape)} got={tuple(got.shape)}")
        return False
    diff = (ref - got).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    ref_norm = float(torch.linalg.vector_norm(ref).item())
    rel = max_abs / max(ref_norm, 1e-12)
    cosine = float(F.cosine_similarity(ref.reshape(1, -1), got.reshape(1, -1), dim=-1).item())
    ok = max_abs <= tolerance
    print(
        f"{name:<28} {'PASS' if ok else 'FAIL'} "
        f"max_abs={max_abs:.8g} mean_abs={mean_abs:.8g} rel={rel:.8g} cosine={cosine:.9f}"
    )
    return ok


def compare_scores(ref: torch.Tensor, got: torch.Tensor, tolerance: float) -> bool:
    """Compare attention scores while diagnosing softmax-invariant row offsets."""
    ref = ref.float()
    got = got.float()
    if ref.shape != got.shape:
        print(f"{'scores':<28} SHAPE_MISMATCH ref={tuple(ref.shape)} got={tuple(got.shape)}")
        return False

    print("scores diagnostics:")
    raw_ok = compare("scores", ref, got, tolerance)

    ref_centered = ref - ref.mean(dim=-1, keepdim=True)
    got_centered = got - got.mean(dim=-1, keepdim=True)
    centered_ok = compare("scores_centered", ref_centered, got_centered, tolerance)

    delta = got - ref
    row_offset = delta.mean(dim=-1, keepdim=True)
    offset_residual = delta - row_offset
    offset_max = float(row_offset.abs().max().item())
    offset_mean = float(row_offset.abs().mean().item())
    residual_max = float(offset_residual.abs().max().item())
    residual_mean = float(offset_residual.abs().mean().item())
    print(
        f"{'score_row_offset':<28} max_abs={offset_max:.8g} mean_abs={offset_mean:.8g} "
        f"residual_max={residual_max:.8g} residual_mean={residual_mean:.8g}"
    )

    if centered_ok and residual_max <= tolerance:
        print("score interpretation: differences are only softmax-invariant row offsets")
    elif raw_ok:
        print("score interpretation: raw scores match")
    else:
        print("score interpretation: non-offset score difference detected")
    return raw_ok


def apply_rope(q: torch.Tensor, k: torch.Tensor, position: int):
    inv_freq = 1.0 / (
        ROPE_THETA ** (torch.arange(0, ROPE_DIM, 2, device=q.device, dtype=torch.float32) / ROPE_DIM)
    )
    angles = float(position) * inv_freq
    emb = torch.cat((angles, angles), dim=-1)
    cos = emb.cos().to(q.dtype).view(1, 1, 1, ROPE_DIM)
    sin = emb.sin().to(q.dtype).view(1, 1, 1, ROPE_DIM)

    def rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    q_rot, q_pass = q[..., :ROPE_DIM], q[..., ROPE_DIM:]
    k_rot, k_pass = k[..., :ROPE_DIM], k[..., ROPE_DIM:]
    q_rot = q_rot * cos + rotate_half(q_rot) * sin
    k_rot = k_rot * cos + rotate_half(k_rot) * sin
    return torch.cat((q_rot, q_pass), -1), torch.cat((k_rot, k_pass), -1)


def reference_token(root: Path, layer: int, x: torch.Tensor, position: int, visible_k, visible_v, device: str):
    prefix = base.layer_prefix(layer)
    input_norm = base.load_layer_weight(root, layer, "input_layernorm.weight", device)
    q_w = cached._cached_load_projection(root, prefix + "self_attn.q_proj", device)
    k_w = cached._cached_load_projection(root, prefix + "self_attn.k_proj", device)
    v_w = cached._cached_load_projection(root, prefix + "self_attn.v_proj", device)
    q_norm_w = base.load_layer_weight(root, layer, "self_attn.q_norm.weight", device)
    k_norm_w = base.load_layer_weight(root, layer, "self_attn.k_norm.weight", device)
    out_w = cached._cached_load_projection(root, prefix + "self_attn.o_proj", device)

    h = rmsnorm(x, input_norm)
    q_gate_raw = F.linear(h.to(dtype=q_w.dtype), q_w)
    q_gate = q_gate_raw.reshape(1, base.FULL_NUM_HEADS, base.FULL_HEAD_DIM * 2)
    q, gate = torch.chunk(q_gate, 2, dim=-1)
    k_raw = F.linear(h.to(dtype=k_w.dtype), k_w)
    v_raw = F.linear(h.to(dtype=v_w.dtype), v_w)
    k = k_raw.reshape(1, base.FULL_NUM_KV_HEADS, base.FULL_HEAD_DIM)
    v = v_raw.reshape(1, base.FULL_NUM_KV_HEADS, base.FULL_HEAD_DIM)
    qn = rmsnorm(q, q_norm_w).float().unsqueeze(2)
    kn = rmsnorm(k, k_norm_w).float().unsqueeze(2)
    qr, kr = apply_rope(qn, kn, position)
    k_expanded = visible_k.repeat_interleave(base.FULL_NUM_KV_GROUPS, dim=1).float()
    v_expanded = visible_v.repeat_interleave(base.FULL_NUM_KV_GROUPS, dim=1).float()
    q_now = qr.squeeze(2)
    scores = torch.einsum("bhd,bhld->bhl", q_now, k_expanded) * (base.FULL_HEAD_DIM ** -0.5)
    weights = torch.softmax(scores, dim=-1)
    attn_raw = torch.einsum("bhl,bhld->bhd", weights, v_expanded)
    attn_gated = attn_raw * torch.sigmoid(gate.float())
    projected = F.linear(attn_gated.reshape(1, base.FULL_Q_DIM).to(dtype=out_w.dtype), out_w).float()
    residual = x.float().reshape(1, base.HIDDEN) + projected
    return {
        "h": h,
        "q_gate_raw": q_gate_raw,
        "q_gate": q_gate,
        "q": q,
        "gate": gate,
        "k_raw": k_raw,
        "k": k,
        "v_raw": v_raw,
        "v": v,
        "q_norm": qn,
        "k_norm": kn,
        "q_rope": qr,
        "k_rope": kr,
        "q_now": q_now,
        "k_expanded": k_expanded,
        "v_expanded": v_expanded,
        "scores": scores,
        "weights": weights,
        "attn_raw": attn_raw,
        "attn_gated": attn_gated,
        "projected": projected,
        "residual": residual,
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

    def norm(input, weight, eps=1e-6):
        out = orig_norm(input, weight)
        captured["norm"].append(out.detach().clone())
        return out

    def rope(q, k, position):
        out = orig_rope(q, k, position)
        captured["rope"].append((out[0].detach().clone(), out[1].detach().clone()))
        return out

    def einsum(equation, *operands, **kwargs):
        out = orig_einsum(equation, *operands, **kwargs)
        captured["einsum"].append(
            {
                "equation": equation,
                "operands": tuple(
                    op.detach().clone() if torch.is_tensor(op) else op for op in operands
                ),
                "out": out.detach().clone(),
            }
        )
        return out

    def softmax(input, dim=-1, _stacklevel=3, dtype=None):
        kwargs = {"dim": dim}
        if dtype is not None:
            kwargs["dtype"] = dtype
        out = orig_softmax(input, **kwargs)
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

            def invoke():
                return cache._full_stateful(root, layer, x, device)

            got, cap = capture_runtime(invoke)
            ref = reference_token(root, layer, x, position, full_k[:, :, :position + 1], full_v[:, :, :position + 1], device)

            print("--- projections ---")
            projection_specs = (
                ("q_gate", "q_gate_raw", 0),
                ("k", "k_raw", 1),
                ("v", "v_raw", 2),
                ("o_proj", "projected", 3),
            )
            for name, ref_name, idx in projection_specs:
                if idx < len(cap["linear"]):
                    all_pass &= compare(name, ref[ref_name], cap["linear"][idx], tolerance)
                else:
                    print(f"{name:<28} MISSING")
                    all_pass = False

            print("--- norms / rope ---")
            for name, ref_name, idx in (("input_norm", "h", 0), ("q_norm", "q_norm", 1), ("k_norm", "k_norm", 2)):
                if idx < len(cap["norm"]):
                    ref_value = ref[ref_name]
                    if name in ("q_norm", "k_norm"):
                        ref_value = ref_value.squeeze(2)
                    all_pass &= compare(name, ref_value, cap["norm"][idx], tolerance)
                else:
                    print(f"{name:<28} MISSING")
                    all_pass = False
            if cap["rope"]:
                all_pass &= compare("q_after_rope", ref["q_rope"], cap["rope"][0][0], tolerance)
                all_pass &= compare("k_after_rope", ref["k_rope"], cap["rope"][0][1], tolerance)
            else:
                print("q_after_rope                 MISSING")
                print("k_after_rope                 MISSING")
                all_pass = False

            print("--- attention operands ---")
            score_op = find_einsum(cap, "bhd,bhld->bhl", 0)
            attn_op = find_einsum(cap, "bhl,bhld->bhd", 0)
            if score_op is None or attn_op is None or not cap["softmax"]:
                print("attention intermediates       MISSING")
                all_pass = False
            else:
                runtime_q, runtime_k = score_op["operands"]
                runtime_score = score_op["out"]
                runtime_weights_input = cap["softmax"][0]["input"]
                runtime_weights = cap["softmax"][0]["out"]
                runtime_attn_weights, runtime_v = attn_op["operands"]
                runtime_attn = attn_op["out"]

                all_pass &= compare("score_q_operand", ref["q_now"], runtime_q, tolerance)
                all_pass &= compare("score_k_operand", ref["k_expanded"], runtime_k, tolerance)
                all_pass &= compare("score_v_operand", ref["v_expanded"], runtime_v, tolerance)
                all_pass &= compare("score_einsum_raw", ref["scores"] / (base.FULL_HEAD_DIM ** -0.5), runtime_score / (base.FULL_HEAD_DIM ** -0.5), tolerance)

                reconstructed_score = torch.einsum("bhd,bhld->bhl", runtime_q, runtime_k)
                reconstructed_score = reconstructed_score * (base.FULL_HEAD_DIM ** -0.5)
                all_pass &= compare("score_reconstructed", runtime_score, reconstructed_score, tolerance)
                all_pass &= compare("score_vs_softmax_input", runtime_score, runtime_weights_input, tolerance)
                all_pass &= compare("softmax", ref["weights"], runtime_weights, tolerance)

                all_pass &= compare("attn_weights_operand", ref["weights"], runtime_attn_weights, tolerance)
                all_pass &= compare("attn_v_operand", ref["v_expanded"], runtime_v, tolerance)
                all_pass &= compare("attn_raw", ref["attn_raw"], runtime_attn, tolerance)

            q_gate = cap["linear"][0]
            q_runtime = q_gate.reshape(1, base.FULL_NUM_HEADS, base.FULL_HEAD_DIM * 2)
            q_runtime, gate_runtime = torch.chunk(q_runtime, 2, dim=-1)
            all_pass &= compare("q_pre_rope", ref["q"], q_runtime, tolerance)
            all_pass &= compare("gate", ref["gate"], gate_runtime, tolerance)
            if attn_op is not None:
                attn_runtime = attn_op["out"] * torch.sigmoid(gate_runtime.float())
                all_pass &= compare("attn_gated", ref["attn_gated"], attn_runtime, tolerance)

            print("--- output / residual ---")
            if len(cap["linear"]) >= 4:
                all_pass &= compare("projected", ref["projected"], cap["linear"][3], tolerance)
            else:
                print("projected                    MISSING")
                all_pass = False
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
