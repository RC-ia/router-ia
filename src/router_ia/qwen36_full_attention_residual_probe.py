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
    q_gate = F.linear(h.to(dtype=q_w.dtype), q_w).reshape(1, base.FULL_NUM_HEADS, base.FULL_HEAD_DIM * 2)
    q, gate = torch.chunk(q_gate, 2, dim=-1)
    k = F.linear(h.to(dtype=k_w.dtype), k_w).reshape(1, base.FULL_NUM_KV_HEADS, base.FULL_HEAD_DIM)
    v = F.linear(h.to(dtype=v_w.dtype), v_w).reshape(1, base.FULL_NUM_KV_HEADS, base.FULL_HEAD_DIM)
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
        "q_gate": q_gate,
        "q": q,
        "gate": gate,
        "k": k,
        "v": v,
        "q_norm": qn,
        "k_norm": kn,
        "q_rope": qr,
        "k_rope": kr,
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
        # Runtime rmsnorm currently accepts exactly (x, weight). Keep the
        # optional eps in the wrapper only so it is compatible with callers
        # that use the conventional three-argument signature.
        out = orig_norm(input, weight)
        captured["norm"].append(out.detach().clone())
        return out

    def rope(q, k, position):
        out = orig_rope(q, k, position)
        captured["rope"].append((out[0].detach().clone(), out[1].detach().clone()))
        return out

    def einsum(equation, *operands, **kwargs):
        out = orig_einsum(equation, *operands, **kwargs)
        captured["einsum"].append((equation, out.detach().clone()))
        return out

    def softmax(input, dim=-1, _stacklevel=3, dtype=None):
        kwargs = {"dim": dim}
        if dtype is not None:
            kwargs["dtype"] = dtype

        out = orig_softmax(input, **kwargs)
        captured["softmax"].append(out.detach().clone())
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


def run(root: Path, layer: int, tokens: int, device: str, seed: int, tolerance: float) -> bool:
    torch.manual_seed(seed)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    hidden = [torch.randn(1, base.HIDDEN, device=device, dtype=dtype) for _ in range(tokens)]

    # Build the exact reference K/V once, so the residual comparison can focus
    # on the current-token Q/gate/attention/output path.
    prefix = base.layer_prefix(layer)
    k_w = cached._cached_load_projection(root, prefix + "self_attn.k_proj", device)
    v_w = cached._cached_load_projection(root, prefix + "self_attn.v_proj", device)
    input_norm = base.load_layer_weight(root, layer, "input_layernorm.weight", device)
    q_norm_w = base.load_layer_weight(root, layer, "self_attn.q_norm.weight", device)
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
            names = ["q_gate", "k", "v", "o_proj"]
            for i, name in enumerate(names):
                if i < len(cap["linear"]):
                    all_pass &= compare(name, ref[name] if name != "o_proj" else ref["projected"], cap["linear"][i], tolerance)
                else:
                    print(f"{name:<28} MISSING")
                    all_pass = False

            print("--- norms / rope ---")
            # Runtime rmsnorm calls: input h, q norm, k norm.
            for name, ref_name, idx in (("input_norm", "h", 0), ("q_norm", "q_norm", 1), ("k_norm", "k_norm", 2)):
                all_pass &= compare(name, ref[ref_name], cap["norm"][idx], tolerance)
            all_pass &= compare("q_after_rope", ref["q_rope"], cap["rope"][0][0], tolerance)
            all_pass &= compare("k_after_rope", ref["k_rope"], cap["rope"][0][1], tolerance)

            print("--- attention ---")
            all_pass &= compare("scores", ref["scores"], cap["einsum"][0][1], tolerance)
            all_pass &= compare("softmax", ref["weights"], cap["softmax"][0], tolerance)
            all_pass &= compare("attn_raw", ref["attn_raw"], cap["einsum"][1][1], tolerance)
            gate = cap["linear"][0][..., base.FULL_HEAD_DIM:]
            q_gate = cap["linear"][0]
            q_runtime = q_gate[..., :base.FULL_HEAD_DIM]
            gate_runtime = gate
            # q/gate are slices of the captured fused projection.
            all_pass &= compare("q_pre_rope", ref["q"], q_runtime, tolerance)
            all_pass &= compare("gate", ref["gate"], gate_runtime, tolerance)
            attn_runtime = cap["einsum"][1][1] * torch.sigmoid(gate_runtime.float())
            all_pass &= compare("attn_gated", ref["attn_gated"], attn_runtime, tolerance)

            print("--- output / residual ---")
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
