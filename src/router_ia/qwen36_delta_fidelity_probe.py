from __future__ import annotations

"""Standalone Gated DeltaNet fidelity probe."""

import argparse
import gc
from pathlib import Path

import torch
import torch.nn.functional as F

from . import qwen36_attention_cache as ours
from . import qwen36_40layer_loop as base
from . import qwen36_cached_loop as cached
from .qwen36_gated_norm_probe import gated_rmsnorm
from .qwen36_op_probe import rmsnorm

DEFAULT_SEED = 1234
DEFAULT_TOKENS = 6
DEFAULT_TOLERANCE = 1e-5


def _projection(root: Path, prefix: str, device: str) -> torch.Tensor:
    return cached._cached_load_projection(root, prefix, device)


def _reference_step(root: Path, layer: int, x: torch.Tensor, linear_state: torch.Tensor, conv_state: torch.Tensor, device: str):
    prefix = base.layer_prefix(layer)
    input_norm = base.load_layer_weight(root, layer, "input_layernorm.weight", device)
    h = rmsnorm(x, input_norm)
    compute_dtype = torch.float16 if device == "cuda" else torch.float32
    h_compute = h.to(dtype=compute_dtype)
    qkv_w = _projection(root, prefix + "linear_attn.in_proj_qkv", device)
    mixed = F.linear(h_compute.to(dtype=qkv_w.dtype), qkv_w).reshape(1, base.LINEAR_KEY_DIM * 2 + base.LINEAR_VALUE_DIM)
    conv_weight = base.load_layer_weight(root, layer, "linear_attn.conv1d.weight", device)
    try:
        conv_bias = base.load_layer_weight(root, layer, "linear_attn.conv1d.bias", device)
    except KeyError:
        conv_bias = None
    current = mixed.to(dtype=conv_weight.dtype).reshape(1, -1, 1)
    history = torch.cat((conv_state, current), dim=-1)
    new_conv_state = history[:, :, -ours.LINEAR_CONV_STATE:].detach()
    conv_out = F.conv1d(history, conv_weight, bias=conv_bias, padding=0, groups=ours.LINEAR_CONV_DIM)
    conv_out = F.silu(conv_out[:, :, -1:])[:, :, 0].to(dtype=mixed.dtype)
    q, k, v = torch.split(conv_out, [base.LINEAR_KEY_DIM, base.LINEAR_KEY_DIM, base.LINEAR_VALUE_DIM], dim=-1)
    q = q.reshape(1, base.LINEAR_NUM_K_HEADS, 128).repeat_interleave(2, dim=1)
    k = k.reshape(1, base.LINEAR_NUM_K_HEADS, 128).repeat_interleave(2, dim=1)
    v = v.reshape(1, base.LINEAR_NUM_V_HEADS, 128)
    a_w = _projection(root, prefix + "linear_attn.in_proj_a", device)
    b_w = _projection(root, prefix + "linear_attn.in_proj_b", device)
    a_log = base.load_layer_weight(root, layer, "linear_attn.A_log", device).float().reshape(1, base.LINEAR_NUM_V_HEADS)
    dt_bias = base.load_layer_weight(root, layer, "linear_attn.dt_bias", device).float().reshape(1, base.LINEAR_NUM_V_HEADS)
    a_raw = F.linear(h_compute.to(dtype=a_w.dtype), a_w).reshape(1, base.LINEAR_NUM_V_HEADS).float()
    b_raw = F.linear(h_compute.to(dtype=b_w.dtype), b_w).reshape(1, base.LINEAR_NUM_V_HEADS).float()
    beta = torch.sigmoid(b_raw)
    g = -torch.exp(a_log) * F.softplus(a_raw + dt_bias)
    decay = torch.exp(g)
    qn = ours._l2norm(q.float()) * (128 ** -0.5)
    kn = ours._l2norm(k.float())
    linear_state = linear_state * decay.unsqueeze(-1).unsqueeze(-1)
    retrieved = (linear_state * kn.unsqueeze(-1)).sum(dim=-2)
    delta = (v.float() - retrieved) * beta.unsqueeze(-1)
    linear_state = linear_state + kn.unsqueeze(-1) * delta.unsqueeze(-2)
    attn = (linear_state * qn.unsqueeze(-1)).sum(dim=-2)
    z_w = _projection(root, prefix + "linear_attn.in_proj_z", device)
    z = F.linear(h_compute.to(dtype=z_w.dtype), z_w).reshape(1, base.LINEAR_NUM_V_HEADS, 128)
    norm_w = base.load_layer_weight(root, layer, "linear_attn.norm.weight", device)
    gated, _, _ = gated_rmsnorm(attn, z, norm_w)
    out_w = _projection(root, prefix + "linear_attn.out_proj", device)
    gated_compute = gated.reshape(1, base.LINEAR_VALUE_DIM).to(dtype=out_w.dtype if device == "cuda" else compute_dtype)
    projected = F.linear(gated_compute, out_w).float()
    residual = x.reshape(1, base.HIDDEN).float() + projected
    return residual, linear_state.detach(), new_conv_state


def _diff(a: torch.Tensor, b: torch.Tensor):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    if a.numel() != b.numel():
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    d = (a - b).abs()
    return float(d.max().item()), float(d.mean().item()), float(torch.linalg.vector_norm(a - b).item() / max(torch.linalg.vector_norm(a).item(), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone Qwen3.6 Gated DeltaNet fidelity probe")
    parser.add_argument("root", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    if not 0 <= args.layer < base.DEFAULT_LAYERS:
        raise SystemExit(f"--layer must be in [0, {base.DEFAULT_LAYERS - 1}]")
    if args.tokens < 1:
        raise SystemExit("--tokens must be >= 1")
    root = args.root.resolve()
    if base.attention_type(root, args.layer) != "linear_attention":
        raise SystemExit(f"Layer {args.layer} is not a linear-attention/Gated DeltaNet layer")
    torch.manual_seed(args.seed)
    hidden_stream = torch.randn(args.tokens, base.HIDDEN, dtype=torch.float32)
    state = ours.state_for(root, args.device)
    state.reset()
    ours.activate(root, state)
    ref_state = torch.zeros(1, base.LINEAR_NUM_V_HEADS, 128, 128, device=args.device, dtype=torch.float32)
    conv_dtype = torch.float16 if args.device == "cuda" else torch.float32
    ref_conv = torch.zeros(1, ours.LINEAR_CONV_DIM, ours.LINEAR_CONV_STATE, device=args.device, dtype=conv_dtype)
    print("op=gated-deltanet-fidelity")
    print(f"layer={args.layer}")
    print(f"tokens={args.tokens}")
    print(f"device={args.device}")
    print(f"seed={args.seed}")
    print(f"tolerance={args.tolerance:g}")
    all_ok = True
    try:
        for index in range(args.tokens):
            x = hidden_stream[index:index + 1].to(args.device)
            ref_out, ref_state, ref_conv = _reference_step(root, args.layer, x, ref_state, ref_conv, args.device)
            got_out = ours._linear_stateful(root, args.layer, x, args.device)
            got_state = state.linear_states[int(args.layer)].detach().float()
            got_conv = state.linear_conv_states[int(args.layer)].detach().float()
            output_diff = _diff(ref_out, got_out)
            state_diff = _diff(ref_state, got_state)
            conv_diff = _diff(ref_conv, got_conv)
            step_ok = output_diff[0] <= args.tolerance and state_diff[0] <= args.tolerance and conv_diff[0] <= args.tolerance
            all_ok &= step_ok
            print(f"\n[token {index:02d}] status={'PASS' if step_ok else 'FAIL'}")
            print(f"  output     max_abs={output_diff[0]:.6g} | mean_abs={output_diff[1]:.6g} | rel={output_diff[2]:.6g}")
            print(f"  recurrent  max_abs={state_diff[0]:.6g} | mean_abs={state_diff[1]:.6g} | rel={state_diff[2]:.6g}")
            print(f"  conv_state max_abs={conv_diff[0]:.6g} | mean_abs={conv_diff[1]:.6g} | rel={conv_diff[2]:.6g}")
            state.tokens_seen += 1
            del x, ref_out, got_out
    finally:
        ours.deactivate(root)
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
    print("\n=== RESULT ===")
    print(f"status={'PASS' if all_ok else 'FAIL'}")
    if not all_ok:
        print("A difference above tolerance means the stateful Gated DeltaNet path still diverges from the reference equations.")
        raise SystemExit(2)
    print("Gated DeltaNet recurrent and causal-convolution states match the independent reference within tolerance.")


if __name__ == "__main__":
    main()
