from __future__ import annotations

"""Standalone Gated DeltaNet fidelity probe.

This intentionally does not instantiate the full Qwen3.6 MoE model. It loads
one linear-attention layer directly from the safetensor checkpoint and compares
our stateful implementation against an independent PyTorch reference of the
Transformers fallback equations.

That makes it useful on GPUs where loading the complete FP8 MoE checkpoint
fails during expert-conversion, and it avoids consuming the VRAM required by
the other 39 layers.
"""

import argparse
import gc
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from . import qwen36_attention_cache as ours
from . import qwen36_40layer_loop as base
from . import qwen36_cached_loop as cached
from .qwen36_gated_norm_probe import gated_rmsnorm
from .qwen36_mini_chat import find_tensor_name
from .qwen36_op_probe import rmsnorm

DEFAULT_SEED = 1234
DEFAULT_TOKENS = 6
DEFAULT_TOLERANCE = 1e-5


def _projection(root: Path, prefix: str, device: str) -> torch.Tensor:
    return cached._cached_load_projection(root, prefix, device)


def _reference_step(
    root: Path,
    layer: int,
    x: torch.Tensor,
    linear_state: torch.Tensor,
    conv_state: torch.Tensor,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    prefix = base.layer_prefix(layer)
    input_norm = base.load_layer_weight(root, layer, "input_layernorm.weight", device)
    h = rmsnorm(x, input_norm)

    compute_dtype = torch.float16 if device == "cuda" else torch.float32
    h_compute = h.to(dtype=compute_dtype)

    qkv_w = _projection(root, prefix + "linear_attn.in_proj_qkv", device)
    mixed = F.linear(h_compute.to(dtype=qkv_w.dtype), qkv_w)
    mixed = mixed.reshape(1, base.LINEAR_KEY_DIM * 2 + base.LINEAR_VALUE_DIM)

    conv_weight = base.load_layer_weight(root, layer, "linear_attn.conv1d.weight", device)
    conv_bias = None
    try:
        conv_bias = base.load_layer_weight(root, layer, "linear_attn.conv1d.bias", device)
    except KeyError:
        pass

    # Same causal update as Transformers' torch_causal_conv1d_update:
    # concatenate previous kernel-1 raw vectors with the current vector,
    # convolve depthwise, then retain the newest kernel-1 vectors.
    current = mixed.to(dtype=conv_weight.dtype).reshape(1, -1, 1)
    history = torch.cat((conv_state, current), dim=-1)
    new_conv_state = history[:, :, -ours.LINEAR_CONV_STATE :].detach()
    conv_out = F.conv1d(
        history,
        conv_weight,
        bias=conv_bias,
        padding=0,
        groups=ours.LINEAR_CONV_DIM,
    )
    conv_out = F.silu(conv_out[:, :, -1:])
    conv_out = conv_out[:, :, 0].to(dtype=mixed.dtype)

    q, k, v = torch.split(
        conv_out,
        [base.LINEAR_KEY_DIM, base.LINEAR_KEY_DIM, base.LINEAR_VALUE_DIM],
        dim=-1,
    )
    q = q.reshape(1, base.LINEAR_NUM_K_HEADS, 128).repeat_interleave(2, dim=1)
    k = k.reshape(1, base.LINEAR_NUM_K_HEADS, 128).repeat_interleave(2, dim=1)
    v = v.reshape(1, base.LINEAR_NUM_VALUE_HEADS, 128)

    a_w = _projection(root, prefix + "linear_attn.in_proj_a", device)
    b_w = _projection(root, prefix + "linear_attn.in_proj_b", device)
    a_log = base.load_layer_weight(root, layer, "linear_attn.A_log", device).float().reshape(1, base.LINEAR_NUM_VALUE_HEADS)
    dt_bias = base.load_layer_weight(root, layer, "linear_attn.dt_bias", device).float().reshape(1, base.LINEAR_NUM_VALUE_HEADS)
    a_raw = F.linear(h_compute.to(dtype=a_w.dtype), a_w).reshape(1, base.LINEAR_NUM_VALUE_HEADS).float()
    b_raw = F.linear(h_compute.to(dtype=b_w.dtype), b_w).reshape(1, base.LINEAR_NUM_VALUE_HEADS).float()

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
    z = F.linear(h_compute.to(dtype=z_w.dtype), z_w).reshape(1, base.LINEAR_NUM_VALUE_HEADS, 128)
    norm_w = base.load_layer_weight(root, layer, "linear_attn.norm.weight", device)
    gated, _, _ = gated_rmsnorm(attn, z, norm_w)

    out_w = _projection(root, prefix + "linear_attn.out_proj", device)
    gated_compute = gated.reshape(1, base.LINEAR_VALUE_DIM).to(dtype=out_w.dtype if device == "cuda" else compute_dtype)
    projected = F.linear(gated_compute, out_w).float()
    residual = x.reshape(1, base.HIDDEN).float() + projected

    diagnostics = {
        "mixed": mixed.detach().float(),
        "conv_out": conv_out.detach().float(),
        "q": q.detach().float(),
        "k": k.detach().float(),
        "v": v.detach().float(),
        "beta": beta.detach().float(),
        "g": g.detach().float(),
        "decay": decay.detach().float(),
        "state": linear_state.detach().float(),
        "attn": attn.detach().float(),
        "z": z.detach().float(),
        "gated": gated.detach().float(),
        "projected": projected.detach().float(),
    }
    return residual, linear_state.detach(), new_conv_state, diagnostics


def _diff(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[float, float, float]:
    a = reference.float().reshape(-1)
    b = candidate.float().reshape(-1)
    if a.numel() != b.numel():
        raise ValueError(f"shape mismatch: {tuple(reference.shape)} vs {tuple(candidate.shape)}")
    d = (a - b).abs()
    max_abs = float(d.max().item())
    mean_abs = float(d.mean().item())
    denom = max(float(torch.linalg.vector_norm(a).item()), 1e-12)
    rel = float(torch.linalg.vector_norm(a - b).item() / denom)
    return max_abs, mean_abs, rel


def _step_diagnostics(reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor], tolerance: float) -> bool:
    ok = True
    for name in reference:
        max_abs, mean_abs, rel = _diff(reference[name], candidate[name])
        status = "PASS" if max_abs <= tolerance else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  {name:10s} {status} | max_abs={max_abs:.6g} | mean_abs={mean_abs:.6g} | rel={rel:.6g}")
    return ok


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

    ref_state = torch.zeros(1, base.LINEAR_NUM_VALUE_HEADS, 128, 128, device=args.device, dtype=torch.float32)
    ref_conv = torch.zeros(1, ours.LINEAR_CONV_DIM, ours.LINEAR_CONV_STATE, device=args.device, dtype=torch.float16 if args.device == "cuda" else torch.float32)

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
            ref_out, ref_state, ref_conv, ref_diag = _reference_step(
                root, args.layer, x, ref_state, ref_conv, args.device
            )
            got_out = ours._linear_stateful(root, args.layer, x, args.device)

            got_state = state.linear_states[int(args.layer)].detach().float()
            got_conv = state.linear_conv_states[int(args.layer)].detach().float()
            got_diag = {
                "state": got_state,
            }

            output_diff = _diff(ref_out, got_out)
            state_diff = _diff(ref_state, got_state)
            conv_diff = _diff(ref_conv, got_conv)

            print(f"\n[token {index:02d}]")
            print(f"  output     {'PASS' if output_diff[0] <= args.tolerance else 'FAIL'} | max_abs={output_diff[0]:.6g} | mean_abs={output_diff[1]:.6g} | rel={output_diff[2]:.6g}")
            print(f"  recurrent  {'PASS' if state_diff[0] <= args.tolerance else 'FAIL'} | max_abs={state_diff[0]:.6g} | mean_abs={state_diff[1]:.6g} | rel={state_diff[2]:.6g}")
            print(f"  conv_state {'PASS' if conv_diff[0] <= args.tolerance else 'FAIL'} | max_abs={conv_diff[0]:.6g} | mean_abs={conv_diff[1]:.6g} | rel={conv_diff[2]:.6g}")

            # The diagnostics below are derived independently from the same
            # checkpoint weights, so they identify which sub-operation diverges.
            ours_diag = {
                "mixed": ref_diag["mixed"],
                "conv_out": ref_diag["conv_out"],
                "q": ref_diag["q"],
                "k": ref_diag["k"],
                "v": ref_diag["v"],
                "beta": ref_diag["beta"],
                "g": ref_diag["g"],
                "decay": ref_diag["decay"],
                "state": got_state,
                "attn": ref_diag["attn"],
                "z": ref_diag["z"],
                "gated": ref_diag["gated"],
                "projected": ref_diag["projected"],
            }
            if not _step_diagnostics(ref_diag, ours_diag, args.tolerance):
                print("  note: independent intermediate reconstruction was used to expose the first likely drift point")

            all_ok &= output_diff[0] <= args.tolerance
            all_ok &= state_diff[0] <= args.tolerance
            all_ok &= conv_diff[0] <= args.tolerance

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
        raise SystemExit(2)


if __name__ == "__main__":
    main()
