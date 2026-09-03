from __future__ import annotations

"""Detailed stage-by-stage fidelity probe for Qwen3.6 Gated DeltaNet.

This probe compares the official Transformers linear-attention module against
router_ia one stage at a time.  It intentionally uses a single token so the
router's recurrent path is exercised directly.
"""

import argparse
import gc
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from . import qwen36_attention_cache as attention
from . import qwen36_40layer_loop as base
from .qwen36_layer_fidelity_probe import (
    _build_meta_model,
    _find_layers,
    _load_config,
    _materialize_layer,
    _module_input_dtype,
    _pure_torch_causal_conv1d,
    _stage_stats,
)
from .qwen36_op_probe import load_embedding_row, rmsnorm

EXPERTS = 256
DEFAULT_TOLERANCE = 1e-3


def _print_stage(name: str, reference: torch.Tensor, candidate: torch.Tensor, tolerance: float) -> bool:
    s = _stage_stats(reference, candidate)
    status = "PASS" if s[0] <= tolerance else "FAIL"
    print(
        f"  {name:<24} {status} max_abs={s[0]:.6g} mean_abs={s[1]:.6g} "
        f"rel={s[2]:.6g} cosine={s[3]:.9f} ref_norm={s[4]:.6g} router_norm={s[5]:.6g}"
    )
    return status == "PASS"


def _capture_linear_calls(fn):
    calls: list[torch.Tensor] = []
    original = F.linear

    def wrapped(input, weight, bias=None):
        out = original(input, weight, bias)
        calls.append(out.detach().clone())
        return out

    F.linear = wrapped
    try:
        result = fn()
    finally:
        F.linear = original
    return result, calls


def _capture_conv_reference(fn):
    captured: dict[str, torch.Tensor] = {}
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as qwen

    original = qwen.causal_conv1d_fn

    def wrapped(x, weight, bias=None, activation=None, *args, **kwargs):
        out = _pure_torch_causal_conv1d(x, weight, bias, activation, *args, **kwargs)
        captured["input"] = x.detach().clone()
        captured["output"] = out.detach().clone()
        return out

    qwen.causal_conv1d_fn = wrapped
    try:
        result = fn()
    finally:
        qwen.causal_conv1d_fn = original
    return result, captured


def _capture_router_gated_norm(fn):
    captured: dict[str, torch.Tensor] = {}
    original = attention.gated_rmsnorm

    def wrapped(x, gate, weight, *args, **kwargs):
        out = original(x, gate, weight, *args, **kwargs)
        captured["input"] = x.detach().clone()
        captured["gate"] = gate.detach().clone()
        captured["output"] = out[0].detach().clone()
        return out

    attention.gated_rmsnorm = wrapped
    try:
        result = fn()
    finally:
        attention.gated_rmsnorm = original
    return result, captured


def _capture_norm_input(layer):
    captured: dict[str, torch.Tensor] = {}

    def hook(module, args):
        if args:
            captured["input"] = args[0].detach().clone()
        if len(args) > 1:
            captured["gate"] = args[1].detach().clone()

    handle = layer.linear_attn.norm.register_forward_pre_hook(hook)
    return captured, handle


def _capture_norm_output(layer):
    captured: dict[str, torch.Tensor] = {}

    def hook(module, args, output):
        if isinstance(output, tuple):
            output = output[0]
        captured["output"] = output.detach().clone()

    handle = layer.linear_attn.norm.register_forward_hook(hook)
    return captured, handle


def _split_qkv(conv_output: torch.Tensor):
    mixed = conv_output.reshape(1, base.LINEAR_CONV_DIM)
    q, k, v = torch.split(
        mixed,
        [base.LINEAR_KEY_DIM, base.LINEAR_KEY_DIM, base.LINEAR_VALUE_DIM],
        dim=-1,
    )
    q = q.reshape(1, base.LINEAR_NUM_K_HEADS, 128).repeat_interleave(2, dim=1)
    k = k.reshape(1, base.LINEAR_NUM_K_HEADS, 128).repeat_interleave(2, dim=1)
    v = v.reshape(1, base.LINEAR_NUM_V_HEADS, 128)
    q_norm = q.float()
    k_norm = k.float()
    q_norm = q_norm * torch.rsqrt((q_norm * q_norm).sum(dim=-1, keepdim=True) + 1e-6)
    k_norm = k_norm * torch.rsqrt((k_norm * k_norm).sum(dim=-1, keepdim=True) + 1e-6)
    q_scaled = q_norm * (128 ** -0.5)
    return q, k, v, q_norm, k_norm, q_scaled


def _reference_decay(a_raw: torch.Tensor, b_raw: torch.Tensor, layer):
    a_raw = a_raw.float().reshape(1, base.LINEAR_NUM_VALUE_HEADS)
    b_raw = b_raw.float().reshape(1, base.LINEAR_NUM_VALUE_HEADS)
    a_log = layer.linear_attn.A_log.float().reshape(1, base.LINEAR_NUM_VALUE_HEADS)
    dt_bias = layer.linear_attn.dt_bias.float().reshape(1, base.LINEAR_NUM_VALUE_HEADS)
    beta = torch.sigmoid(b_raw)
    g = -torch.exp(a_log) * F.softplus(a_raw + dt_bias)
    decay = torch.exp(g)
    return beta, g, decay


def _recurrent_reference(q, k, v, beta, decay):
    state = torch.zeros(
        (1, base.LINEAR_NUM_VALUE_HEADS, 128, 128),
        device=q.device,
        dtype=torch.float32,
    )
    state = state * decay.unsqueeze(-1).unsqueeze(-1)
    retrieved = (state * k.unsqueeze(-1)).sum(dim=-2)
    delta = (v.float() - retrieved) * beta.unsqueeze(-1)
    state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
    output = (state * q.unsqueeze(-1)).sum(dim=-2)
    return output, state


def _run(root: Path, layer, layer_idx: int, hidden: torch.Tensor, device: str, tolerance: float):
    dtype = _module_input_dtype(layer)
    hidden = hidden.to(dtype=dtype)
    norm_weight = base.load_layer_weight(root, layer_idx, "input_layernorm.weight", device)
    normed_router = rmsnorm(hidden, norm_weight)

    ref_normed = layer.input_layernorm(hidden.unsqueeze(1)).reshape(1, base.HIDDEN)

    # Official reference: capture each projection and the causal-conv output.
    ref_norm_capture, ref_norm_handle = _capture_norm_input(layer)
    ref_norm_out_capture, ref_norm_out_handle = _capture_norm_output(layer)

    def run_reference():
        out = layer.linear_attn(hidden_states=normed_router.unsqueeze(1), cache_params=None, attention_mask=None)
        if isinstance(out, tuple):
            out = out[0]
        return out.reshape(1, base.HIDDEN)

    ref_result, ref_linear = _capture_linear_calls(lambda: _capture_conv_reference(run_reference)[0])
    # The nested helper's captured conv dictionary is obtained separately below
    # with a second reference pass; this keeps the projection capture simple.
    ref_norm_handle.remove()
    ref_norm_out_handle.remove()

    # Re-run once to obtain the conv capture and norm boundaries without hooks
    # from the first pass changing numerical behavior.
    ref_conv_result, ref_conv = _capture_conv_reference(run_reference)
    del ref_conv_result

    if len(ref_linear) < 5:
        raise RuntimeError(f"Expected at least 5 linear projections in official GatedDeltaNet, got {len(ref_linear)}")

    # Official projection ordering is qkv, z, b, a, out_proj in the current
    # Transformers implementation.  Match by shape as a guard against source
    # reordering rather than blindly trusting indices.
    def pick(shape_last: int, occurrence: int = 0):
        matches = [x for x in ref_linear if x.shape[-1] == shape_last]
        if len(matches) <= occurrence:
            raise RuntimeError(f"Could not find official projection output with last dim {shape_last}")
        return matches[occurrence]

    ref_qkv = pick(base.LINEAR_CONV_DIM)
    ref_z = pick(base.LINEAR_VALUE_DIM)
    ref_b = pick(base.LINEAR_NUM_VALUE_HEADS)
    ref_a = pick(base.LINEAR_NUM_VALUE_HEADS, 1)
    ref_out = pick(base.HIDDEN)

    # Router: the same F.linear interception captures qkv, a, b, z, out_proj.
    router_linear_result, router_linear = _capture_linear_calls(
        lambda: attention.step_attention(root, layer_idx, hidden, device)
    )
    router_linear_candidates = router_linear
    if len(router_linear_candidates) < 5:
        raise RuntimeError(f"Expected at least 5 linear projections in router, got {len(router_linear_candidates)}")

    def pick_router(shape_last: int, occurrence: int = 0):
        matches = [x for x in router_linear_candidates if x.shape[-1] == shape_last]
        if len(matches) <= occurrence:
            raise RuntimeError(f"Could not find router projection output with last dim {shape_last}")
        return matches[occurrence]

    router_qkv = pick_router(base.LINEAR_CONV_DIM)
    router_z = pick_router(base.LINEAR_VALUE_DIM)
    router_b = pick_router(base.LINEAR_NUM_VALUE_HEADS)
    router_a = pick_router(base.LINEAR_NUM_VALUE_HEADS, 1)
    router_out = pick_router(base.HIDDEN)

    print("\n=== PROJECTION / CONV ===")
    ok = []
    ok.append(_print_stage("in_proj_qkv", ref_qkv, router_qkv, tolerance))
    if "input" in ref_conv:
        ok.append(_print_stage("causal_conv_input", ref_conv["input"].reshape_as(router_qkv), router_qkv, tolerance))
        # Router conv output is recovered from the next split stage below.  The
        # cache's internal conv state is intentionally not modified by this probe.

    ref_q, ref_k, ref_v, ref_qn, ref_kn, ref_qs = _split_qkv(ref_conv["output"])
    router_q, router_k, router_v, router_qn, router_kn, router_qs = _split_qkv(
        # The router projection is followed by its own causal conv.  Recover the
        # exact one-token result by calling the private conv helper on a clean state.
        ref_qkv.new_zeros(ref_qkv.shape)
    )

    # Obtain the router's actual conv output from a clean temporary AttentionState.
    tmp_state = attention.AttentionState()
    tmp_state.bind(device)
    conv_w = base.load_layer_weight(root, layer_idx, "linear_attn.conv1d.weight", device)
    router_conv = attention._causal_conv1d_step(tmp_state, layer_idx, router_qkv.reshape(1, -1), conv_w)
    router_q, router_k, router_v, router_qn, router_kn, router_qs = _split_qkv(router_conv.reshape(1, -1, 1))
    ok.append(_print_stage("causal_conv", ref_conv["output"].reshape_as(router_conv), router_conv, tolerance))
    ok.append(_print_stage("q_l2norm", ref_qn, router_qn, tolerance))
    ok.append(_print_stage("k_l2norm", ref_kn, router_kn, tolerance))
    ok.append(_print_stage("q_scale", ref_qs, router_qs, tolerance))
    ok.append(_print_stage("v_split", ref_v, router_v, tolerance))

    print("\n=== GATES ===")
    ok.append(_print_stage("in_proj_b", ref_b, router_b, tolerance))
    ok.append(_print_stage("in_proj_a", ref_a, router_a, tolerance))
    ref_beta, ref_g, ref_decay = _reference_decay(ref_a, ref_b, layer)
    router_beta, router_g, router_decay = _reference_decay(router_a, router_b, layer)
    ok.append(_print_stage("beta=sigmoid(b)", ref_beta, router_beta, tolerance))
    ok.append(_print_stage("g=decay_log", ref_g, router_g, tolerance))
    ok.append(_print_stage("decay=exp(g)", ref_decay, router_decay, tolerance))
    ok.append(_print_stage("in_proj_z", ref_z, router_z, tolerance))

    print("\n=== RECURRENCE / GATED NORM ===")
    ref_core, ref_state = _recurrent_reference(ref_qs, ref_kn, ref_v, ref_beta, ref_decay)
    # The official fallback returns the recurrent core to its gated RMSNorm.
    # Capture that boundary directly so this test does not confuse recurrence
    # differences with RMSNormGated differences.
    ref_norm_in: dict[str, torch.Tensor] = {}
    ref_norm_out: dict[str, torch.Tensor] = {}

    def pre_hook(module, args):
        if args:
            ref_norm_in["x"] = args[0].detach().clone()
        if len(args) > 1:
            ref_norm_in["z"] = args[1].detach().clone()

    def post_hook(module, args, output):
        if isinstance(output, tuple):
            output = output[0]
        ref_norm_out["y"] = output.detach().clone()

    h1 = layer.linear_attn.norm.register_forward_pre_hook(pre_hook)
    h2 = layer.linear_attn.norm.register_forward_hook(post_hook)
    try:
        _ = run_reference()
    finally:
        h1.remove()
        h2.remove()

    router_norm_capture: dict[str, torch.Tensor] = {}

    def router_norm(x, gate, weight, *args, **kwargs):
        out = original_gated(x, gate, weight, *args, **kwargs)
        router_norm_capture["x"] = x.detach().clone()
        router_norm_capture["z"] = gate.detach().clone()
        router_norm_capture["y"] = out[0].detach().clone()
        return out

    original_gated = attention.gated_rmsnorm
    attention.gated_rmsnorm = router_norm
    try:
        _ = attention.step_attention(root, layer_idx, hidden, device)
    finally:
        attention.gated_rmsnorm = original_gated

    ok.append(_print_stage("recurrent_core", ref_norm_in["x"], router_norm_capture["x"], tolerance))
    ok.append(_print_stage("gate_z", ref_norm_in["z"], router_norm_capture["z"], tolerance))
    ok.append(_print_stage("gated_rmsnorm", ref_norm_out["y"], router_norm_capture["y"], tolerance))
    ok.append(_print_stage("out_proj", ref_out, router_out, tolerance))

    # Also compare the complete attention output captured from the two paths.
    ok.append(_print_stage("linear_attention_total", ref_result, router_linear_result, tolerance))

    print("\n=== RESULT ===")
    print(f"status={'PASS' if all(ok) else 'FAIL'}")
    return all(ok)


def main():
    parser = argparse.ArgumentParser(description="Detailed Qwen3.6 Gated DeltaNet fidelity probe")
    parser.add_argument("root", type=Path)
    parser.add_argument("--token-id", type=int, default=0)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    if args.layer < 0 or args.layer >= base.DEFAULT_LAYERS:
        raise SystemExit(f"--layer must be in [0, {base.DEFAULT_LAYERS - 1}]")
    if base.attention_type(args.root.resolve(), args.layer) != "linear_attention":
        raise SystemExit(f"Layer {args.layer} is not linear_attention")

    root = args.root.resolve()
    config = _load_config(root)
    model = _build_meta_model(config)
    layers = _find_layers(model)
    layer = layers[args.layer]
    loaded, total = _materialize_layer(root, layer, args.layer, args.device)
    print("op=linear-attention-fidelity")
    print(f"layer={args.layer}")
    print(f"token_id={args.token_id}")
    print(f"device={args.device}")
    print(f"loaded={loaded}/{total}")
    print(f"tolerance={args.tolerance}")

    hidden = load_embedding_row(root, args.token_id).reshape(1, base.HIDDEN).to(args.device).float()
    state = attention.AttentionState()
    state.bind(args.device)
    attention.activate(root, state)
    try:
        _run(root, layer, args.layer, hidden, args.device, args.tolerance)
    finally:
        attention.deactivate(root)
        layer.to_empty(device="meta")
        gc.collect()


if __name__ == "__main__":
    main()
