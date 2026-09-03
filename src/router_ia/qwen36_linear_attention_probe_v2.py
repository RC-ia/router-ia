from __future__ import annotations

"""Stage-by-stage fidelity probe for Qwen3.6 Gated DeltaNet."""

import argparse
import gc
from pathlib import Path

import torch
import torch.nn.functional as F

from . import qwen36_attention_cache as attention
from . import qwen36_40layer_loop as base
from .qwen36_layer_fidelity_probe import (_build_meta_model, _find_layers, _load_config, _materialize_layer, _module_input_dtype, _pure_torch_causal_conv1d, _stage_stats)
from .qwen36_op_probe import load_embedding_row, rmsnorm

DEFAULT_TOLERANCE = 1e-3
LINEAR_CONV_DIM = base.LINEAR_KEY_DIM * 2 + base.LINEAR_VALUE_DIM


def report(name, reference, candidate, tolerance):
    s = _stage_stats(reference, candidate)
    status = "PASS" if s[0] <= tolerance else "FAIL"
    print(f"  {name:<24} {status} max_abs={s[0]:.6g} mean_abs={s[1]:.6g} rel={s[2]:.6g} cosine={s[3]:.9f} ref_norm={s[4]:.6g} router_norm={s[5]:.6g}")
    return status == "PASS"


def capture_linears(fn):
    original = F.linear
    calls = []
    def wrapped(x, weight, bias=None):
        y = original(x, weight, bias)
        calls.append(y.detach().clone())
        return y
    F.linear = wrapped
    try:
        result = fn()
    finally:
        F.linear = original
    return result, calls


def capture_conv(fn):
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as qwen
    original = qwen.causal_conv1d_fn
    captured = {}
    def wrapped(x, weight, bias=None, activation=None, *args, **kwargs):
        y = _pure_torch_causal_conv1d(x, weight, bias, activation, *args, **kwargs)
        captured["input"] = x.detach().clone()
        captured["output"] = y.detach().clone()
        return y
    qwen.causal_conv1d_fn = wrapped
    try:
        result = fn()
    finally:
        qwen.causal_conv1d_fn = original
    return result, captured


def split_qkv(conv):
    mixed = conv.reshape(1, LINEAR_CONV_DIM)
    q, k, v = torch.split(mixed, [base.LINEAR_KEY_DIM, base.LINEAR_KEY_DIM, base.LINEAR_VALUE_DIM], dim=-1)
    q = q.reshape(1, base.LINEAR_NUM_K_HEADS, 128).repeat_interleave(2, dim=1)
    k = k.reshape(1, base.LINEAR_NUM_K_HEADS, 128).repeat_interleave(2, dim=1)
    v = v.reshape(1, base.LINEAR_NUM_V_HEADS, 128)
    q = q.float(); k = k.float()
    q_norm = q * torch.rsqrt((q * q).sum(dim=-1, keepdim=True) + 1e-6)
    k_norm = k * torch.rsqrt((k * k).sum(dim=-1, keepdim=True) + 1e-6)
    q_scaled = q_norm * (128 ** -0.5)
    return q, k, v, q_norm, k_norm, q_scaled


def gates(a, b, layer):
    a = a.float().reshape(1, base.LINEAR_NUM_V_HEADS)
    b = b.float().reshape(1, base.LINEAR_NUM_V_HEADS)
    a_log = layer.linear_attn.A_log.float().reshape(1, base.LINEAR_NUM_V_HEADS)
    dt = layer.linear_attn.dt_bias.float().reshape(1, base.LINEAR_NUM_V_HEADS)
    beta = torch.sigmoid(b)
    g = -torch.exp(a_log) * F.softplus(a + dt)
    decay = torch.exp(g)
    return beta, g, decay


def recurrent(q, k, v, beta, decay):
    state = torch.zeros((1, base.LINEAR_NUM_V_HEADS, 128, 128), device=q.device, dtype=torch.float32)
    state = state * decay.unsqueeze(-1).unsqueeze(-1)
    retrieved = (state * k.unsqueeze(-1)).sum(dim=-2)
    delta = (v.float() - retrieved) * beta.unsqueeze(-1)
    state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
    output = (state * q.unsqueeze(-1)).sum(dim=-2)
    return output, state


def run(root, layer, layer_idx, hidden, device, tolerance):
    dtype = _module_input_dtype(layer)
    hidden = hidden.to(dtype=dtype)
    norm_weight = base.load_layer_weight(root, layer_idx, "input_layernorm.weight", device)
    normed = rmsnorm(hidden, norm_weight)

    def official():
        y = layer.linear_attn(hidden_states=normed.unsqueeze(1), cache_params=None, attention_mask=None)
        if isinstance(y, tuple): y = y[0]
        return y.reshape(1, base.HIDDEN)

    ref_result, ref_linear = capture_linears(lambda: capture_conv(official)[0])
    _, ref_conv = capture_conv(official)
    if len(ref_linear) < 5:
        raise RuntimeError(f"Official GatedDeltaNet exposed only {len(ref_linear)} linear calls")

    def by_last_dim(calls, dim, occurrence=0):
        matches = [x for x in calls if x.shape[-1] == dim]
        if len(matches) <= occurrence:
            raise RuntimeError(f"Could not locate projection output with last dim {dim}")
        return matches[occurrence]

    ref_qkv = by_last_dim(ref_linear, LINEAR_CONV_DIM)
    ref_z = by_last_dim(ref_linear, base.LINEAR_VALUE_DIM)
    ref_a = by_last_dim(ref_linear, base.LINEAR_NUM_V_HEADS)
    ref_b = by_last_dim(ref_linear, base.LINEAR_NUM_V_HEADS, 1)
    ref_out = by_last_dim(ref_linear, base.HIDDEN)

    state = attention.active(root, device); state.reset()
    router_result, router_linear = capture_linears(lambda: attention.step_attention(root, layer_idx, hidden, device))
    router_qkv = by_last_dim(router_linear, LINEAR_CONV_DIM)
    router_z = by_last_dim(router_linear, base.LINEAR_VALUE_DIM)
    router_a = by_last_dim(router_linear, base.LINEAR_NUM_V_HEADS)
    router_b = by_last_dim(router_linear, base.LINEAR_NUM_V_HEADS, 1)
    router_out = by_last_dim(router_linear, base.HIDDEN)

    print("\n=== PROJECTION ===")
    ok = [report("in_proj_qkv", ref_qkv, router_qkv, tolerance), report("in_proj_z", ref_z, router_z, tolerance), report("in_proj_b", ref_b, router_b, tolerance), report("in_proj_a", ref_a, router_a, tolerance)]

    tmp = attention.AttentionState(); tmp.bind(device)
    conv_w = base.load_layer_weight(root, layer_idx, "linear_attn.conv1d.weight", device)
    router_conv = attention._causal_conv1d_step(tmp, layer_idx, router_qkv.reshape(1, -1), conv_w)
    ref_conv_out = ref_conv["output"].reshape_as(router_conv)
    print("\n=== CAUSAL CONV / QKV ===")
    ok.append(report("causal_conv", ref_conv_out, router_conv, tolerance))
    ref_q, ref_k, ref_v, ref_qn, ref_kn, ref_qs = split_qkv(ref_conv_out)
    r_q, r_k, r_v, r_qn, r_kn, r_qs = split_qkv(router_conv.reshape(1, -1, 1))
    ok += [report("q_l2norm", ref_qn, r_qn, tolerance), report("k_l2norm", ref_kn, r_kn, tolerance), report("q_scale", ref_qs, r_qs, tolerance), report("v_split", ref_v, r_v, tolerance)]

    print("\n=== BETA / DECAY ===")
    ref_beta, ref_g, ref_decay = gates(ref_a, ref_b, layer)
    r_beta, r_g, r_decay = gates(router_a, router_b, layer)
    ok += [report("beta=sigmoid(b)", ref_beta, r_beta, tolerance), report("g=-exp(A)*softplus", ref_g, r_g, tolerance), report("decay=exp(g)", ref_decay, r_decay, tolerance)]

    print("\n=== RECURRENCE / GATED NORM ===")
    ref_core, _ = recurrent(ref_qs, ref_kn, ref_v, ref_beta, ref_decay)
    ref_norm = {}
    def pre(module, args):
        if args: ref_norm["x"] = args[0].detach().clone()
        if len(args) > 1: ref_norm["z"] = args[1].detach().clone()
    def post(module, args, output):
        if isinstance(output, tuple): output = output[0]
        ref_norm["y"] = output.detach().clone()
    h1 = layer.linear_attn.norm.register_forward_pre_hook(pre); h2 = layer.linear_attn.norm.register_forward_hook(post)
    try: official()
    finally: h1.remove(); h2.remove()

    state.reset(); router_norm = {}; original_gated = attention.gated_rmsnorm
    def gated(x, z, weight, *args, **kwargs):
        y = original_gated(x, z, weight, *args, **kwargs)
        router_norm["x"] = x.detach().clone(); router_norm["z"] = z.detach().clone(); router_norm["y"] = y[0].detach().clone()
        return y
    attention.gated_rmsnorm = gated
    try: router_norm_result = attention.step_attention(root, layer_idx, hidden, device)
    finally: attention.gated_rmsnorm = original_gated

    router_attention_only = router_norm_result - hidden.reshape(1, base.HIDDEN).float()
    ok += [report("recurrent_core", ref_norm["x"], router_norm["x"], tolerance), report("gate_z", ref_norm["z"], router_norm["z"], tolerance), report("gated_rmsnorm", ref_norm["y"], router_norm["y"], tolerance), report("out_proj", ref_out, router_out, tolerance), report("linear_attention_total", ref_result, router_attention_only, tolerance)]

    print("\n=== RESULT ==="); print(f"status={'PASS' if all(ok) else 'FAIL'}"); return all(ok)


def main():
    parser = argparse.ArgumentParser(description="Qwen3.6 Linear Attention detailed fidelity probe")
    parser.add_argument("root", type=Path); parser.add_argument("--token-id", type=int, default=0); parser.add_argument("--layer", type=int, default=0); parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda"); parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available(): raise SystemExit("CUDA unavailable")
    if not 0 <= args.layer < base.DEFAULT_LAYERS: raise SystemExit(f"--layer must be in [0, {base.DEFAULT_LAYERS - 1}]")
    root = args.root.resolve();
    if base.attention_type(root, args.layer) != "linear_attention": raise SystemExit(f"Layer {args.layer} is not linear_attention")
    config = _load_config(root); model = _build_meta_model(config); layers = _find_layers(model); layer = layers[args.layer]
    loaded, total = _materialize_layer(root, layer, args.layer, args.device)
    print("op=linear-attention-fidelity-detailed"); print(f"layer={args.layer}"); print(f"token_id={args.token_id}"); print(f"device={args.device}"); print(f"loaded={loaded}/{total}"); print(f"tolerance={args.tolerance}")
    hidden = load_embedding_row(root, args.token_id).reshape(1, base.HIDDEN).to(args.device).float()
    state = attention.state_for(root, args.device); state.reset(); attention.activate(root, state)
    try: run(root, layer, args.layer, hidden, args.device, args.tolerance)
    finally: attention.deactivate(root); layer.to_empty(device="meta"); gc.collect()


if __name__ == "__main__": main()
