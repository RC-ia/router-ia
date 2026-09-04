from __future__ import annotations

"""Independent line-by-line recurrent GDN diagnostic."""

import argparse
import gc
from pathlib import Path

import torch

from . import qwen36_linear_attention_hf as candidate
from . import qwen36_40layer_loop as base
from .qwen36_linear_attention_stateful_probe import _make_reference_cache, _patch_official_conv
from .qwen36_layer_fidelity_probe import _build_meta_model, _find_layers, _load_config, _materialize_layer, _module_input_dtype
from .qwen36_op_probe import load_embedding_row, rmsnorm


def stats(a, b):
    a = a.detach().float(); b = b.detach().float()
    d = (a - b).abs(); denom = a.norm().clamp_min(1e-12)
    cos = torch.nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()
    return float(d.max()), float(d.mean()), float(d.norm() / denom), cos


def report(name, ref, got):
    x = stats(ref, got)
    print(f"    {name:<24} max={x[0]:.7g} mean={x[1]:.7g} rel={x[2]:.7g} cos={x[3]:.9f}")


def clone(x):
    return None if x is None else x.detach().clone()


def hf_literal(query, key, value, g, beta, state):
    """Literal operation order from HF torch_recurrent_gated_delta_rule."""
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)

    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, g)
    ]
    inv_q = torch.rsqrt((query * query).sum(dim=-1, keepdim=True) + 1e-6)
    query = query * inv_q
    inv_k = torch.rsqrt((key * key).sum(dim=-1, keepdim=True) + 1e-6)
    key = key * inv_k
    query = query * (k_head_dim ** -0.5)

    recurrent = torch.zeros(recurrent_state_shape, device=value.device, dtype=value.dtype) if state is None else state

    q_t = query[:, :, 0]
    k_t = key[:, :, 0]
    v_t = value[:, :, 0]
    decay_t = decay[:, :, 0].exp()[..., None, None]
    beta_t = beta[:, :, 0].unsqueeze(-1)

    recurrent = recurrent * decay_t
    kv_mem = (recurrent * k_t.unsqueeze(-1)).sum(dim=-2)
    delta = (v_t - kv_mem) * beta_t
    recurrent = recurrent + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
    output = (recurrent * q_t.unsqueeze(-1)).sum(dim=-2)

    stages = {
        "query_fp32": query,
        "key_fp32": key,
        "value_fp32": value,
        "beta_fp32": beta,
        "decay_fp32": decay,
        "q_t": q_t,
        "k_t": k_t,
        "v_t": v_t,
        "decay_t": decay_t,
        "beta_t": beta_t,
        "decayed_state": recurrent if False else None,
        "kv_mem": kv_mem,
        "delta": delta,
    }
    # Recompute decayed state explicitly so the diagnostic exposes it without alias confusion.
    decayed_state = (state if state is not None else torch.zeros(recurrent_state_shape, device=value.device, dtype=value.dtype)) * decay_t
    stages["decayed_state"] = decayed_state
    stages["update"] = k_t.unsqueeze(-1) * delta.unsqueeze(-2)
    stages["final_state"] = recurrent
    return output[:, None].contiguous().to(initial_dtype), recurrent.detach(), stages


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--layer", type=int, default=1)
    p.add_argument("--tokens", type=int, default=4)
    p.add_argument("--token-id", type=int, default=0)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = p.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")

    root = args.root.resolve()
    config = _load_config(root)
    meta = _build_meta_model(config)
    layers = _find_layers(meta)
    layer = layers[args.layer]
    loaded, total = _materialize_layer(root, layer, args.layer, args.device)
    input_dtype = _module_input_dtype(layer)
    input_norm = base.load_layer_weight(root, args.layer, "input_layernorm.weight", args.device)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    tokens = [load_embedding_row(root, args.token_id + i).reshape(1, base.HIDDEN).to(args.device).to(dtype) for i in range(args.tokens)]

    qwen, conv_originals = _patch_official_conv()
    ref_cache = _make_reference_cache(config)
    old_ref = qwen.torch_recurrent_gated_delta_rule
    old_cand = candidate.gated_delta_recurrent
    ref_cap = {}
    cand_cap = {}

    def ref_wrap(query, key, value, g, beta, initial_state=None, **kwargs):
        out, st = old_ref(query, key, value, g=g, beta=beta, initial_state=initial_state, **kwargs)
        ref_cap.update({"query":clone(query),"key":clone(key),"value":clone(value),"g":clone(g),"beta":clone(beta),"state":clone(st),"initial":clone(initial_state),"core":clone(out)})
        return out, st

    def cand_wrap(query, key, value, g, beta, state):
        out, st = old_cand(query, key, value, g, beta, state)
        cand_cap.update({"query":clone(query),"key":clone(key),"value":clone(value),"g":clone(g),"beta":clone(beta),"state":clone(st),"initial":clone(state),"core":clone(out)})
        return out, st

    qwen.torch_recurrent_gated_delta_rule = ref_wrap
    candidate.gated_delta_recurrent = cand_wrap
    print(f"op=linear-attention-recurrence-literal layer={args.layer} tokens={args.tokens} device={args.device} materialized={loaded}/{total}")

    try:
        cand_conv = cand_state = None
        for pos, raw in enumerate(tokens):
            token = raw.to(dtype=input_dtype)
            normed = rmsnorm(token, input_norm)
            ref_cap.clear(); cand_cap.clear()
            reference = layer.linear_attn(hidden_states=normed.unsqueeze(1), cache_params=ref_cache, attention_mask=None)
            if isinstance(reference, tuple): reference = reference[0]
            cand_out, cand_conv, cand_state = candidate.linear_attention_step(root, args.layer, token, cand_conv, cand_state, args.device)
            if not ref_cap or not cand_cap:
                print(f"\nTOKEN {pos}: chunk path"); continue
            print(f"\nTOKEN {pos}")
            # First compare captured inputs, then independently execute the exact HF operation order twice.
            for name in ("query","key","value","g","beta","initial"):
                if ref_cap[name] is not None and cand_cap[name] is not None:
                    report(name, ref_cap[name], cand_cap[name])

            ref_o, ref_s, rs = hf_literal(ref_cap["query"], ref_cap["key"], ref_cap["value"], ref_cap["g"], ref_cap["beta"], ref_cap["initial"])
            cand_o, cand_s, cs = hf_literal(cand_cap["query"], cand_cap["key"], cand_cap["value"], cand_cap["g"], cand_cap["beta"], cand_cap["initial"])

            print("  literal stages (reference inputs vs candidate inputs)")
            for name in ("query_fp32","key_fp32","value_fp32","beta_fp32","decay_fp32","q_t","k_t","v_t","decay_t","beta_t","decayed_state","kv_mem","delta","update","final_state"):
                report(name, rs[name], cs[name])
            report("literal_core", ref_o, cand_o)
            report("HF_core", ref_cap["core"], ref_o)
            report("HF_state", ref_cap["state"], ref_s)
            report("candidate_state", cand_cap["state"], cand_s)
            report("candidate_final_state_vs_HF", cand_cap["state"], ref_cap["state"])
            report("final_output", reference.reshape_as(cand_out), cand_out)
    finally:
        qwen.torch_recurrent_gated_delta_rule = old_ref
        candidate.gated_delta_recurrent = old_cand
        qwen.causal_conv1d_fn, qwen.causal_conv1d_update = conv_originals
        layer.to_empty(device="meta")
        del meta
        gc.collect()
        if args.device == "cuda": torch.cuda.empty_cache()
    return 0

if __name__ == "__main__": raise SystemExit(main())
