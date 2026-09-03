from __future__ import annotations

"""Layer-by-layer fidelity probe for Qwen3.6."""

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from . import qwen36_40layer_loop as base
from . import qwen36_attention_cache as attention
from . import qwen36_chat_batch as chat
from .qwen36_gated_norm_probe import gated_rmsnorm
from .qwen36_op_probe import dequantize_fp8_blockwise, load_embedding_row, rmsnorm

DEFAULT_TOLERANCE = 1e-3
EXPERTS = 256


def _load_config(root: Path):
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(str(root), local_files_only=True)


def _pure_torch_causal_conv1d(x, weight, bias=None, activation=None, *args, **kwargs):
    if x.ndim != 3 or weight.ndim != 2:
        raise ValueError(f"Unexpected causal conv shapes: x={tuple(x.shape)} weight={tuple(weight.shape)}")
    batch, channels, length = x.shape
    if weight.shape[0] != channels:
        raise ValueError(f"Causal conv channel mismatch: x={channels} weight={weight.shape[0]}")
    kernel = weight.shape[1]
    padded = F.pad(x, (kernel - 1, 0))
    windows = padded.unfold(-1, kernel, 1)[..., :length, :]
    out = (windows * weight.unsqueeze(0).unsqueeze(2)).sum(dim=-1)
    if bias is not None:
        out = out + bias.reshape(1, -1, 1)
    if activation is not None:
        if str(activation).lower() not in {"silu", "swish"}:
            raise ValueError(f"Unsupported causal conv activation: {activation!r}")
        out = F.silu(out)
    return out


def _disable_optional_qwen_kernels() -> None:
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as qwen
    qwen.causal_conv1d_fn = _pure_torch_causal_conv1d
    qwen.causal_conv1d_update = None
    qwen.chunk_gated_delta_rule = None
    qwen.fused_recurrent_gated_delta_rule = None
    qwen.FusedRMSNormGated = None


def _build_meta_model(config):
    from transformers import AutoModelForCausalLM
    _disable_optional_qwen_kernels()
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model.eval()
    return model


def _find_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    for path in (("model", "language_model", "layers"), ("language_model", "layers"), ("model", "layers"), ("layers",), ("transformer", "h")):
        obj: Any = model
        try:
            for name in path:
                obj = getattr(obj, name)
        except AttributeError:
            continue
        if isinstance(obj, torch.nn.ModuleList):
            return obj
    raise RuntimeError("Could not find decoder layers in Transformers model")


def _checkpoint_index(root: Path) -> dict[str, str]:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        data = json.loads(index_path.read_text(encoding="utf-8"))
        if isinstance(data.get("weight_map"), dict):
            return {str(k): str(v) for k, v in data["weight_map"].items()}
    single = root / "model.safetensors"
    if single.is_file():
        return {"__single__": single.name}
    files = sorted(root.glob("*.safetensors"))
    if len(files) == 1:
        return {"__single__": files[0].name}
    raise FileNotFoundError(f"Expected safetensors checkpoint under {root}")


def _filename(weight_map, key):
    if "__single__" in weight_map:
        return weight_map["__single__"]
    return weight_map[key]


def _load_checkpoint_tensor(root, weight_map, key):
    from safetensors import safe_open
    filename = _filename(weight_map, key)
    with safe_open(str(root / filename), framework="pt", device="cpu") as handle:
        if key not in handle.keys():
            raise KeyError(key)
        tensor = handle.get_tensor(key)
        if tensor.dtype == torch.float8_e4m3fn:
            scale_key = key + "_scale_inv"
            scale_filename = _filename(weight_map, scale_key)
            with safe_open(str(root / scale_filename), framework="pt", device="cpu") as scale_handle:
                tensor = dequantize_fp8_blockwise(tensor, scale_handle.get_tensor(scale_key))
        return tensor


def _load_fused_expert_parameter(root, weight_map, layer_idx, local_name, expected_shape):
    prefix = base.layer_prefix(layer_idx)
    if local_name.endswith("mlp.experts.gate_up_proj"):
        parts = []
        for expert in range(EXPERTS):
            gate = _load_checkpoint_tensor(root, weight_map, f"{prefix}mlp.experts.{expert}.gate_proj.weight")
            up = _load_checkpoint_tensor(root, weight_map, f"{prefix}mlp.experts.{expert}.up_proj.weight")
            parts.append(torch.cat((gate, up), dim=0))
        fused = torch.stack(parts, dim=0)
    elif local_name.endswith("mlp.experts.down_proj"):
        fused = torch.stack([_load_checkpoint_tensor(root, weight_map, f"{prefix}mlp.experts.{expert}.down_proj.weight") for expert in range(EXPERTS)], dim=0)
    else:
        raise KeyError(local_name)
    if tuple(fused.shape) != expected_shape:
        raise RuntimeError(f"Fused expert shape mismatch: built={tuple(fused.shape)} model={expected_shape}")
    return fused


def _materialize_layer(root, layer, layer_idx, device):
    weight_map = _checkpoint_index(root)
    prefix = base.layer_prefix(layer_idx)
    layer.to_empty(device=device)
    destinations = {**dict(layer.named_parameters()), **dict(layer.named_buffers())}
    loaded = 0
    missing = []
    with torch.no_grad():
        for local_name, destination in destinations.items():
            checkpoint_key = prefix + local_name
            try:
                tensor = _load_checkpoint_tensor(root, weight_map, checkpoint_key)
            except KeyError:
                if local_name.endswith("mlp.experts.gate_up_proj") or local_name.endswith("mlp.experts.down_proj"):
                    try:
                        tensor = _load_fused_expert_parameter(root, weight_map, layer_idx, local_name, tuple(destination.shape))
                    except KeyError:
                        missing.append(checkpoint_key)
                        continue
                else:
                    missing.append(checkpoint_key)
                    continue
            if tuple(tensor.shape) != tuple(destination.shape):
                raise RuntimeError(f"Shape mismatch for {checkpoint_key}: checkpoint={tuple(tensor.shape)} model={tuple(destination.shape)}")
            if destination.dtype.is_floating_point:
                tensor = tensor.to(dtype=destination.dtype)
            destination.copy_(tensor.to(device=device))
            loaded += 1
    if missing:
        raise RuntimeError(f"Layer {layer_idx} is missing {len(missing)} checkpoint tensors. First missing: {missing[:5]}")
    return loaded, len(destinations)


def _module_input_dtype(layer):
    for p in layer.parameters():
        if p.dtype.is_floating_point:
            return p.dtype
    for b in layer.buffers():
        if b.dtype.is_floating_point:
            return b.dtype
    return torch.float32


def _stage_stats(reference, candidate):
    ref = reference.float().reshape(-1)
    got = candidate.float().reshape(-1)
    diff = (got - ref).abs()
    rn = torch.linalg.vector_norm(ref).item()
    gn = torch.linalg.vector_norm(got).item()
    cosine = torch.dot(ref, got).item() / max(rn * gn, 1e-12)
    rel = diff.max().item() / max(rn / math.sqrt(ref.numel()), 1e-12)
    return diff.max().item(), diff.mean().item(), rel, cosine, rn, gn


def _print_stage(name, reference, candidate, tolerance):
    s = _stage_stats(reference, candidate)
    status = "PASS" if s[0] <= tolerance else "FAIL"
    print(f"  {name:<18} {status} max_abs={s[0]:.6g} mean_abs={s[1]:.6g} rel={s[2]:.6g} cosine={s[3]:.9f} ref_norm={s[4]:.6g} router_norm={s[5]:.6g}")
    return status, s


def _reference_attention(layer, hidden, dtype):
    h = hidden.to(dtype=dtype).unsqueeze(1)
    normed = layer.input_layernorm(h)
    out = layer.linear_attn(hidden_states=normed, cache_params=None, attention_mask=None)
    if isinstance(out, tuple):
        out = out[0]
    return normed.reshape(1, base.HIDDEN).float(), out.reshape(1, base.HIDDEN).float(), (hidden + out.reshape(1, base.HIDDEN).float())


def _reference_full_attention(layer, hidden, dtype):
    raise RuntimeError("Stage diagnostics for full attention are not implemented yet")


def _reference_moe(layer, residual):
    h = layer.post_attention_layernorm(residual.to(dtype=_module_input_dtype(layer)).unsqueeze(1))
    out = layer.mlp(h)
    if isinstance(out, tuple):
        out = out[0]
    return h.reshape(1, base.HIDDEN).float(), out.reshape(1, base.HIDDEN).float(), residual + out.reshape(1, base.HIDDEN).float()


def _run_diagnostic(root, layer, hidden_ref, hidden_router, layer_idx, device, tolerance):
    dtype = _module_input_dtype(layer)
    # The official decoder receives BF16 hidden states on this path.  RMSNorm
    # computes in FP32 internally but returns the original input dtype, so the
    # router must enter the layer in the same dtype before comparing anything.
    hidden_router = hidden_router.to(dtype=dtype)
    kind = base.attention_type(root, layer_idx)
    if kind == "full_attention":
        ref_normed, ref_attn, ref_residual = _reference_full_attention(layer, hidden_ref, dtype)
    else:
        ref_normed, ref_attn, ref_residual = _reference_attention(layer, hidden_ref, dtype)
    router_norm_weight = base.load_layer_weight(root, layer_idx, "input_layernorm.weight", device)
    router_normed = rmsnorm(hidden_router, router_norm_weight).float()
    statuses = []
    statuses.append(_print_stage("input_layernorm", ref_normed, router_normed, tolerance)[0])
    router_residual = attention.step_attention(root, layer_idx, hidden_router, device)
    router_attn = router_residual - hidden_router.float()
    statuses.append(_print_stage("attention", ref_attn, router_attn, tolerance)[0])
    statuses.append(_print_stage("post_attn_residual", ref_residual, router_residual, tolerance)[0])
    ref_postnorm, ref_moe, ref_final = _reference_moe(layer, ref_residual)
    router_postnorm_weight = base.load_layer_weight(root, layer_idx, "post_attention_layernorm.weight", device)
    router_postnorm = rmsnorm(router_residual, router_postnorm_weight)
    statuses.append(_print_stage("post_attention_norm", ref_postnorm, router_postnorm, tolerance)[0])
    router_final, *_ = chat.batched_moe_step(root, layer_idx, router_residual, top_k=8, device=device)
    router_moe = router_final - router_residual
    statuses.append(_print_stage("moe", ref_moe, router_moe, tolerance)[0])
    statuses.append(_print_stage("final", ref_final, router_final, tolerance)[0])
    return ref_final, router_final, all(status == "PASS" for status in statuses)


def main():
    parser = argparse.ArgumentParser(description="Qwen3.6 layer-by-layer official fidelity probe")
    parser.add_argument("root", type=Path)
    parser.add_argument("--token-id", type=int, default=0)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--stop-on-fail", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    if args.tolerance <= 0:
        raise SystemExit("--tolerance must be > 0")
    if args.layer is not None and not 0 <= args.layer < base.DEFAULT_LAYERS:
        raise SystemExit(f"--layer must be in [0, {base.DEFAULT_LAYERS - 1}]")
    root = args.root.resolve()
    config = _load_config(root)
    meta_model = _build_meta_model(config)
    layers = _find_layers(meta_model)
    if len(layers) < base.DEFAULT_LAYERS:
        raise SystemExit(f"Expected {base.DEFAULT_LAYERS} layers, found {len(layers)}")
    hidden_ref = load_embedding_row(root, args.token_id).reshape(1, base.HIDDEN).to(args.device).float()
    hidden_router = hidden_ref.detach().clone()
    state = attention.state_for(root, args.device)
    state.reset()
    attention.activate(root, state)
    print("op=layer-fidelity")
    print(f"token_id={args.token_id}")
    print(f"device={args.device}")
    print(f"layers={base.DEFAULT_LAYERS}")
    print(f"target_layer={args.layer if args.layer is not None else 'all'}")
    print(f"tolerance={args.tolerance}")
    print("\n=== LAYER COMPARISON ===")
    first_fail = None
    try:
        for layer_idx in range(base.DEFAULT_LAYERS):
            official_layer = layers[layer_idx]
            loaded, total = _materialize_layer(root, official_layer, layer_idx, args.device)
            if args.layer is None or args.layer == layer_idx:
                print(f"L{layer_idx:02d} kind={base.attention_type(root, layer_idx)} loaded={loaded}/{total}")
                ref_out, router_out, target_pass = _run_diagnostic(root, official_layer, hidden_ref, hidden_router, layer_idx, args.device, args.tolerance)
                max_abs, mean_abs, rel, cosine, ref_norm, router_norm = _stage_stats(ref_out, router_out)
                status = "PASS" if max_abs <= args.tolerance else "FAIL"
                print(f"  SUMMARY            {status} max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} rel={rel:.6g} cosine={cosine:.9f} ref_norm={ref_norm:.6g} router_norm={router_norm:.6g}")
                if not target_pass or status == "FAIL":
                    first_fail = layer_idx
            else:
                # Build the independent reference chain for earlier layers.
                dtype = _module_input_dtype(official_layer)
                h = hidden_ref.to(dtype=dtype).unsqueeze(1)
                out = official_layer(hidden_states=h, position_embeddings=None, position_ids=torch.zeros((1, 1), device=args.device, dtype=torch.long), attention_mask=None, past_key_values=None)
                hidden_ref = (out[0] if isinstance(out, tuple) else out).reshape(1, base.HIDDEN).float()
                hidden_router = attention.step_attention(root, layer_idx, hidden_router, args.device)
                hidden_router, *_ = chat.batched_moe_step(root, layer_idx, hidden_router, top_k=8, device=args.device)
                hidden_router = hidden_router.detach().float()
            official_layer.to_empty(device="meta")
            gc.collect()
            if args.stop_on_fail and first_fail is not None:
                break
    finally:
        attention.deactivate(root)
    print("\n=== RESULT ===")
    print(f"status={'FAIL' if first_fail is not None else 'PASS'}")


if __name__ == "__main__":
    main()
