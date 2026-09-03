from __future__ import annotations

"""Layer-by-layer fidelity probe for Qwen3.6."""

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any

import torch

from . import qwen36_40layer_loop as base
from . import qwen36_attention_cache as attention
from . import qwen36_chat_batch as chat
from .qwen36_op_probe import dequantize_fp8_blockwise, load_embedding_row

DEFAULT_TOLERANCE = 1e-3
EXPERTS = 256


def _load_config(root: Path):
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(str(root), local_files_only=True)


def _disable_optional_qwen_kernels() -> None:
    """Force Qwen3.5/3.6 reference modules onto pure-PyTorch fallbacks."""
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as qwen
    qwen.causal_conv1d_fn = None
    qwen.causal_conv1d_update = None
    qwen.chunk_gated_delta_rule = None
    qwen.fused_recurrent_gated_delta_rule = None
    qwen.FusedRMSNormGated = None


def _build_meta_model(config) -> torch.nn.Module:
    from transformers import AutoModelForCausalLM
    _disable_optional_qwen_kernels()
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model.eval()
    return model


def _find_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    candidates = (("model", "language_model", "layers"), ("language_model", "layers"), ("model", "layers"), ("layers",), ("transformer", "h"))
    for path in candidates:
        obj: Any = model
        try:
            for name in path:
                obj = getattr(obj, name)
        except AttributeError:
            continue
        if isinstance(obj, torch.nn.ModuleList):
            return obj
    raise RuntimeError("Could not find decoder layers in Transformers model")


def _find_rotary(meta_model: torch.nn.Module):
    candidates = (("model", "language_model", "rotary_emb"), ("model", "rotary_emb"), ("language_model", "rotary_emb"), ("rotary_emb",))
    for path in candidates:
        obj: Any = meta_model
        try:
            for name in path:
                obj = getattr(obj, name)
        except AttributeError:
            continue
        return obj
    return None


def _checkpoint_index(root: Path) -> dict[str, str]:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        data = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = data.get("weight_map")
        if isinstance(weight_map, dict):
            return {str(k): str(v) for k, v in weight_map.items()}
    single = root / "model.safetensors"
    if single.is_file():
        return {"__single__": single.name}
    files = sorted(root.glob("*.safetensors"))
    if len(files) == 1:
        return {"__single__": files[0].name}
    raise FileNotFoundError(f"Expected model.safetensors or model.safetensors.index.json under {root}")


def _filename(weight_map: dict[str, str], key: str) -> str:
    if "__single__" in weight_map:
        return weight_map["__single__"]
    if key not in weight_map:
        raise KeyError(key)
    return weight_map[key]


def _load_checkpoint_tensor(root: Path, weight_map: dict[str, str], key: str) -> torch.Tensor:
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
                scale = scale_handle.get_tensor(scale_key)
            tensor = dequantize_fp8_blockwise(tensor, scale)
        return tensor


def _load_fused_expert_parameter(root: Path, weight_map: dict[str, str], layer_idx: int, local_name: str, expected_shape: tuple[int, ...]) -> torch.Tensor:
    prefix = base.layer_prefix(layer_idx)
    if local_name.endswith("mlp.experts.gate_up_proj"):
        parts = []
        for expert in range(EXPERTS):
            gate = _load_checkpoint_tensor(root, weight_map, f"{prefix}mlp.experts.{expert}.gate_proj.weight")
            up = _load_checkpoint_tensor(root, weight_map, f"{prefix}mlp.experts.{expert}.up_proj.weight")
            if gate.ndim != 2 or up.ndim != 2 or gate.shape[1] != up.shape[1]:
                raise RuntimeError(f"Invalid expert projection shapes for layer={layer_idx} expert={expert}: gate={tuple(gate.shape)} up={tuple(up.shape)}")
            parts.append(torch.cat((gate, up), dim=0))
        fused = torch.stack(parts, dim=0)
    elif local_name.endswith("mlp.experts.down_proj"):
        fused = torch.stack([_load_checkpoint_tensor(root, weight_map, f"{prefix}mlp.experts.{expert}.down_proj.weight") for expert in range(EXPERTS)], dim=0)
    else:
        raise KeyError(local_name)
    if tuple(fused.shape) != expected_shape:
        raise RuntimeError(f"Fused expert shape mismatch for layer {layer_idx} {local_name}: built={tuple(fused.shape)} model={expected_shape}")
    return fused


def _materialize_layer(root: Path, layer: torch.nn.Module, layer_idx: int, device: str) -> tuple[int, int]:
    weight_map = _checkpoint_index(root)
    prefix = base.layer_prefix(layer_idx)
    layer.to_empty(device=device)
    loaded = 0
    missing: list[str] = []
    destinations = {**dict(layer.named_parameters()), **dict(layer.named_buffers())}
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


def _position_embeddings(meta_model: torch.nn.Module, config, hidden: torch.Tensor):
    rotary = _find_rotary(meta_model)
    if rotary is None:
        return None
    text_config = getattr(config, "text_config", config)
    try:
        rotary = rotary.__class__(text_config)
    except TypeError:
        rotary = rotary.__class__(config=text_config)
    rotary = rotary.to(device=hidden.device)
    position_ids = torch.zeros((1, 1), device=hidden.device, dtype=torch.long)
    with torch.inference_mode():
        return rotary(hidden.unsqueeze(1), position_ids=position_ids)


def _module_input_dtype(layer: torch.nn.Module) -> torch.dtype:
    for parameter in layer.parameters():
        if parameter.dtype.is_floating_point:
            return parameter.dtype
    for buffer in layer.buffers():
        if buffer.dtype.is_floating_point:
            return buffer.dtype
    return torch.float32


def _run_reference_layer(root: Path, layer: torch.nn.Module, meta_model: torch.nn.Module, config, hidden: torch.Tensor, layer_idx: int) -> torch.Tensor:
    is_full = base.attention_type(root, layer_idx) == "full_attention"
    dtype = _module_input_dtype(layer)
    hidden_input = hidden.to(dtype=dtype)
    position_embeddings = _position_embeddings(meta_model, config, hidden_input) if is_full else None
    position_ids = torch.zeros((1, 1), device=hidden_input.device, dtype=torch.long)
    kwargs = {"hidden_states": hidden_input.unsqueeze(1), "position_embeddings": position_embeddings, "position_ids": position_ids, "attention_mask": None, "past_key_values": None}
    with torch.inference_mode():
        try:
            output = layer(**kwargs)
        except TypeError:
            kwargs.pop("past_key_values", None)
            try:
                output = layer(**kwargs)
            except TypeError:
                kwargs.pop("position_ids", None)
                output = layer(**kwargs)
    if isinstance(output, tuple):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise RuntimeError(f"Unexpected reference output type: {type(output)!r}")
    return output.reshape(1, base.HIDDEN).float()


def _stats(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[float, float, float, float, float, float]:
    ref = reference.float().reshape(-1)
    got = candidate.float().reshape(-1)
    if ref.numel() != got.numel():
        raise ValueError(f"Shape mismatch: reference={tuple(reference.shape)} candidate={tuple(candidate.shape)}")
    diff = (got - ref).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    ref_norm = float(torch.linalg.vector_norm(ref).item())
    got_norm = float(torch.linalg.vector_norm(got).item())
    rel = max_abs / max(ref_norm / math.sqrt(ref.numel()), 1e-12)
    cosine = float(torch.dot(ref, got).item() / max(ref_norm * got_norm, 1e-12))
    return max_abs, mean_abs, rel, cosine, ref_norm, got_norm


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.6 layer-by-layer official fidelity probe")
    parser.add_argument("root", type=Path)
    parser.add_argument("--token-id", type=int, default=0)
    parser.add_argument("--layer", type=int, default=None, help="Only print this layer; earlier layers still run to build its input.")
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
    first_fail: int | None = None
    for layer_idx in range(base.DEFAULT_LAYERS):
        official_layer = layers[layer_idx]
        loaded, total = _materialize_layer(root, official_layer, layer_idx, args.device)
        hidden_ref = _run_reference_layer(root, official_layer, meta_model, config, hidden_ref, layer_idx)
        hidden_router = attention.step_attention(root, layer_idx, hidden_router, args.device)
        hidden_router, *_ = chat.batched_moe_step(root, layer_idx, hidden_router, top_k=8, device=args.device)
        hidden_router = hidden_router.detach().float()
        max_abs, mean_abs, rel, cosine, ref_norm, router_norm = _stats(hidden_ref, hidden_router)
        status = "PASS" if max_abs <= args.tolerance else "FAIL"
        if status == "FAIL" and first_fail is None:
            first_fail = layer_idx
        if args.layer is None or args.layer == layer_idx:
            kind = base.attention_type(root, layer_idx)
            print(f"L{layer_idx:02d} {status} kind={kind} loaded={loaded}/{total} | max_abs={max_abs:.6g} | mean_abs={mean_abs:.6g} | rel={rel:.6g} | cosine={cosine:.9f} | ref_norm={ref_norm:.6g} | router_norm={router_norm:.6g}")
        official_layer.to_empty(device="meta")
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
        if status == "FAIL" and args.stop_on_fail:
            break
    attention.deactivate(root)
    print("\n=== RESULT ===")
    if first_fail is None:
        print("status=PASS")
        return
    print(f"status=FAIL first_failing_layer={first_fail}")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
