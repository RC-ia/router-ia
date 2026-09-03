from __future__ import annotations

"""Isolated fidelity probe for Qwen3.6 linear-attention in_proj_z."""

import argparse
import gc
import json
from pathlib import Path

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
    _stage_stats,
)
from .qwen36_op_probe import dequantize_fp8_blockwise, load_embedding_row, rmsnorm


def report(name, reference, candidate, tolerance=1e-3):
    candidate = candidate.to(device=reference.device)
    s = _stage_stats(reference, candidate)
    status = "PASS" if s[0] <= tolerance else "FAIL"
    print(
        f"  {name:<30} {status} max_abs={s[0]:.6g} "
        f"mean_abs={s[1]:.6g} rel={s[2]:.6g} cosine={s[3]:.9f} "
        f"ref_norm={s[4]:.6g} router_norm={s[5]:.6g}"
    )
    return status == "PASS"


def checkpoint_index(root: Path) -> dict[str, str]:
    path = root / "model.safetensors.index.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        return dict(data["weight_map"])
    single = root / "model.safetensors"
    if single.is_file():
        return {"__single__": single.name}
    raise FileNotFoundError("No safetensors checkpoint found")


def filename(weight_map: dict[str, str], key: str) -> str:
    return weight_map.get("__single__", weight_map[key])


def raw_checkpoint_tensor(root: Path, weight_map: dict[str, str], key: str):
    from safetensors import safe_open

    with safe_open(str(root / filename(weight_map, key)), framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(key)
    scale = None
    if tensor.dtype == torch.float8_e4m3fn:
        scale_key = key + "_scale_inv"
        with safe_open(str(root / filename(weight_map, scale_key)), framework="pt", device="cpu") as handle:
            scale = handle.get_tensor(scale_key)
    return tensor, scale


def main():
    parser = argparse.ArgumentParser(description="Qwen3.6 isolated in_proj_z fidelity probe")
    parser.add_argument("root", type=Path)
    parser.add_argument("--token-id", type=int, default=0)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    if not 0 <= args.layer < base.DEFAULT_LAYERS:
        raise SystemExit(f"--layer must be in [0, {base.DEFAULT_LAYERS - 1}]")

    root = args.root.resolve()
    if base.attention_type(root, args.layer) != "linear_attention":
        raise SystemExit(f"Layer {args.layer} is not linear_attention")

    config = _load_config(root)
    model = _build_meta_model(config)
    layers = _find_layers(model)
    layer = layers[args.layer]
    loaded, total = _materialize_layer(root, layer, args.layer, args.device)

    print("op=in-proj-z-fidelity")
    print(f"layer={args.layer}")
    print(f"token_id={args.token_id}")
    print(f"device={args.device}")
    print(f"loaded={loaded}/{total}")
    print(f"tolerance={args.tolerance}")

    input_dtype = _module_input_dtype(layer)
    hidden = load_embedding_row(root, args.token_id).reshape(1, base.HIDDEN).to(args.device).float()
    norm_weight = base.load_layer_weight(root, args.layer, "input_layernorm.weight", args.device)
    normed = rmsnorm(hidden.to(dtype=input_dtype), norm_weight)

    prefix = base.layer_prefix(args.layer)
    z_key = prefix + "linear_attn.in_proj_z"

    captured: dict[str, torch.Tensor] = {}
    original_linear = F.linear

    def wrapped(x, weight, bias=None):
        y = original_linear(x, weight, bias)
        if y.shape[-1] == base.LINEAR_VALUE_DIM and "output" not in captured:
            captured["input"] = x.detach().clone()
            captured["weight"] = weight.detach().clone()
            captured["output"] = y.detach().clone()
        return y

    F.linear = wrapped
    try:
        with torch.no_grad():
            official_full = layer.linear_attn(hidden_states=normed.unsqueeze(1), cache_params=None, attention_mask=None)
            if isinstance(official_full, tuple):
                official_full = official_full[0]
    finally:
        F.linear = original_linear

    # The captured 4096-wide F.linear is the actual in_proj_z result. The
    # complete linear-attention module output is only the final 2048-wide
    # attention vector and must never be reshaped as LINEAR_VALUE_DIM.
    official_z = captured["output"].detach()

    state = attention.state_for(root, args.device)
    state.reset()
    attention.activate(root, state)
    try:
        router_w = attention._projection(root, z_key, args.device)
        router_input = captured["input"]
        router_output = F.linear(router_input.to(dtype=router_w.dtype), router_w)
    finally:
        attention.deactivate(root)

    print("\n=== SHAPES / DTYPES ===")
    print(f"  official input : shape={tuple(captured['input'].shape)} dtype={captured['input'].dtype}")
    print(f"  official weight: shape={tuple(captured['weight'].shape)} dtype={captured['weight'].dtype}")
    print(f"  router weight  : shape={tuple(router_w.shape)} dtype={router_w.dtype}")
    print(f"  official z     : shape={tuple(official_z.shape)} dtype={official_z.dtype}")
    print(f"  router z       : shape={tuple(router_output.shape)} dtype={router_output.dtype}")
    print(f"  full attn out  : shape={tuple(official_full.shape)} dtype={official_full.dtype}")

    print("\n=== INPUT ===")
    ok = [report("official_input_vs_normed", captured["input"], normed)]

    print("\n=== WEIGHT REPRESENTATION ===")
    ok.append(report("official_weight_vs_router_weight", captured["weight"], router_w, args.tolerance))

    weight_map = checkpoint_index(root)
    raw_weight, raw_scale = raw_checkpoint_tensor(root, weight_map, z_key)
    if raw_scale is not None:
        deq = dequantize_fp8_blockwise(raw_weight, raw_scale)
        ok.append(report("checkpoint_dequant_vs_router_weight", deq, router_w, args.tolerance))
        print(f"  checkpoint raw   : dtype={raw_weight.dtype} shape={tuple(raw_weight.shape)}")
        print(f"  checkpoint scale : dtype={raw_scale.dtype} shape={tuple(raw_scale.shape)}")
        print(f"  dequant dtype    : {deq.dtype}")
    else:
        deq = raw_weight.float()
        ok.append(report("checkpoint_float_vs_router_weight", deq, router_w, args.tolerance))
        print(f"  checkpoint raw   : dtype={raw_weight.dtype} shape={tuple(raw_weight.shape)}")

    print("\n=== SAME INPUT, DIFFERENT WEIGHTS ===")
    same_input = captured["input"]
    official_weight_output = original_linear(same_input, captured["weight"])
    router_weight_output = original_linear(same_input.to(dtype=router_w.dtype), router_w)
    ok += [
        report("official_layer_vs_captured_linear", official_z, captured["output"], args.tolerance),
        report("official_weight_linear_vs_router", official_weight_output, router_weight_output, args.tolerance),
        report("official_layer_vs_router", official_z, router_output, args.tolerance),
    ]

    print("\n=== INPUT DTYPE SENSITIVITY ===")
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        try:
            candidate = original_linear(same_input.to(dtype=dtype), router_w)
            print(f"  input={str(dtype):<18}", end="")
            s = _stage_stats(official_z, candidate)
            print(
                f" max_abs={s[0]:.6g} mean_abs={s[1]:.6g} "
                f"cosine={s[3]:.9f} norm={s[5]:.6g}"
            )
        except RuntimeError as exc:
            print(f"  input={str(dtype):<18} ERROR {exc}")

    print("\n=== RESULT ===")
    print(f"status={'PASS' if all(ok) else 'FAIL'}")

    layer.to_empty(device="meta")
    gc.collect()


if __name__ == "__main__": main()
