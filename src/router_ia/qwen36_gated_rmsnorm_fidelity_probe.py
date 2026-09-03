from __future__ import annotations

"""Stage-by-stage fidelity probe for Qwen3.6 Gated RMSNorm."""

import argparse
import gc
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
from .qwen36_op_probe import load_embedding_row, rmsnorm

DEFAULT_TOLERANCE = 1e-3


def report(name: str, reference: torch.Tensor, candidate: torch.Tensor, tolerance: float) -> bool:
    s = _stage_stats(reference, candidate)
    status = "PASS" if s[0] <= tolerance else "FAIL"
    print(
        f"  {name:<28} {status} max_abs={s[0]:.6g} mean_abs={s[1]:.6g} "
        f"rel={s[2]:.6g} cosine={s[3]:.9f} "
        f"ref_norm={s[4]:.6g} router_norm={s[5]:.6g}"
    )
    return status == "PASS"


def isolated_reference(x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor):
    """Literal implementation of the expected Qwen Gated RMSNorm semantics."""
    input_dtype = x.dtype
    x_fp32 = x.float()
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    normalized_fp32 = x_fp32 * torch.rsqrt(variance + 1e-6)
    normalized = normalized_fp32.to(input_dtype)
    weighted = weight.reshape(1, 1, -1) * normalized
    gate = F.silu(z.float())
    output = (weighted * gate).to(input_dtype)
    return {
        "variance": variance,
        "normalized": normalized,
        "weighted": weighted,
        "gate": gate,
        "output": output,
    }


def router_gated_norm(x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor):
    return attention.gated_rmsnorm(x, z, weight)


def run(root: Path, layer, layer_idx: int, hidden: torch.Tensor, device: str, tolerance: float) -> bool:
    dtype = _module_input_dtype(layer)
    hidden = hidden.to(dtype=dtype)

    input_weight = base.load_layer_weight(root, layer_idx, "input_layernorm.weight", device)
    normed = rmsnorm(hidden, input_weight)

    def official_attention():
        y = layer.linear_attn(hidden_states=normed.unsqueeze(1), cache_params=None, attention_mask=None)
        if isinstance(y, tuple):
            y = y[0]
        return y

    official_capture: dict[str, torch.Tensor] = {}
    original_forward = layer.linear_attn.norm.forward

    def wrapped_forward(x, z, *args, **kwargs):
        official_capture["x"] = x.detach().clone()
        official_capture["z"] = z.detach().clone()
        y = original_forward(x, z, *args, **kwargs)
        official_capture["y"] = y.detach().clone() if torch.is_tensor(y) else y[0].detach().clone()
        return y

    layer.linear_attn.norm.forward = wrapped_forward
    try:
        official_attention()
    finally:
        layer.linear_attn.norm.forward = original_forward

    # Transformers can expose per-head tensors as (heads, head_dim), while
    # the router contract is (batch, heads, head_dim). Canonicalize the
    # representation; RMSNorm itself always operates over the final axis.
    x = official_capture["x"]
    z = official_capture["z"]
    official_y = official_capture["y"]
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if z.ndim == 2:
        z = z.unsqueeze(0)
    if official_y.ndim == 2:
        official_y = official_y.unsqueeze(0)

    weight = base.load_layer_weight(root, layer_idx, "linear_attn.norm.weight", device)
    ref_formula = isolated_reference(x, z, weight)
    router_result = router_gated_norm(x, z, weight)
    router_y = router_result[0]

    print("\n=== INPUTS ===")
    print(f"  x dtype={x.dtype} shape={tuple(x.shape)}")
    print(f"  z dtype={z.dtype} shape={tuple(z.shape)}")
    print(f"  weight dtype={weight.dtype} shape={tuple(weight.shape)}")

    print("\n=== OFFICIAL vs LITERAL FORMULA ===")
    ok = [report("official_output", official_y, ref_formula["output"], tolerance)]

    print("\n=== GATED RMSNORM INTERNALS ===")
    official_variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    official_normalized = (x.float() * torch.rsqrt(official_variance + 1e-6)).to(x.dtype)
    official_weighted = weight.reshape(1, 1, -1) * official_normalized
    official_gate = F.silu(z.float())

    ok += [
        report("variance", official_variance, ref_formula["variance"], tolerance),
        report("normalized", official_normalized, ref_formula["normalized"], tolerance),
        report("weighted", official_weighted, ref_formula["weighted"], tolerance),
        report("silu_gate", official_gate, ref_formula["gate"], tolerance),
    ]

    print("\n=== ROUTER FUNCTION (SAME INPUTS) ===")
    ok += [
        report("same_input_output", ref_formula["output"], router_y, tolerance),
        report("official_vs_router", official_y, router_y, tolerance),
    ]

    print("\n=== ORDER / DTYPE SENSITIVITY ===")
    normalized_fp32 = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    variant_weight_fp32 = weight.float().reshape(1, 1, -1) * normalized_fp32
    variant_gate = F.silu(z.float())
    variant_fp32_then_cast = (variant_weight_fp32 * variant_gate).to(x.dtype)
    variant_cast_before_weight = weight.reshape(1, 1, -1) * normalized_fp32.to(x.dtype)
    variant_cast_before_gate = (variant_cast_before_weight * z.to(x.dtype).sigmoid()).to(x.dtype)
    ok += [
        report("expected_vs_fp32_weight", ref_formula["output"], variant_fp32_then_cast, tolerance),
        report("expected_vs_sigmoid", ref_formula["output"], variant_cast_before_gate, tolerance),
    ]

    print("\n=== RESULT ===")
    print(f"status={'PASS' if all(ok) else 'FAIL'}")
    return all(ok)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.6 Gated RMSNorm fidelity probe")
    parser.add_argument("root", type=Path)
    parser.add_argument("--token-id", type=int, default=0)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
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

    print("op=gated-rmsnorm-fidelity")
    print(f"layer={args.layer}")
    print(f"token_id={args.token_id}")
    print(f"device={args.device}")
    print(f"loaded={loaded}/{total}")
    print(f"tolerance={args.tolerance}")

    hidden = load_embedding_row(root, args.token_id).reshape(1, base.HIDDEN).to(args.device).float()
    try:
        run(root, layer, args.layer, hidden, args.device, args.tolerance)
    finally:
        layer.to_empty(device="meta")
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
