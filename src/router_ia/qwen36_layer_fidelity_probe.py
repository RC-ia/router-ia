from __future__ import annotations

"""Layer-by-layer fidelity probe for Qwen3.6.

Unlike the full-model fidelity probe, this test never materializes the whole
35B model.  The Transformers architecture is instantiated on ``meta`` and
only one decoder layer is materialized at a time.  Checkpoint tensors for that
layer are loaded directly from safetensors, with FP8 weights dequantized using
the same blockwise routine used by the router.

The probe then advances both implementations from the same embedding:

    reference Transformers layer N -> reference hidden N
    router layer N              -> router hidden N

Only one official layer is resident at a time, making the test practical on a
GPU/host that cannot hold the complete reference model.
"""

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
from .qwen36_mini_chat import load_tokenizer
from .qwen36_op_probe import dequantize_fp8_blockwise, load_embedding_row


DEFAULT_TOLERANCE = 1e-3


def _load_config(root: Path):
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(str(root), local_files_only=True)


def _text_config(config):
    return getattr(config, "text_config", config)


def _find_layers_container(model: torch.nn.Module) -> torch.nn.ModuleList:
    candidates = (
        ("model", "language_model", "layers"),
        ("language_model", "layers"),
        ("model", "layers"),
        ("layers",),
        ("transformer", "h"),
    )
    for path in candidates:
        obj: Any = model
        try:
            for name in path:
                obj = getattr(obj, name)
        except AttributeError:
            continue
        if isinstance(obj, torch.nn.ModuleList):
            return obj
    raise RuntimeError("Could not find decoder layer ModuleList in Transformers model")


def _build_meta_model(config) -> torch.nn.Module:
    from transformers import AutoModelForCausalLM

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model.eval()
    return model


def _checkpoint_index(root: Path) -> dict[str, str]:
    index_candidates = (
        root / "model.safetensors.index.json",
        root / "model.safetensors.json",
    )
    for path in index_candidates:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            weight_map = data.get("weight_map")
            if isinstance(weight_map, dict):
                return {str(k): str(v) for k, v in weight_map.items()}

    single = root / "model.safetensors"
    if single.is_file():
        return {"__single__": single.name}

    files = sorted(root.glob("*.safetensors"))
    if len(files) == 1:
        return {"__single__": files[0].name}
    if not files:
        raise FileNotFoundError(f"No safetensors checkpoint found under {root}")
    raise FileNotFoundError(
        "Multiple safetensors files found but no model.safetensors.index.json"
    )


def _checkpoint_file(weight_map: dict[str, str], key: str) -> str:
    if "__single__" in weight_map:
        return weight_map["__single__"]
    try:
        return weight_map[key]
    except KeyError as exc:
        raise KeyError(f"Checkpoint tensor not found: {key}") from exc


def _load_layer_from_checkpoint(
    root: Path,
    layer: torch.nn.Module,
    layer_idx: int,
    device: str,
) -> None:
    """Materialize one meta decoder layer from the checkpoint."""
    from safetensors import safe_open

    prefix = base.layer_prefix(layer_idx)
    weight_map = _checkpoint_index(root)

    # Materialize the layer first.  All parameters are then filled one by one,
    # avoiding a second complete layer-sized CPU state_dict.
    layer.to_empty(device=device)
    layer_state = layer.state_dict()

    # Keep one safetensors handle per shard rather than reopening for every key.
    handles: dict[str, Any] = {}
    try:
        for local_name, destination in layer_state.items():
            checkpoint_key = prefix + local_name
            # Some buffers are not persisted in the checkpoint.  They are
            # initialization-only/runtime buffers and should retain their
            # materialized value if possible.
            try:
                filename = _checkpoint_file(weight_map, checkpoint_key)
            except KeyError:
                continue

            handle = handles.get(filename)
            if handle is None:
                handle = safe_open(str(root / filename), framework="pt", device="cpu")
                handles[filename] = handle
            if checkpoint_key not in handle.keys():
                continue

            tensor = handle.get_tensor(checkpoint_key)
            scale_key = checkpoint_key + "_scale_inv"
            if tensor.dtype == torch.float8_e4m3fn:
                if scale_key not in handle.keys():
                    raise RuntimeError(
                        f"FP8 tensor {checkpoint_key} has no {scale_key}"
                    )
                scale = handle.get_tensor(scale_key)
                tensor = dequantize_fp8_blockwise(tensor, scale)

            target = destination
            target_dtype = target.dtype
            if target_dtype.is_floating_point:
                tensor = tensor.to(dtype=target_dtype)
            tensor = tensor.to(device=device)
            with torch.no_grad():
                destination.copy_(tensor)
            del tensor
    finally:
        for handle in handles.values():
            handle.__exit__(None, None, None)


def _find_rotary_class(meta_model: torch.nn.Module):
    candidates = (
        ("model", "language_model", "rotary_emb"),
        ("model", "rotary_emb"),
        ("language_model", "rotary_emb"),
        ("rotary_emb",),
    )
    for path in candidates:
        obj: Any = meta_model
        try:
            for name in path:
                obj = getattr(obj, name)
        except AttributeError:
            continue
        return obj.__class__
    return None


def _make_position_embeddings(meta_model: torch.nn.Module, config, hidden: torch.Tensor, position: int):
    rotary_cls = _find_rotary_class(meta_model)
    if rotary_cls is None:
        return None

    text_config = _text_config(config)
    try:
        rotary = rotary_cls(text_config)
    except TypeError:
        rotary = rotary_cls(config=text_config)
    rotary = rotary.to(device=hidden.device)
    position_ids = torch.tensor([[position]], device=hidden.device, dtype=torch.long)
    with torch.inference_mode():
        return rotary(hidden, position_ids=position_ids)


def _run_reference_layer(
    layer: torch.nn.Module,
    meta_model: torch.nn.Module,
    config,
    hidden: torch.Tensor,
    layer_idx: int,
    device: str,
) -> torch.Tensor:
    position_embeddings = None
    if base.attention_type_placeholder if False else False:
        pass

    kind = base.attention_type_placeholder if False else None
    # Determine the type from the checkpoint, not from the Python class name.
    is_full = base.attention_type(root=_CURRENT_ROOT, layer=layer_idx) == "full_attention"
    if is_full:
        position_embeddings = _make_position_embeddings(meta_model, config, hidden, layer_idx_position())

    position = layer_idx_position()
    if is_full:
        position_embeddings = _make_position_embeddings(meta_model, config, hidden, position)

    position_ids = torch.tensor([[position]], device=hidden.device, dtype=torch.long)
    kwargs = {
        "hidden_states": hidden.unsqueeze(1),
        "position_embeddings": position_embeddings,
        "position_ids": position_ids,
        "attention_mask": None,
        "past_key_values": None,
    }

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
        raise RuntimeError(f"Unexpected reference layer output type: {type(output)!r}")
    return output.reshape(1, base.HIDDEN).float()


_CURRENT_ROOT: Path | None = None
_CURRENT_POSITION = 0


def layer_idx_position() -> int:
    return _CURRENT_POSITION


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


def _run_router_layer(root: Path, layer_idx: int, hidden: torch.Tensor, device: str) -> torch.Tensor:
    state = attention.state_for(root, device)
    if layer_idx == 0:
        state.reset()
        attention.activate(root, state)
    residual = attention.step_attention(root, layer_idx, hidden, device)
    output, *_ = chat.batched_moe_step(root, layer_idx, residual, top_k=8, device=device)
    del residual
    return output.detach().float()


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.6 layer-by-layer official fidelity probe")
    parser.add_argument("root", type=Path)
    parser.add_argument("--token-id", type=int, default=0)
    parser.add_argument("--layer", type=int, default=None, help="Only report this layer; earlier layers are still advanced to build its input.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--stop-on-fail", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    if args.tolerance <= 0:
        raise SystemExit("--tolerance must be > 0")

    root = args.root.resolve()
    if args.layer is not None and not 0 <= args.layer < base.DEFAULT_LAYERS:
        raise SystemExit(f"--layer must be in [0, {base.DEFAULT_LAYERS - 1}]")

    global _CURRENT_ROOT, _CURRENT_POSITION
    _CURRENT_ROOT = root

    # Importing Transformers is intentionally delayed until after argument
    # validation. The full architecture is created on meta, so it consumes
    # negligible real memory.
    config = _load_config(root)
    meta_model = _build_meta_model(config)
    layers = _find_layers_container(meta_model)
    if len(layers) < base.DEFAULT_LAYERS:
        raise SystemExit(f"Expected {base.DEFAULT_LAYERS} layers, found {len(layers)}")

    tokenizer = load_tokenizer(root)
    # Keep the test deterministic and identical on both implementations.
    tokens = tokenizer.encode(str(args.token_id), add_special_tokens=False)
    del tokens
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
        _CURRENT_POSITION = layer_idx
        official_layer = layers[layer_idx]
        _load_layer_from_checkpoint(root, official_layer, layer_idx, args.device)

        hidden_ref = _run_reference_layer(
            official_layer,
            meta_model,
            config,
            hidden_ref,
            layer_idx,
            args.device,
        )
        hidden_router = _run_router_layer(root, layer_idx, hidden_router, args.device)
        state.tokens_seen = layer_idx + 1

        stats = _stats(hidden_ref, hidden_router)
        should_report = args.layer is None or args.layer == layer_idx
        status = "PASS" if stats[0] <= args.tolerance else "FAIL"
        if status == "FAIL" and first_fail is None:
            first_fail = layer_idx
        if should_report:
            print(
                f"L{layer_idx:02d} {status} | "
                f"max_abs={stats[0]:.6g} | mean_abs={stats[1]:.6g} | "
                f"rel={stats[2]:.6g} | cosine={stats[3]:.9f} | "
                f"ref_norm={stats[4]:.6g} | router_norm={stats[5]:.6g}"
            )

        # Release the official layer before the next one. The model itself
        # stays on meta and therefore does not consume layer-sized memory.
        del official_layer
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
