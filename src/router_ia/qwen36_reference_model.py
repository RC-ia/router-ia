from __future__ import annotations

"""Helpers for loading a Transformers reference model on older GPUs.

Qwen3.6 checkpoints may advertise FP8 quantization. Older GPUs such as the
NVIDIA T4 cannot execute FP8 kernels directly, so the reference loader forces
Transformers to instantiate the ordinary PyTorch model and dequantize weights
when necessary.
"""

from pathlib import Path


def load_reference_model(root: Path, *, device: str):
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:
        raise SystemExit(
            "transformers is required. Install it with: pip install transformers"
        ) from exc

    config = AutoConfig.from_pretrained(str(root), local_files_only=True)

    # Recent Transformers releases inspect quantization_config during model
    # construction. On GPUs that cannot execute the advertised FP8 format,
    # disable the quantized construction path and let the checkpoint be
    # materialized as ordinary PyTorch tensors.
    if hasattr(config, "quantization_config"):
        config.quantization_config = None

    kwargs = {
        "config": config,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }

    if device == "cuda":
        import torch

        kwargs["dtype"] = torch.bfloat16
        # The reference model can exceed a single T4's VRAM. Accelerate may
        # spill complete modules to disk; from_pretrained requires an explicit
        # folder for that case, especially with MoE/internal weight formats.
        offload_folder = root / ".reference_offload"
        offload_folder.mkdir(parents=True, exist_ok=True)
        kwargs["device_map"] = "auto"
        kwargs["offload_folder"] = str(offload_folder)
        kwargs["offload_state_dict"] = True
    else:
        import torch

        kwargs["dtype"] = torch.float32

    try:
        model = AutoModelForCausalLM.from_pretrained(str(root), **kwargs)
    except TypeError:
        # Compatibility with older Transformers releases that still expose
        # torch_dtype rather than dtype in from_pretrained.
        kwargs.pop("dtype", None)
        import torch

        kwargs["torch_dtype"] = torch.bfloat16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(str(root), **kwargs)

    model.eval()
    return model
