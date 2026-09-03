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

    # Some recent Transformers releases inspect quantization_config during
    # model construction and assume it is populated when an FP8 checkpoint is
    # detected. On pre-FP8 GPUs we explicitly disable that path; the safetensor
    # loader can then materialize the weights in the requested dtype.
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
        kwargs["device_map"] = "auto"
    else:
        import torch

        kwargs["dtype"] = torch.float32

    try:
        model = AutoModelForCausalLM.from_pretrained(str(root), **kwargs)
    except TypeError:
        # Compatibility with Transformers versions that still expose
        # torch_dtype rather than dtype in from_pretrained.
        kwargs.pop("dtype", None)
        import torch
        kwargs["torch_dtype"] = torch.bfloat16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(str(root), **kwargs)

    model.eval()
    return model
