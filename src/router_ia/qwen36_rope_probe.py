from __future__ import annotations

"""Small fidelity probe for Qwen3.6 text RoPE.

This intentionally does not load the 35B model. It validates the local RoPE
implementation against the reference configuration used by Transformers.
"""

import argparse

import torch

from .qwen36_attention_cache import ROPE_DIM, ROPE_THETA, _apply_rope, _rope


def reference_rope(position: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    dim = ROPE_DIM
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    angles = float(position) * inv_freq
    emb = torch.cat((angles, angles), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Qwen3.6 text RoPE implementation")
    parser.add_argument("--position", type=int, default=17)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")

    device = torch.device(args.device)
    q = torch.randn(1, 16, 1, 256, device=device, dtype=torch.float32)
    k = torch.randn(1, 2, 1, 256, device=device, dtype=torch.float32)

    cos_local, sin_local = _rope(args.position, device, q.dtype)
    cos_ref, sin_ref = reference_rope(args.position, device, q.dtype)
    cos_error = (cos_local - cos_ref).abs().max().item()
    sin_error = (sin_local - sin_ref).abs().max().item()

    # Expand K only after rotation, matching the full-attention pipeline.
    q_rot, k_rot = _apply_rope(q, k, args.position)
    if q_rot.shape != q.shape or k_rot.shape != k.shape:
        raise RuntimeError(
            f"RoPE changed tensor shapes: q={tuple(q_rot.shape)} k={tuple(k_rot.shape)}"
        )

    # Dimensions beyond the partial rotary span must remain unchanged.
    q_pass_error = (q_rot[..., ROPE_DIM:] - q[..., ROPE_DIM:]).abs().max().item()
    k_pass_error = (k_rot[..., ROPE_DIM:] - k[..., ROPE_DIM:]).abs().max().item()

    print(f"rope_theta={ROPE_THETA:g}")
    print(f"rope_dim={ROPE_DIM}")
    print(f"position={args.position}")
    print(f"cos_max_error={cos_error:.3e}")
    print(f"sin_max_error={sin_error:.3e}")
    print(f"q_shape={tuple(q_rot.shape)}")
    print(f"k_shape={tuple(k_rot.shape)}")
    print(f"q_pass_through_error={q_pass_error:.3e}")
    print(f"k_pass_through_error={k_pass_error:.3e}")

    if max(cos_error, sin_error, q_pass_error, k_pass_error) > 1e-6:
        raise SystemExit("RoPE fidelity check failed")
    print("status=PASS")


if __name__ == "__main__":
    main()
