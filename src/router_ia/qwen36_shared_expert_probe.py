from __future__

"""Execute the Qwen3.6 Layer-0 shared expert in isolation."""

import argparse
import gc
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from .qwen36_moe8_probe import build_moe_input, load_expert_projection
from .qwen36_op_probe import load_tensor

HIDDEN = 2048
DEFAULT_LAYER = 0


def load_shared_projection(root: Path, layer: int, kind: str, device: str) -> torch.Tensor:
    prefix = f"model.language_model.layers.{layer}.mlp.shared_expert.{kind}"
    try:
        return load_expert_projection(root, layer, 0, f"shared_expert.{kind}", device)
    except KeyError:
        weight = load_tensor(root, prefix + ".weight", device="cpu")
        if weight.ndim != 2:
            raise ValueError(f"Unexpected shared expert {kind} weight shape: {tuple(weight.shape)}")
        out = weight.float().to(device)
        del weight
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Qwen3.6 Layer-0 shared expert")
    parser.add_argument("root", type=Path)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--token-id", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")

    root = args.root.resolve()
    device = args.device
    start_total = perf_counter()

    moe_in = build_moe_input(root, args.token_id, device)
    x = moe_in.reshape(1, HIDDEN).float()

    shared_gate_w = load_tensor(
        root,
        f"model.language_model.layers.{args.layer}.mlp.shared_expert_gate.weight",
        device=device,
    ).float()
    shared_gate = torch.sigmoid(F.linear(x, shared_gate_w))

    gate_w = load_shared_projection(root, args.layer, "gate_proj", device)
    up_w = load_shared_projection(root, args.layer, "up_proj", device)
    down_w = load_shared_projection(root, args.layer, "down_proj", device)

    gate = F.linear(x, gate_w)
    up = F.linear(x, up_w)
    hidden = F.silu(gate) * up
    raw_out = F.linear(hidden, down_w)
    out = raw_out * shared_gate

    if device == "cuda":
        torch.cuda.synchronize()
    total_ms = (perf_counter() - start_total) * 1000.0

    print("op=shared_expert")
    print(f"layer: {args.layer}")
    print(f"token id: {args.token_id}")
    print(f"moe input shape: {tuple(moe_in.shape)}")
    print(f"shared gate weight shape: {tuple(shared_gate_w.shape)}")
    print(f"shared gate value: {shared_gate.item():.8f}")
    print(f"gate shape: {tuple(gate.shape)}")
    print(f"up shape: {tuple(up.shape)}")
    print(f"down input shape: {tuple(hidden.shape)}")
    print(f"shared raw output shape: {tuple(raw_out.shape)}")
    print(f"shared output shape: {tuple(out.shape)}")
    print(f"moe input norm: {torch.linalg.vector_norm(x).item():.8f}")
    print(f"gate norm: {torch.linalg.vector_norm(gate).item():.8f}")
    print(f"up norm: {torch.linalg.vector_norm(up).item():.8f}")
    print(f"gated hidden norm: {torch.linalg.vector_norm(hidden).item():.8f}")
    print(f"raw shared output norm: {torch.linalg.vector_norm(raw_out).item():.8f}")
    print(f"shared output norm: {torch.linalg.vector_norm(out).item():.8f}")
    print(f"shared output mean: {out.mean().item():.8f}")
    print(f"shared output min: {out.min().item():.8f}")
    print(f"shared output max: {out.max().item():.8f}")
    print(f"total time: {total_ms:.3f} ms")

    del moe_in, x, shared_gate_w, shared_gate, gate_w, up_w, down_w, gate, up, hidden, raw_out, out
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
