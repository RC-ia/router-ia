from __future__ import annotations

"""Probe Gated Delta Rule recurrence across two tokens.

This intentionally reuses the isolated Layer-0 helpers and keeps the recurrent
state alive between tokens. The convolution history is reset for each token;
this probe is specifically meant to validate Delta Rule state persistence.
"""

import argparse
import gc
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from .qwen36_op_probe import (
    HEAD_DIM,
    LAYER_PREFIX,
    NUM_V_HEADS,
    EPS,
    compute_conv,
    load_embedding_row,
    load_projection,
    load_tensor,
    rmsnorm,
    split_qkv,
    stats,
)


def token_params(root: Path, token_id: int, device: str, norm_weight: torch.Tensor,
                 a_weight: torch.Tensor, b_weight: torch.Tensor,
                 a_log: torch.Tensor, dt_bias: torch.Tensor):
    conv = compute_conv(root, token_id, device)
    _, _, v, q, k = split_qkv(conv)
    h = rmsnorm(load_embedding_row(root, token_id).to(device), norm_weight)

    a_raw = F.linear(h.float(), a_weight.float()).reshape(1, NUM_V_HEADS)
    b_raw = F.linear(h.float(), b_weight.float()).reshape(1, NUM_V_HEADS)
    beta = torch.sigmoid(b_raw)
    g = -torch.exp(a_log) * F.softplus(a_raw + dt_bias)
    decay = torch.exp(g)

    q = F.normalize(q.float(), dim=-1, eps=EPS)
    k = F.normalize(k.float(), dim=-1, eps=EPS)
    q = q * (HEAD_DIM ** -0.5)

    del conv, h, a_raw, b_raw
    return q, k, v.float(), beta, g, decay


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.6 two-token Delta Rule recurrence probe")
    parser.add_argument("root", type=Path)
    parser.add_argument("--token-id", type=int, default=0)
    parser.add_argument("--token-id-2", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    root = args.root.resolve()

    norm_weight = load_tensor(root, LAYER_PREFIX + "input_layernorm.weight", device=args.device)
    a_weight = load_projection(root, LAYER_PREFIX + "linear_attn.in_proj_a", args.device)
    b_weight = load_projection(root, LAYER_PREFIX + "linear_attn.in_proj_b", args.device)
    a_log = load_tensor(root, LAYER_PREFIX + "linear_attn.A_log", device=args.device).float().reshape(1, NUM_V_HEADS)
    dt_bias = load_tensor(root, LAYER_PREFIX + "linear_attn.dt_bias", device=args.device).float().reshape(1, NUM_V_HEADS)

    state = torch.zeros(1, NUM_V_HEADS, HEAD_DIM, HEAD_DIM, device=args.device, dtype=torch.float32)

    print("op=delta_sequence")
    print(f"token 1 id={args.token_id}")
    print(f"token 2 id={args.token_id_2}")
    print("state shape=(1,32,128,128) dtype=float32")

    results = []
    for index, token_id in enumerate((args.token_id, args.token_id_2), start=1):
        start = perf_counter()
        q, k, v, beta, g, decay = token_params(
            root, token_id, args.device, norm_weight, a_weight, b_weight, a_log, dt_bias
        )

        state = state * decay.unsqueeze(-1).unsqueeze(-1)
        retrieved = torch.einsum("bhkd,bhk->bhd", state, k)
        delta = (v - retrieved) * beta.unsqueeze(-1)
        state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
        out = torch.einsum("bhkd,bhk->bhd", state, q)

        if args.device == "cuda":
            torch.cuda.synchronize()
        elapsed = (perf_counter() - start) * 1000.0

        print(f"\ntoken {index}")
        print(f"id: {token_id}")
        print(f"beta: mean={beta.mean().item():.8f} min={beta.min().item():.8f} max={beta.max().item():.8f}")
        print(f"decay: mean={decay.mean().item():.8f} min={decay.min().item():.8f} max={decay.max().item():.8f}")
        print(f"retrieved norm: {torch.linalg.vector_norm(retrieved).item():.8f}")
        print(f"delta norm: {torch.linalg.vector_norm(delta).item():.8f}")
        print(f"state norm: {torch.linalg.vector_norm(state).item():.8f}")
        print(f"output norm: {torch.linalg.vector_norm(out).item():.8f}")
        print(f"token time: {elapsed:.3f} ms")
        results.append((retrieved.detach().clone(), out.detach().clone()))

        del q, k, v, beta, g, decay, retrieved, delta, out
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    first_retrieved, first_out = results[0]
    second_retrieved, second_out = results[1]
    print("\nrecurrence check")
    print(f"token 1 retrieved norm: {torch.linalg.vector_norm(first_retrieved).item():.8f}")
    print(f"token 2 retrieved norm: {torch.linalg.vector_norm(second_retrieved).item():.8f}")
    print(f"token 1 output norm: {torch.linalg.vector_norm(first_out).item():.8f}")
    print(f"token 2 output norm: {torch.linalg.vector_norm(second_out).item():.8f}")

    if torch.linalg.vector_norm(second_retrieved).item() > 1e-8:
        print("RESULT: recurrent state is active; token 2 reads token 1 memory.")
    else:
        print("RESULT: token 2 retrieved ~0; recurrence did not retain a measurable state.")

    del state, norm_weight, a_weight, b_weight, a_log, dt_bias, first_retrieved, second_retrieved, first_out, second_out
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
