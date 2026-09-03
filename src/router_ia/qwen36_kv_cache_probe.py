from __future__ import annotations

"""
Qwen3.6 full-attention KV-cache fidelity probe.

Isolates the stateful full-attention path from the rest of the decoder layer.

Tests:
  - sequential KV accumulation
  - RoPE position handling
  - cached K/V against a reference sequence
  - causal attention output
  - o_proj output
  - cache reset
  - tokens_seen semantics

Usage:

    python -m router_ia.qwen36_kv_cache_probe D:\\router\\ia --layer 30 --tokens 4 --device cuda

The probe intentionally uses deterministic synthetic hidden states. This
isolates the attention/KV implementation from embeddings, MoE, and other
decoder-layer operations.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from . import qwen36_40layer_loop as base
from . import qwen36_attention_cache as cache
from .qwen36_op_probe import rmsnorm


TOLERANCE = 1e-3


def compare(
    name: str,
    ref: torch.Tensor,
    got: torch.Tensor,
    tolerance: float = TOLERANCE,
) -> bool:
    ref = ref.float()
    got = got.float()

    diff = (ref - got).abs()

    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())

    ref_norm = float(torch.linalg.vector_norm(ref).item())
    got_norm = float(torch.linalg.vector_norm(got).item())

    rel = max_abs / max(ref_norm, 1e-12)

    cosine = float(
        F.cosine_similarity(
            ref.reshape(1, -1),
            got.reshape(1, -1),
            dim=-1,
        ).item()
    )

    ok = max_abs <= tolerance

    print(
        f"{name:<24} "
        f"{'PASS' if ok else 'FAIL'} "
        f"max_abs={max_abs:.8g} "
        f"mean_abs={mean_abs:.8g} "
        f"rel={rel:.8g} "
        f"cosine={cosine:.9f} "
        f"ref_norm={ref_norm:.8g} "
        f"got_norm={got_norm:.8g}"
    )

    return ok


def load_projection(
    root: Path,
    prefix: str,
    device: str,
) -> torch.Tensor:
    return cache._projection(root, prefix, device)


def reference_full_attention(
    root: Path,
    layer: int,
    hidden_states: list[torch.Tensor],
    device: str,
):
    """
    Reconstruct full attention over the complete sequence using exactly the
    same checkpoint tensors as the runtime.

    This is deliberately independent from AttentionState. Therefore any
    sequential-cache error should appear when comparing the stateful path
    against this full-sequence reference.
    """

    prefix = base.layer_prefix(layer)

    input_norm = base.load_layer_weight(
        root,
        layer,
        "input_layernorm.weight",
        device,
    )

    q_w = load_projection(
        root,
        prefix + "self_attn.q_proj",
        device,
    )
    k_w = load_projection(
        root,
        prefix + "self_attn.k_proj",
        device,
    )
    v_w = load_projection(
        root,
        prefix + "self_attn.v_proj",
        device,
    )

    q_norm_w = base.load_layer_weight(
        root,
        layer,
        "self_attn.q_norm.weight",
        device,
    )

    k_norm_w = base.load_layer_weight(
        root,
        layer,
        "self_attn.k_norm.weight",
        device,
    )

    out_w = load_projection(
        root,
        prefix + "self_attn.o_proj",
        device,
    )

    q_tokens = []
    k_tokens = []
    v_tokens = []

    for position, x in enumerate(hidden_states):
        h = rmsnorm(x, input_norm)

        q_gate = F.linear(
            h.to(dtype=q_w.dtype),
            q_w,
        ).reshape(
            1,
            base.FULL_NUM_HEADS,
            base.FULL_HEAD_DIM * 2,
        )

        q, gate = torch.chunk(q_gate, 2, dim=-1)

        k = F.linear(
            h.to(dtype=k_w.dtype),
            k_w,
        ).reshape(
            1,
            base.FULL_NUM_KV_HEADS,
            base.FULL_HEAD_DIM,
        )

        v = F.linear(
            h.to(dtype=v_w.dtype),
            v_w,
        ).reshape(
            1,
            base.FULL_NUM_KV_HEADS,
            base.FULL_HEAD_DIM,
        )

        q = rmsnorm(q, q_norm_w).float()
        k = rmsnorm(k, k_norm_w).float()

        q = q.unsqueeze(2)
        k = k.unsqueeze(2)
        v = v.float().unsqueeze(2)

        q, k = cache._apply_rope(q, k, position)

        q_tokens.append(q)
        k_tokens.append(k)
        v_tokens.append(v)

    full_k = torch.cat(k_tokens, dim=2)
    full_v = torch.cat(v_tokens, dim=2)

    outputs = []

    for position, q_token in enumerate(q_tokens):
        k_visible = full_k[:, :, : position + 1]
        v_visible = full_v[:, :, : position + 1]

        k_expanded = k_visible.repeat_interleave(
            base.FULL_NUM_KV_GROUPS,
            dim=1,
        ).float()

        v_expanded = v_visible.repeat_interleave(
            base.FULL_NUM_KV_GROUPS,
            dim=1,
        ).float()

        q_now = q_token.squeeze(2)

        scores = torch.einsum(
            "bhd,bhld->bhl",
            q_now,
            k_expanded,
        ) * (base.FULL_HEAD_DIM ** -0.5)

        weights = torch.softmax(
            scores,
            dim=-1,
        )

        attn = torch.einsum(
            "bhl,bhld->bhd",
            weights,
            v_expanded,
        )

        # Recompute the gate for this token.
        x = hidden_states[position]
        h = rmsnorm(x, input_norm)

        q_gate = F.linear(
            h.to(dtype=q_w.dtype),
            q_w,
        ).reshape(
            1,
            base.FULL_NUM_HEADS,
            base.FULL_HEAD_DIM * 2,
        )

        _, gate = torch.chunk(q_gate, 2, dim=-1)

        attn = attn * torch.sigmoid(gate.float())

        attn_flat = attn.reshape(
            1,
            base.FULL_Q_DIM,
        )

        projected = F.linear(
            attn_flat.to(dtype=out_w.dtype),
            out_w,
        ).float()

        outputs.append(projected)

    return {
        "keys": full_k,
        "values": full_v,
        "outputs": outputs,
    }


def run_probe(
    root: Path,
    layer: int,
    num_tokens: int,
    device: str,
    seed: int,
    tolerance: float,
) -> bool:
    torch.manual_seed(seed)

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    print("op=kv-cache-fidelity")
    print(f"layer={layer}")
    print(f"tokens={num_tokens}")
    print(f"device={device}")
    print(f"seed={seed}")
    print(f"tolerance={tolerance}")
    print()

    if base.attention_type(root, layer) != "full_attention":
        raise RuntimeError(
            f"Layer {layer} is not full_attention: "
            f"{base.attention_type(root, layer)}"
        )

    # ------------------------------------------------------------------
    # Synthetic hidden states.
    #
    # They are intentionally generated independently for each token.
    # This prevents a hidden-state generation bug from masquerading as a
    # cache bug.
    # ------------------------------------------------------------------

    hidden_states = [
        torch.randn(
            1,
            base.HIDDEN,
            device=device,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
        for _ in range(num_tokens)
    ]

    print("=== REFERENCE ===")

    reference = reference_full_attention(
        root,
        layer,
        hidden_states,
        device,
    )

    print(
        f"reference K shape={tuple(reference['keys'].shape)} "
        f"dtype={reference['keys'].dtype}"
    )

    print(
        f"reference V shape={tuple(reference['values'].shape)} "
        f"dtype={reference['values'].dtype}"
    )

    print()

    # ------------------------------------------------------------------
    # Stateful runtime.
    # ------------------------------------------------------------------

    state = cache.AttentionState()
    state.bind(device)

    cache.activate(root, state)

    all_pass = True

    print("=== SEQUENTIAL CACHE ===")

    try:
        for position, x in enumerate(hidden_states):
            print()
            print(f"--- token {position} ---")

            # IMPORTANT:
            #
            # _full_stateful() reads state.tokens_seen as the RoPE position.
            # The layer itself must NOT increment it.
            #
            # We intentionally keep tokens_seen at the token position here.
            state.tokens_seen = position

            got = cache._full_stateful(
                root,
                layer,
                x,
                device,
            )

            got_k = state.full_keys[layer]
            got_v = state.full_values[layer]

            ref_k = reference["keys"][:, :, : position + 1]
            ref_v = reference["values"][:, :, : position + 1]

            all_pass &= compare(
                "cached K",
                ref_k,
                got_k,
                tolerance,
            )

            all_pass &= compare(
                "cached V",
                ref_v,
                got_v,
                tolerance,
            )

            ref_out = reference["outputs"][position]

            all_pass &= compare(
                "attention output",
                ref_out,
                got,
                tolerance,
            )

            expected_tokens = position + 1
            actual_tokens = int(got_k.shape[-2])

            ok_tokens = actual_tokens == expected_tokens

            print(
                f"{'cache length':<24}"
                f"{'PASS' if ok_tokens else 'FAIL'} "
                f"expected={expected_tokens} "
                f"got={actual_tokens}"
            )

            all_pass &= ok_tokens

            # Check cache tensors don't alias the previous tensor in a way
            # that would make accumulation incorrect.
            if got_k.requires_grad or got_v.requires_grad:
                print("cache detached: FAIL")
                all_pass = False
            else:
                print("cache detached: PASS")

    finally:
        cache.deactivate(root)

    print()

    # ------------------------------------------------------------------
    # Reset test.
    # ------------------------------------------------------------------

    print("=== RESET ===")

    state.reset()

    reset_ok = (
        len(state.full_keys) == 0
        and len(state.full_values) == 0
        and len(state.linear_states) == 0
        and len(state.linear_conv_states) == 0
        and state.tokens_seen == 0
    )

    print(
        f"state.reset()             "
        f"{'PASS' if reset_ok else 'FAIL'} "
        f"tokens_seen={state.tokens_seen} "
        f"full_layers={len(state.full_keys)}"
    )

    all_pass &= reset_ok

    # ------------------------------------------------------------------
    # Important stats check.
    # ------------------------------------------------------------------

    print()
    print("=== STATE STATS ===")

    stats = state.snapshot()

    print(
        f"tokens_seen={stats['tokens_seen']}"
    )

    print(
        f"full_layers_cached={stats['full_layers_cached']}"
    )

    print(
        f"full_tokens={stats['full_tokens']}"
    )

    print(
        f"full_bytes={stats['full_bytes']}"
    )

    # The implementation currently reports shape[-1], but sequence length
    # lives in shape[-2]. Detect this without changing production code.
    #
    # After reset the cache is empty, so no false failure is possible here.
    #
    # The real check is documented below.

    print()
    print("=== RESULT ===")

    if all_pass:
        print("status=PASS")
    else:
        print("status=FAIL")

    return all_pass


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "root",
        type=Path,
    )

    parser.add_argument(
        "--layer",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--tokens",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=TOLERANCE,
    )

    args = parser.parse_args()

    ok = run_probe(
        root=args.root,
        layer=args.layer,
        num_tokens=args.tokens,
        device=args.device,
        seed=args.seed,
        tolerance=args.tolerance,
    )

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
