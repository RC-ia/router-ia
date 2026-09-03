from __future__ import annotations

"""Compare the router forward against Transformers layer-by-layer.

The probe uses the same tokenizer/input_ids in both paths and compares the
hidden state after every transformer layer plus final logits. It is intended
to isolate numerical/architectural drift before sampling is involved.
"""

import argparse
import gc
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from . import qwen36_attention_cache as attention
from . import qwen36_chat_batch as chat
from . import qwen36_40layer_loop as base
from . import qwen36_cached_loop as cached
from .qwen36_mini_chat import find_tensor_name, load_tokenizer
from .qwen36_op_probe import rmsnorm
from .qwen36_reference_model import load_reference_model

DEFAULT_TOP_K = 10
DEFAULT_TOLERANCE = 1e-3


def _model_input_device(model: torch.nn.Module, fallback: str) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(fallback)


def _reference_forward(
    model: torch.nn.Module, input_ids: torch.Tensor
) -> tuple[list[torch.Tensor], torch.Tensor]:
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    hidden_states = list(outputs.hidden_states or [])
    expected = base.DEFAULT_LAYERS + 1
    if len(hidden_states) < expected:
        raise RuntimeError(
            f"Expected embeddings + {base.DEFAULT_LAYERS} layer hidden states, got {len(hidden_states)}"
        )
    logits = outputs.logits[:, -1, :].detach().float()
    return hidden_states, logits


def _load_router_output_weights(
    root: Path, device: str
) -> tuple[str, torch.Tensor, str, torch.Tensor]:
    norm_name = find_tensor_name(
        root,
        ("language_model.norm.weight", "model.norm.weight", ".norm.weight"),
    )
    lm_name = find_tensor_name(root, ("lm_head.weight",))
    norm = cached._cached_load_tensor(root, norm_name, device=device).float()
    lm_head = cached._cached_load_tensor(root, lm_name, device=device).float()
    return norm_name, norm, lm_name, lm_head


def _router_forward(
    root: Path, token_ids: list[int], device: str
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    state = attention.state_for(root, device)
    state.reset()
    attention.activate(root, state)
    layer_hidden_states: list[torch.Tensor] = []
    final_x: torch.Tensor | None = None

    try:
        with torch.inference_mode():
            for token_id in token_ids:
                x = (
                    base.load_embedding_row(root, int(token_id))
                    .reshape(1, base.HIDDEN)
                    .to(device)
                    .float()
                )
                current_layer_states: list[torch.Tensor] = []

                for layer in range(base.DEFAULT_LAYERS):
                    residual = attention.step_attention(root, layer, x, device)
                    x, *_ = chat.batched_moe_step(
                        root, layer, residual, top_k=8, device=device
                    )
                    current_layer_states.append(x.detach().float())
                    del residual

                # Keep the states from the final prompt position. These align
                # with outputs.hidden_states[layer + 1][:, -1, :].
                layer_hidden_states = current_layer_states
                final_x = x.detach().float()
                state.tokens_seen += 1
                del x, current_layer_states

        if final_x is None or len(layer_hidden_states) != base.DEFAULT_LAYERS:
            raise RuntimeError("Router forward did not produce the expected states")

        _, norm, _, lm_head = _load_router_output_weights(root, device)
        final_norm = rmsnorm(final_x, norm).float()
        logits = F.linear(final_norm, lm_head).float()
        return layer_hidden_states, final_norm, logits
    finally:
        attention.deactivate(root)


def _stats(
    reference: torch.Tensor, candidate: torch.Tensor
) -> tuple[float, float, float, float]:
    ref = reference.float().reshape(-1)
    got = candidate.float().reshape(-1)
    if ref.numel() != got.numel():
        raise ValueError(
            f"Shape mismatch: reference={tuple(reference.shape)} candidate={tuple(candidate.shape)}"
        )
    diff = (got - ref).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    ref_norm = float(torch.linalg.vector_norm(ref).item())
    got_norm = float(torch.linalg.vector_norm(got).item())
    rel = max_abs / max(ref_norm / math.sqrt(ref.numel()), 1e-12)
    denom = max(ref_norm * got_norm, 1e-12)
    cosine = float(torch.dot(ref, got).item() / denom)
    return max_abs, mean_abs, rel, cosine


def _top_ids(logits: torch.Tensor, k: int) -> list[int]:
    return [int(x) for x in torch.topk(logits.reshape(-1), k).indices.cpu().tolist()]


def _format_ids(ids: list[int]) -> str:
    return ", ".join(str(x) for x in ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.6 official-vs-router fidelity probe")
    parser.add_argument("root", type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--max-prompt-tokens", type=int, default=32)
    parser.add_argument("--chat-template", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    if args.top_k < 1 or args.max_prompt_tokens < 1:
        raise SystemExit("--top-k and --max-prompt-tokens must be >= 1")

    root = args.root.resolve()
    tokenizer = load_tokenizer(root)

    if args.chat_template:
        if not hasattr(tokenizer, "apply_chat_template"):
            raise SystemExit("Tokenizer has no chat template support")
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(encoded, dict):
            encoded = encoded["input_ids"]
        token_ids = [int(x) for x in encoded.reshape(-1).tolist()]
    else:
        token_ids = [int(x) for x in tokenizer.encode(args.prompt, add_special_tokens=False)]

    if not token_ids:
        raise SystemExit("Prompt produced no tokens")
    if len(token_ids) > args.max_prompt_tokens:
        print(
            f"warning=prompt has {len(token_ids)} tokens; using last {args.max_prompt_tokens} tokens"
        )
        token_ids = token_ids[-args.max_prompt_tokens :]

    print("op=fidelity")
    print(f"prompt={args.prompt!r}")
    print(f"chat_template={args.chat_template}")
    print(f"prompt_tokens={len(token_ids)}")
    print(f"token_ids={_format_ids(token_ids)}")
    print(f"device={args.device}")
    print(f"tolerance={args.tolerance:g}")

    print("\n[1/3] official Transformers forward...")
    try:
        model = load_reference_model(root, device=args.device)
        ref_device = _model_input_device(model, args.device)
        ref_input = torch.tensor([token_ids], device=ref_device, dtype=torch.long)
        ref_states, ref_logits = _reference_forward(model, ref_input)
    except Exception as exc:
        raise SystemExit(f"Reference model forward failed: {exc}") from exc

    print("[2/3] router forward...")
    router_states, router_final_norm, router_logits = _router_forward(
        root, token_ids, args.device
    )

    print("[3/3] comparing...")
    failing_layer: int | None = None
    print("\n=== LAYER DIFFS ===")
    for layer in range(base.DEFAULT_LAYERS):
        reference = ref_states[layer + 1][:, -1, :]
        candidate = router_states[layer]
        max_abs, mean_abs, rel, cosine = _stats(reference, candidate)
        status = "PASS" if max_abs <= args.tolerance else "FAIL"
        if status == "FAIL" and failing_layer is None:
            failing_layer = layer
        print(
            f"L{layer:02d} {status} | max_abs={max_abs:.6g} | "
            f"mean_abs={mean_abs:.6g} | rel={rel:.6g} | cosine={cosine:.9f}"
        )

    max_abs, mean_abs, rel, cosine = _stats(
        ref_states[-1][:, -1, :], router_final_norm
    )
    print("\n=== FINAL HIDDEN (NOTE: reference raw vs router normalized) ===")
    print(
        f"max_abs={max_abs:.6g} | mean_abs={mean_abs:.6g} | "
        f"rel={rel:.6g} | cosine={cosine:.9f}"
    )

    max_abs, mean_abs, rel, cosine = _stats(ref_logits, router_logits)
    print("\n=== LOGITS ===")
    print(
        f"max_abs={max_abs:.6g} | mean_abs={mean_abs:.6g} | "
        f"rel={rel:.6g} | cosine={cosine:.9f}"
    )

    ref_top = _top_ids(ref_logits, args.top_k)
    router_top = _top_ids(router_logits, args.top_k)
    overlap = len(set(ref_top) & set(router_top))
    print(f"official_top_{args.top_k}={_format_ids(ref_top)}")
    print(f"router_top_{args.top_k}={_format_ids(router_top)}")
    print(f"top_{args.top_k}_overlap={overlap}/{args.top_k}")

    print("\n=== RESULT ===")
    if failing_layer is not None:
        print(f"status=FAIL first_failing_layer={failing_layer}")
    else:
        print("status=PASS")

    del ref_input, ref_states, ref_logits
    del router_states, router_final_norm, router_logits, model
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()

    if failing_layer is not None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
