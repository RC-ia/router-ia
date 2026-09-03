from __future__ import annotations

"""Two-phase Qwen3.6 fidelity test.

Phase ``reference`` runs the official Transformers model alone and stores only
small CPU snapshots. Phase ``router`` runs our implementation alone and
compares against that snapshot, avoiding simultaneous VRAM pressure.
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


def _encode_prompt(tokenizer, prompt: str, chat_template: bool) -> list[int]:
    if chat_template:
        if not hasattr(tokenizer, "apply_chat_template"):
            raise SystemExit("Tokenizer has no chat template support")
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(encoded, dict):
            encoded = encoded["input_ids"]
        return [int(x) for x in encoded.reshape(-1).tolist()]
    return [int(x) for x in tokenizer.encode(prompt, add_special_tokens=False)]


def _reference_snapshot(model: torch.nn.Module, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
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
            f"Expected embeddings + {base.DEFAULT_LAYERS} hidden states, got {len(hidden_states)}"
        )

    layer_hidden = torch.stack(
        [hidden[:, -1, :].detach().float().cpu() for hidden in hidden_states[1 : base.DEFAULT_LAYERS + 1]],
        dim=0,
    )
    raw_final = hidden_states[-1][:, -1, :].detach().float().cpu()
    logits = outputs.logits[:, -1, :].detach().float().cpu()
    return {
        "input_ids": input_ids.detach().cpu(),
        "layer_hidden": layer_hidden,
        "raw_final": raw_final,
        "logits": logits,
    }


def _load_router_output_weights(root: Path, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    norm_name = find_tensor_name(
        root,
        ("language_model.norm.weight", "model.norm.weight", ".norm.weight"),
    )
    lm_name = find_tensor_name(root, ("lm_head.weight",))
    norm = cached._cached_load_tensor(root, norm_name, device=device).float()
    lm_head = cached._cached_load_tensor(root, lm_name, device=device).float()
    return norm, lm_head


def _router_forward(root: Path, token_ids: list[int], device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state = attention.state_for(root, device)
    state.reset()
    attention.activate(root, state)
    final_x: torch.Tensor | None = None
    last_layer_states: list[torch.Tensor] = []

    try:
        with torch.inference_mode():
            for token_id in token_ids:
                x = base.load_embedding_row(root, int(token_id)).reshape(1, base.HIDDEN).to(device).float()
                current_layer_states: list[torch.Tensor] = []
                for layer in range(base.DEFAULT_LAYERS):
                    residual = attention.step_attention(root, layer, x, device)
                    x, *_ = chat.batched_moe_step(root, layer, residual, top_k=8, device=device)
                    current_layer_states.append(x.detach().float().cpu())
                    del residual
                last_layer_states = current_layer_states
                final_x = x.detach().float()
                state.tokens_seen += 1
                del x, current_layer_states

        if final_x is None or len(last_layer_states) != base.DEFAULT_LAYERS:
            raise RuntimeError("Router forward did not produce the expected states")

        norm, lm_head = _load_router_output_weights(root, device)
        final_norm = rmsnorm(final_x, norm).float()
        logits = F.linear(final_norm, lm_head).float()
        return torch.stack(last_layer_states, dim=0), final_norm.detach().cpu(), logits.detach().cpu()
    finally:
        attention.deactivate(root)


def _stats(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[float, float, float, float]:
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
    return max_abs, mean_abs, rel, cosine


def _top_ids(logits: torch.Tensor, k: int) -> list[int]:
    return [int(x) for x in torch.topk(logits.reshape(-1), k).indices.tolist()]


def _compare(snapshot: dict[str, torch.Tensor], router_states: torch.Tensor, router_logits: torch.Tensor, top_k: int, tolerance: float) -> int:
    reference_states = snapshot["layer_hidden"]
    if tuple(reference_states.shape) != (base.DEFAULT_LAYERS, 1, base.HIDDEN):
        raise RuntimeError(f"Unexpected snapshot layer shape: {tuple(reference_states.shape)}")

    failing_layer: int | None = None
    print("\n=== LAYER DIFFS ===")
    for layer in range(base.DEFAULT_LAYERS):
        stats = _stats(reference_states[layer], router_states[layer])
        status = "PASS" if stats[0] <= tolerance else "FAIL"
        if status == "FAIL" and failing_layer is None:
            failing_layer = layer
        print(f"L{layer:02d} {status} | max_abs={stats[0]:.6g} | mean_abs={stats[1]:.6g} | rel={stats[2]:.6g} | cosine={stats[3]:.9f}")

    print("\n=== LOGITS ===")
    stats = _stats(snapshot["logits"], router_logits)
    print(f"max_abs={stats[0]:.6g} | mean_abs={stats[1]:.6g} | rel={stats[2]:.6g} | cosine={stats[3]:.9f}")

    ref_top = _top_ids(snapshot["logits"], top_k)
    router_top = _top_ids(router_logits, top_k)
    overlap = len(set(ref_top) & set(router_top))
    print(f"official_top_{top_k}={', '.join(map(str, ref_top))}")
    print(f"router_top_{top_k}={', '.join(map(str, router_top))}")
    print(f"top_{top_k}_overlap={overlap}/{top_k}")
    print("\n=== RESULT ===")
    if failing_layer is None:
        print("status=PASS")
        return 0
    print(f"status=FAIL first_failing_layer={failing_layer}")
    return 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-phase Qwen3.6 official-vs-router fidelity test")
    parser.add_argument("root", type=Path)
    parser.add_argument("--mode", choices=("reference", "router"), required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--snapshot", type=Path, default=Path("qwen36-fidelity.pt"))
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
    snapshot_path = args.snapshot.resolve()
    tokenizer = load_tokenizer(root)
    token_ids = _encode_prompt(tokenizer, args.prompt, args.chat_template)
    if not token_ids:
        raise SystemExit("Prompt produced no tokens")
    if len(token_ids) > args.max_prompt_tokens:
        print(f"warning=prompt has {len(token_ids)} tokens; using last {args.max_prompt_tokens} tokens")
        token_ids = token_ids[-args.max_prompt_tokens :]

    print("op=fidelity")
    print(f"mode={args.mode}")
    print(f"prompt={args.prompt!r}")
    print(f"chat_template={args.chat_template}")
    print(f"prompt_tokens={len(token_ids)}")
    print(f"token_ids={', '.join(map(str, token_ids))}")
    print(f"device={args.device}")
    print(f"snapshot={snapshot_path}")

    if args.mode == "reference":
        print("\n[REFERENCE] Loading official Transformers model alone...")
        model = load_reference_model(root, device=args.device)
        ref_device = _model_input_device(model, args.device)
        ref_input = torch.tensor([token_ids], device=ref_device, dtype=torch.long)
        print("[REFERENCE] Running forward...")
        snapshot = _reference_snapshot(model, ref_input)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(snapshot, snapshot_path)
        print(f"[REFERENCE] Snapshot saved: {snapshot_path}")
        print(f"layer_hidden_shape={tuple(snapshot['layer_hidden'].shape)} | logits_shape={tuple(snapshot['logits'].shape)}")
        del ref_input, snapshot, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    if not snapshot_path.is_file():
        raise SystemExit(f"Snapshot not found: {snapshot_path}. Run --mode reference first.")
    snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=True)
    stored_ids = [int(x) for x in snapshot["input_ids"].reshape(-1).tolist()]
    if stored_ids != token_ids:
        raise SystemExit(f"Prompt/token mismatch: snapshot={stored_ids} current={token_ids}")

    print("\n[ROUTER] Running router alone...")
    router_states, _router_final_norm, router_logits = _router_forward(root, token_ids, args.device)
    exit_code = _compare(snapshot, router_states, router_logits, args.top_k, args.tolerance)

    del snapshot, router_states, _router_final_norm, router_logits
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
