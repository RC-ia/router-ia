from __future__ import annotations

"""Memory policy v2 for the stateful Qwen3.6 router.

Goals:
- keep only genuinely small, layer-local control tensors in the resident VRAM pool;
- put large per-layer projection weights in the rotating VRAM stream;
- keep the embedding matrix on host memory and fetch only one row per token;
- keep lm_head on host memory and execute it in VRAM chunks instead of caching
  the entire vocabulary projection on the GPU.

This module patches the canonical chat path without changing model math beyond
using the same FP16 autocast behavior for the chunked lm_head projection.
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from . import qwen36_cached_loop as cached
from . import qwen36_40layer_loop as base
from . import qwen36_chat_batch as chat
from . import qwen36_mini_chat as mini_chat

EMBEDDING_NAME = "model.language_model.embed_tokens.weight"
LM_HEAD_SUFFIX = "lm_head.weight"
LM_HEAD_CHUNK_ROWS = 16384

_ORIGINAL_PROJECTION = cached._cached_load_projection
_ORIGINAL_EMBEDDING_ROW = base.load_embedding_row
_ORIGINAL_RUN_FORWARD = chat.run_forward_token
_ORIGINAL_LOAD_LM_HEAD = chat.load_lm_head


def _is_layer_projection(prefix: str) -> bool:
    if ".layers." not in prefix:
        return False
    return prefix.endswith((
        "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_a",
        "linear_attn.in_proj_b",
        "linear_attn.in_proj_z",
        "linear_attn.out_proj",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.shared_expert.gate_proj",
        "mlp.shared_expert.up_proj",
        "mlp.shared_expert.down_proj",
    ))


def _projection(root: Path, prefix: str, device: str):
    if device == "cuda" and _is_layer_projection(prefix):
        return cached._store(root).stream_projection(prefix)
    return _ORIGINAL_PROJECTION(root, prefix, device)


def _embedding_row(root: Path, token_id: int) -> torch.Tensor:
    """Read one embedding row from the safetensors slice instead of materializing the table."""
    store = cached._store(root)
    name = EMBEDDING_NAME
    shard_name = store.weight_map.get(name)
    if shard_name is not None:
        handle = store._handle(store.root / shard_name)
        try:
            view = handle.get_slice(name)
            shape = tuple(view.get_shape())
            if len(shape) != 2 or shape[1] != base.HIDDEN:
                raise ValueError(f"Unexpected embedding shape: {shape}")
            if not 0 <= int(token_id) < shape[0]:
                raise ValueError(f"token_id {token_id} outside vocabulary")
            return view[int(token_id)].float()
        except AttributeError:
            pass

    # Conservative fallback for old safetensors versions without get_slice().
    return _ORIGINAL_EMBEDDING_ROW(root, token_id)


def _find_tensor_name(root: Path, suffixes: tuple[str, ...]) -> str:
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise SystemExit(f"Missing index: {index_path}")
    import json

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    names = list(payload.get("weight_map", {}).keys())
    for suffix in suffixes:
        matches = [name for name in names if name.endswith(suffix)]
        if len(matches) == 1:
            return matches[0]
        if matches:
            for name in matches:
                if "language_model" in name:
                    return name
            return matches[0]
    raise KeyError(f"Could not find tensor with suffixes={suffixes}")


def _load_lm_head(root: Path) -> tuple[str, torch.Tensor, str]:
    """Keep lm_head on CPU in its source dtype; GPU materialization is chunked."""
    name = _find_tensor_name(root, (LM_HEAD_SUFFIX,))
    weight = cached._cached_load_tensor(root, name, device="cpu")
    return name, weight, "cpu-chunked"


def _dequant_fp8_chunk(weight: torch.Tensor, scale: torch.Tensor, start: int, end: int) -> torch.Tensor:
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError(f"Unexpected FP8 lm_head tensors: {tuple(weight.shape)} / {tuple(scale.shape)}")
    block = 128
    scale_start = start // block
    scale_end = (end + block - 1) // block
    row_scale = scale[scale_start:scale_end]
    expanded = row_scale.float().repeat_interleave(block, dim=0)
    expanded = expanded[:, : weight.shape[1]]
    expanded = expanded[: end - start, :]
    return weight[start:end].float() * expanded


def _lm_head_logits(
    root: Path,
    x: torch.Tensor,
    lm_head: torch.Tensor,
    lm_head_name: str,
    device: str,
) -> torch.Tensor:
    if device != "cuda":
        return F.linear(x, lm_head.float())

    store = cached._store(root)
    source = store.load(lm_head_name, device="cpu")
    vocab, hidden = map(int, source.shape)
    if hidden != base.HIDDEN:
        raise ValueError(f"Unexpected lm_head shape: {tuple(source.shape)}")

    scale = None
    if source.dtype == torch.float8_e4m3fn:
        scale = store.load(lm_head_name.replace(".weight", ".weight_scale_inv"), device="cpu")

    output = torch.empty((x.shape[0], vocab), device="cuda", dtype=torch.float32)
    x_compute = x.to(dtype=torch.float16)
    for start in range(0, vocab, LM_HEAD_CHUNK_ROWS):
        end = min(start + LM_HEAD_CHUNK_ROWS, vocab)
        if scale is None:
            chunk = source[start:end].to(device="cuda", dtype=torch.float16, non_blocking=True)
        else:
            chunk_cpu = _dequant_fp8_chunk(source, scale, start, end)
            chunk = chunk_cpu.to(device="cuda", dtype=torch.float16, non_blocking=True)
            del chunk_cpu
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            chunk_logits = F.linear(x_compute, chunk)
        output[:, start:end].copy_(chunk_logits.float())
        del chunk, chunk_logits
    del source
    if scale is not None:
        del scale
    return output


def _run_forward_token(
    root: Path,
    token_id: int,
    final_norm: torch.Tensor,
    lm_head: torch.Tensor,
    final_norm_name: str,
    lm_head_name: str,
    device: str,
    advance_state: bool = True,
):
    from time import perf_counter
    from . import qwen36_attention_cache as attention_cache

    start = perf_counter()
    x = base.load_embedding_row(root, token_id).reshape(1, base.HIDDEN).to(device).float()
    for layer in range(base.DEFAULT_LAYERS):
        residual = attention_cache.step_attention(root, layer, x, device)
        x, *_ = chat.batched_moe_step(root, layer, residual, top_k=8, device=device)
        del residual

    if device == "cuda":
        final_norm_runtime = cached.cached_runtime_tensor(root, final_norm_name, device, dtype=torch.float32)
    else:
        final_norm_runtime = final_norm

    x = base.rmsnorm(x, final_norm_runtime)
    logits = _lm_head_logits(root, x, lm_head, lm_head_name, device)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = perf_counter() - start
    peak_logit = float(torch.max(logits.float()).item())
    if advance_state:
        state = attention_cache.active(root, device)
        state.tokens_seen += 1
    return logits, elapsed, peak_logit


cached._cached_load_projection = _projection
base.load_embedding_row = _embedding_row
chat.load_lm_head = _load_lm_head
chat.run_forward_token = _run_forward_token

print(
    f"memory_policy_v2=enabled|layer-projections=stream|embedding=row-slice|"
    f"lm_head=cpu-chunked|lm_head_chunk_rows={LM_HEAD_CHUNK_ROWS}|"
    "global-small=resident"
)
