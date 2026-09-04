from __future__ import annotations

"""Generation-scoped expert heat for continuous Qwen3.6 generation.

Each prompt starts a fresh generation epoch. Expert usage from the previous
prompt is invalidated and its dedicated FP8/Q4 expert storage is released.
Within the active generation, repeated expert use raises the adaptive score,
so the current token loop progressively keeps the hottest experts in VRAM.
"""

from collections import OrderedDict
from pathlib import Path
from threading import Lock

from . import qwen36_adaptive_experts as adaptive
from . import qwen36_chat_batch as chat
from . import qwen36_official_optimizations as official

_GENERATION_LOCK = Lock()
_GENERATION_IDS: dict[Path, int] = {}
_ORIGINAL_GENERATE_RESPONSE = chat.generate_response
_ORIGINAL_PRINT_CACHE = chat.print_cache


def _generation_id(root: Path) -> int:
    with _GENERATION_LOCK:
        return _GENERATION_IDS.get(root.resolve(), 0)


def reset_generation(root: Path) -> int:
    """Start a fresh prompt epoch and release all generation-local expert cache."""
    key = root.resolve()
    expert = official._EXPERT_CACHES.get(key)
    if expert is not None:
        with expert.lock:
            expert.fp8_entries = {layer: OrderedDict() for layer in range(expert.layers)}
            expert.q4_entries = {layer: OrderedDict() for layer in range(expert.layers)}
            expert.entry_bytes.clear()
            expert.q4_ram_bytes.clear()
            expert.bytes_used = 0
            expert.q4_bytes_used = 0

    try:
        store = official.cached._store(root)
        store.clear_stream()
    except Exception:
        pass

    policy = adaptive._policy(root)
    with policy.lock:
        policy.tick = 0
        policy.usage.clear()
        policy.prefetch_requests = 0
        policy.prefetch_selected = 0
        policy.prefetch_skipped = 0
        policy.cold_misses = 0
        policy.q4_hits = 0
        policy.fp8_hits = 0
        policy.promotions = 0
        policy.demotions = 0

    with _GENERATION_LOCK:
        generation = _GENERATION_IDS.get(key, 0) + 1
        _GENERATION_IDS[key] = generation
    return generation


def _generation_generate_response(*args, **kwargs):
    root = Path(args[0] if args else kwargs["root"])
    generation = reset_generation(root)
    print(f"  generation heat: epoch={generation} | previous expert boost cleared")
    return _ORIGINAL_GENERATE_RESPONSE(*args, **kwargs)


def _generation_print_cache(root: Path, label: str) -> None:
    _ORIGINAL_PRINT_CACHE(root, label)
    print(f"  generation heat: epoch={_generation_id(root)} | boost=prompt-scoped")


chat.generate_response = _generation_generate_response
chat.print_cache = _generation_print_cache
