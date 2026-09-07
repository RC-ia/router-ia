from __future__ import annotations

"""Indexed generation entry point.

Loads a compact expert-location map into RAM once, then patches the existing
Safetensors store so routed expert requests go straight to their shard
instead of scanning all shards.
"""

from pathlib import Path

from . import qwen36_cached_loop as cached
from .qwen36_compact_index import CompactExpertIndex
from .qwen36_chat_batch import main as _chat_main


_ORIGINAL_LOAD_SSD = cached._ShardStore._load_ssd


def _indexed_load_ssd(self, name: str):
    expert_index = getattr(self, "compact_expert_index", None)
    if expert_index is not None:
        shard = expert_index.lookup(name)
        if shard is not None and shard.is_file():
            handle = self._handle(shard)
            if name in handle.keys():
                return handle.get_tensor(name)
    return _ORIGINAL_LOAD_SSD(self, name)


cached._ShardStore._load_ssd = _indexed_load_ssd


_ORIGINAL_STORE = cached._store


def _indexed_store(root: Path):
    store = _ORIGINAL_STORE(root)
    if not hasattr(store, "compact_expert_index"):
        try:
            store.compact_expert_index = CompactExpertIndex(root)
            stats = store.compact_expert_index.stats()
            print(
                "expert index: "
                f"shards={stats['shards']} | "
                f"lookups=0 | "
                f"file={store.compact_expert_index.path.name}"
            )
        except FileNotFoundError:
            raise RuntimeError(
                "Compact expert index missing. Run:\n"
                "  python -m router_ia.qwen36_build_expert_index <model_dir>\n"
                "before using qwen36_chat_indexed."
            )
    return store


cached._stores.clear()
cached._store = _indexed_store


if __name__ == "__main__":
    _chat_main()
