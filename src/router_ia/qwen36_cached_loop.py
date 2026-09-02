from __future__ import annotations

"""Qwen3.6 loop with optimized Safetensors I/O and FP8 dequantization.

This wrapper keeps the reference math in qwen36_40layer_loop.py unchanged while
replacing only the tensor-loading backend and FP8 dequantizer. Safetensors
shards are opened lazily and kept open for the lifetime of the run. FP8 scales
are broadcast over quantization blocks without materializing a full-size scale
matrix.
"""

import atexit
import json
from contextlib import ExitStack
from pathlib import Path

from safetensors import safe_open

from . import qwen36_40layer_loop as base
from .qwen36_dequant import dequantize_fp8_blockwise


class _ShardStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.stack = ExitStack()
        self.weight_map: dict[str, str] = {}
        self.handles: dict[Path, object] = {}
        self.handle_opens = 0
        self.handle_hits = 0

        index_path = self.root / "model.safetensors.index.json"
        if index_path.is_file():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            self.weight_map = dict(payload.get("weight_map", {}))

    def _handle(self, shard: Path):
        handle = self.handles.get(shard)
        if handle is not None:
            self.handle_hits += 1
            return handle

        handle = self.stack.enter_context(
            safe_open(str(shard), framework="pt", device="cpu")
        )
        self.handles[shard] = handle
        self.handle_opens += 1
        return handle

    def load(self, name: str, device: str):
        shard_name = self.weight_map.get(name)
        if shard_name:
            shards = [self.root / shard_name]
        else:
            shards = sorted(self.root.glob("*.safetensors"))

        for shard in shards:
            if not shard.is_file():
                continue
            handle = self._handle(shard)
            if name in handle.keys():
                return handle.get_tensor(name).to(device=device)

        raise KeyError(f"Tensor not found: {name}")

    def close(self) -> None:
        self.stack.close()
        self.handles.clear()


_stores: dict[Path, _ShardStore] = {}


def _store(root: Path) -> _ShardStore:
    key = root.resolve()
    store = _stores.get(key)
    if store is None:
        store = _ShardStore(key)
        _stores[key] = store
    return store


def _cached_load_tensor(root: Path, name: str, device: str = "cpu"):
    return _store(root).load(name, device)


# qwen36_40layer_loop resolves both names through its module globals. Patching
# them here preserves the exact reference computation and changes only the
# I/O/dequantization implementation.
base.load_tensor = _cached_load_tensor
base.dequantize_fp8_blockwise = dequantize_fp8_blockwise


@atexit.register
def _close_stores() -> None:
    for store in _stores.values():
        store.close()


def main() -> None:
    base.main()

    for root, store in _stores.items():
        print(
            f"cached reader: root={root} | "
            f"shards opened={store.handle_opens} | "
            f"cached handle hits={store.handle_hits}"
        )


if __name__ == "__main__":
    main()
