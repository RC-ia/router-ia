from __future__ import annotations

"""Avoid Windows safetensors mmap commit inflation in the router.

`safetensors.safe_open()` defaults to an mmap backend. On Windows, large
model shards can contribute heavily to system commit. The router already has
its own RAM/VRAM/SSD hierarchy, so read shards on demand with `pread` instead
of mapping giant files.

Use `pread` only on Windows. Linux keeps the normal mmap path.
"""

import os
import sys

from safetensors import safe_open

from . import qwen36_cached_loop as cached

ENABLED = sys.platform.startswith("win") and os.getenv(
    "QWEN36_WINDOWS_MMAP_GUARD", "1"
).strip().lower() not in {"0", "false", "no", "off"}

_ORIGINAL_HANDLE = cached._ShardStore._handle


def _handle_pread(self, shard):
    if not ENABLED:
        return _ORIGINAL_HANDLE(self, shard)

    handle = self.handles.get(shard)
    if handle is not None:
        self.handle_hits += 1
        return handle

    handle = self.stack.enter_context(
        safe_open(str(shard), framework="pt", device="cpu", backend="pread")
    )
    self.handles[shard] = handle
    self.handle_opens += 1
    return handle


if ENABLED:
    cached._ShardStore._handle = _handle_pread
    print(
        "windows_mmap_guard=enabled|safetensors_backend=pread|"
        "reason=avoid-mmap-commit-inflation"
    )
else:
    print("windows_mmap_guard=inactive|backend=mmap-default")
