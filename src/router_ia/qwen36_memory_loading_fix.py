import os, sys
from pathlib import Path
from safetensors import safe_open
from . import qwen36_40layer_loop as base
from . import qwen36_cached_loop as cached
from . import qwen36_chat_batch as chat

TRACE = os.getenv('QWEN36_MEMORY_LOAD_TRACE','0').lower() in {'1','true','yes','on'}
WINDOWS_PREAD = sys.platform.startswith('win') and os.getenv('QWEN36_WINDOWS_MMAP_GUARD','1').lower() not in {'0','false','no','off'}
ORIG_HANDLE = cached._ShardStore._handle

def handle(self, shard):
    if not WINDOWS_PREAD: return ORIG_HANDLE(self, shard)
    h = self.handles.get(shard)
    if h is not None:
        self.handle_hits += 1
        return h
    h = self.stack.enter_context(safe_open(str(shard), framework='pt', device='cpu', backend='pread'))
    self.handles[shard] = h
    self.handle_opens += 1
    return h

def has_name(root: Path, name: str) -> bool:
    store = cached._store(root.resolve())
    if store.weight_map: return name in store.weight_map
    for shard in sorted(root.glob('*.safetensors')):
        with safe_open(str(shard), framework='pt', device='cpu', backend='pread' if WINDOWS_PREAD else 'mmap') as h:
            if name in h.keys(): return True
    return False

TYPES = {}
def attention_type(root: Path, layer: int) -> str:
    key = root.resolve(); detected = TYPES.get(key)
    if detected is None:
        out=[]
        for i in range(base.DEFAULT_LAYERS):
            p=base.layer_prefix(i)
            out.append('linear_attention' if has_name(key,p+'linear_attn.in_proj_qkv.weight') else 'full_attention')
        detected=tuple(out); TYPES[key]=detected
        if TRACE: print(f'memory_load_trace=attention-layout|mode=metadata-only|done|linear={detected.count("linear_attention")}|full={detected.count("full_attention")}')
    return detected[int(layer)]

def no_prefetch(root, layer_prefix, expert_ids):
    if TRACE and expert_ids: print(f'memory_load_trace=legacy-fp8-prefetch|disabled|experts={len(set(expert_ids))}')

base.attention_type=attention_type
chat._warm_expert_raw_cache=no_prefetch
if WINDOWS_PREAD: cached._ShardStore._handle=handle
print(f'memory_loading_fix=enabled|windows_backend={"pread" if WINDOWS_PREAD else "mmap"}|attention=metadata-only|legacy_fp8_prefetch=disabled')
