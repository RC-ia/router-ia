from __future__ import annotations

"""Detailed timing profiler for the Qwen3.6 RAM -> VRAM -> GPU path."""

import argparse
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import torch

from . import qwen36_cached_loop as cached
from . import qwen36_chat_batch as chat

T = defaultdict(float)
N = defaultdict(int)
B = defaultdict(int)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def stream_profile(self, prefix: str):
    if self.target_device != "cuda":
        raise RuntimeError("stream_projection requires CUDA")
    key = prefix + ".__stream__"
    t = perf_counter(); x = self.vram_cache.get_stream(key); sync(); T["stream_lookup"] += perf_counter()-t; N["stream_lookup"] += 1
    if x is not None:
        N["stream_hits"] += 1
        return x
    N["stream_misses"] += 1

    t = perf_counter(); w = self.load(prefix + ".weight", "cpu"); T["ram_weight"] += perf_counter()-t; N["ram_weight"] += 1
    B["fp8"] += w.numel() * w.element_size()
    if w.dtype == torch.float8_e4m3fn:
        t = perf_counter(); s = self.load(prefix + ".weight_scale_inv", "cpu"); T["ram_scale"] += perf_counter()-t; N["ram_scale"] += 1
        B["scale"] += s.numel() * s.element_size()
        sync(); t = perf_counter(); gw = w.to("cuda"); sync(); T["h2d_weight"] += perf_counter()-t; N["h2d_weight"] += 1
        sync(); t = perf_counter(); gs = s.to("cuda"); sync(); T["h2d_scale"] += perf_counter()-t; N["h2d_scale"] += 1
        del w, s
        sync(); t = perf_counter(); d = cached.dequantize_fp8_blockwise(gw, gs); sync(); T["dequant"] += perf_counter()-t; N["dequant"] += 1
        del gw, gs
        sync(); t = perf_counter(); out = d.to(dtype=torch.float16); sync(); T["fp16_cast"] += perf_counter()-t; N["fp16_cast"] += 1
        del d
    else:
        sync(); t = perf_counter(); out = w.to("cuda", dtype=torch.float16); sync(); T["h2d_cast"] += perf_counter()-t; N["h2d_cast"] += 1
        del w
    B["fp16"] += out.numel() * out.element_size()
    t = perf_counter(); ok = self.vram_cache.put_stream(key, out); T["cache_put"] += perf_counter()-t; N["cache_put"] += 1
    if not ok: N["cache_rejected"] += 1
    return out


def timed(name, fn):
    def wrap(*a, **k):
        sync(); t = perf_counter(); r = fn(*a, **k); sync(); T[name] += perf_counter()-t; N[name] += 1; return r
    return wrap


cached._ShardStore.stream_projection = stream_profile
chat.batched_moe_step = timed("moe", chat.batched_moe_step)


def report():
    total = sum(T.values())
    print("\n===== PIPELINE PROFILE =====")
    names = ["ram_weight","ram_scale","h2d_weight","h2d_scale","dequant","fp16_cast","h2d_cast","cache_put","stream_lookup","moe"]
    for x in names:
        sec = T[x]; share = sec/total*100 if total else 0
        print(f"{x:16s} {sec:9.3f}s  calls={N[x]:5d}  share={share:6.2f}%")
    ram = T["ram_weight"] + T["ram_scale"]
    h2d = T["h2d_weight"] + T["h2d_scale"] + T["h2d_cast"]
    prep = T["dequant"] + T["fp16_cast"]
    print("-----------------------------------------------")
    print(f"RAM access       {ram:9.3f}s")
    print(f"RAM -> VRAM      {h2d:9.3f}s")
    print(f"GPU preparation  {prep:9.3f}s")
    print(f"MoE compute      {T['moe']:9.3f}s")
    print("bytes transferred:")
    print(f"  FP8 weights = {B['fp8']/1024**2:.1f} MiB")
    print(f"  scales      = {B['scale']/1024**2:.1f} MiB")
    print(f"  FP16 output = {B['fp16']/1024**2:.1f} MiB")
    print(f"stream hits={N['stream_hits']} misses={N['stream_misses']} rejected={N['cache_rejected']}")
    print(f"profile sum = {total:.3f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--prompt", action="append", required=True)
    p.add_argument("--max-new-tokens", type=int, default=4)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.0)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    import sys
    sys.argv = ["qwen36_chat_batch", str(args.root)]
    for prompt in args.prompt:
        sys.argv += ["--prompt", prompt]
    sys.argv += ["--max-new-tokens", str(args.max_new_tokens), "--top-k", str(args.top_k), "--temperature", str(args.temperature), "--device", "cuda"]
    chat.main()
    report()


if __name__ == "__main__":
    main()
