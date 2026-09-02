from __future__ import annotations

"""Detailed, crash-resistant Qwen3.6 Safetensors loading tracer.

This tool deliberately separates file/index discovery, CPU tensor materialization,
CPU->CUDA transfer, and optional FP8 dequantization. Every step is written to a
log file and flushed/fsynced before potentially dangerous operations, so the
last completed/started operation remains visible even after a hard OS reset.
"""

import argparse
import ctypes
import gc
import json
import os
import platform
import sys
import traceback
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch
from safetensors import safe_open

try:
    from .qwen36_op_probe import BLOCK, dequantize_fp8_blockwise
except ImportError:
    BLOCK = 128
    dequantize_fp8_blockwise = None


class CrashSafeLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = self.path.open("a", encoding="utf-8", buffering=1)
        self.seq = 0

    def close(self) -> None:
        try:
            self.fp.flush()
            os.fsync(self.fp.fileno())
        except OSError:
            pass
        self.fp.close()

    def log(self, message: str, *, sync: bool = True) -> None:
        self.seq += 1
        line = f"[{self.seq:07d}] {datetime.now().astimezone().isoformat(timespec='milliseconds')} | {message}"
        print(line, flush=True)
        self.fp.write(line + "\n")
        self.fp.flush()
        if sync:
            try:
                os.fsync(self.fp.fileno())
            except OSError:
                pass


def available_ram() -> int | None:
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys)
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):
        return None


def mib(n: int | None) -> str:
    return "n/a" if n is None else f"{n / 1024**2:.1f} MiB"


def tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def cuda_snapshot() -> str:
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    try:
        free, total = torch.cuda.mem_get_info()
        return (
            f"VRAM free={free / 1024**2:.1f} MiB total={total / 1024**2:.1f} MiB | "
            f"alloc={torch.cuda.memory_allocated() / 1024**2:.1f} MiB "
            f"reserved={torch.cuda.memory_reserved() / 1024**2:.1f} MiB"
        )
    except Exception as exc:
        return f"CUDA memory query failed: {type(exc).__name__}: {exc}"


def memory_line() -> str:
    return f"RAM free={mib(available_ram())} | {cuda_snapshot()}"


def layer_of(name: str) -> str:
    marker = "model.language_model.layers."
    if not name.startswith(marker):
        return "global"
    rest = name[len(marker):]
    return rest.split(".", 1)[0]


def shape_of_slice(handle, name: str) -> tuple[int, ...] | None:
    try:
        return tuple(int(x) for x in handle.get_slice(name).get_shape())
    except Exception:
        return None


def parse_layers(spec: str, max_layer: int) -> tuple[int, int]:
    if ":" in spec:
        a, b = spec.split(":", 1)
        start, end = int(a), int(b)
    else:
        start = end = int(spec)
    if start < 0 or end < start or end > max_layer:
        raise ValueError(f"invalid layer range {spec}; valid range is 0:{max_layer}")
    return start, end


def load_one_tensor_cpu(
    logger: CrashSafeLogger,
    shard: Path,
    name: str,
    expected_shape: tuple[int, ...] | None,
) -> torch.Tensor:
    logger.log(f"LOAD BEGIN | layer={layer_of(name)} | tensor={name} | shard={shard.name} | {memory_line()}")
    logger.log(f"OPEN BEGIN | shard={shard.name}")
    t0 = perf_counter()
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        logger.log(f"OPEN OK | shard={shard.name} | open_ms={(perf_counter()-t0)*1000:.2f} | keys={len(handle.keys())} | {memory_line()}")
        if name not in handle.keys():
            raise KeyError(f"tensor not present in shard: {name}")
        metadata_shape = shape_of_slice(handle, name)
        logger.log(f"METADATA OK | tensor={name} | shape={metadata_shape} | expected={expected_shape}")
        logger.log(f"GET_TENSOR ABOUT_TO_START | tensor={name} | {memory_line()}")
        t0 = perf_counter()
        tensor = handle.get_tensor(name)
        logger.log(
            f"GET_TENSOR OK | tensor={name} | shape={tuple(tensor.shape)} | dtype={tensor.dtype} | "
            f"bytes={tensor_nbytes(tensor)} | elapsed_ms={(perf_counter()-t0)*1000:.2f} | {memory_line()}"
        )
    logger.log(f"SHARD CLOSE OK | shard={shard.name} | tensor={name} | {memory_line()}")
    if expected_shape is not None and tuple(tensor.shape) != expected_shape:
        raise ValueError(f"shape mismatch after load: {name}: {tuple(tensor.shape)} != {expected_shape}")
    if tensor.numel() == 0:
        raise ValueError(f"empty tensor after load: {name}")
    logger.log(f"CPU TENSOR VALIDATED | tensor={name} | device={tensor.device} | {memory_line()}")
    return tensor


def run(args: argparse.Namespace) -> int:
    root = Path(args.model).resolve()
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing {index_path}")

    log_path = Path(args.log).resolve() if args.log else Path.cwd() / f"qwen36-load-trace-{datetime.now():%Y%m%d-%H%M%S}.log"
    logger = CrashSafeLogger(log_path)
    try:
        logger.log("=" * 100)
        logger.log("QWEN3.6 LOAD TRACE START")
        logger.log(f"PID={os.getpid()} | Python={sys.version.split()[0]} | OS={platform.platform()}")
        logger.log(f"Torch={torch.__version__} | CUDA available={torch.cuda.is_available()} | device={args.device}")
        if torch.cuda.is_available():
            logger.log(f"CUDA device={torch.cuda.get_device_name(0)} | capability={torch.cuda.get_device_capability(0)}")
        logger.log(f"MODEL ROOT={root}")
        logger.log(f"INDEX={index_path}")
        logger.log(f"LOG FILE={log_path}")
        logger.log(f"INITIAL MEMORY | {memory_line()}")

        logger.log("INDEX READ BEGIN")
        t0 = perf_counter()
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("invalid/empty weight_map")
        logger.log(f"INDEX READ OK | tensors_in_index={len(weight_map)} | elapsed_ms={(perf_counter()-t0)*1000:.2f}")

        layers = []
        for name in weight_map:
            if name.startswith("model.language_model.layers."):
                try:
                    layers.append(int(layer_of(name)))
                except ValueError:
                    pass
        max_layer = max(layers) if layers else 0
        start_layer, end_layer = parse_layers(args.layers, max_layer)
        logger.log(f"LAYER RANGE | requested={start_layer}:{end_layer} | model_max_layer={max_layer}")

        selected = []
        for name, shard_name in weight_map.items():
            layer = layer_of(name)
            if layer == "global":
                if args.include_global:
                    selected.append((name, shard_name))
            else:
                li = int(layer)
                if start_layer <= li <= end_layer:
                    selected.append((name, shard_name))
        if args.limit is not None:
            selected = selected[:args.limit]
        logger.log(f"LOAD PLAN | selected_tensors={len(selected)} | stage={args.stage} | limit={args.limit}")

        shard_cache: dict[str, list[str]] = {}
        expected_suffix = {
            "input_layernorm.weight": (2048,),
            "post_attention_layernorm.weight": (2048,),
            "mlp.gate.weight": (256, 2048),
            "mlp.shared_expert_gate.weight": (1, 2048),
            "linear_attn.in_proj_qkv.weight": (8192, 2048),
            "linear_attn.conv1d.weight": (8192, 1, 4),
            "self_attn.q_proj.weight": (8192, 2048),
            "self_attn.k_proj.weight": (512, 2048),
            "self_attn.v_proj.weight": (512, 2048),
            "self_attn.o_proj.weight": (2048, 4096),
        }

        completed = 0
        for seq, (name, shard_name) in enumerate(selected, 1):
            shard = root / shard_name
            if not shard.is_file():
                raise FileNotFoundError(f"missing shard: {shard}")
            suffix = name.split(f"model.language_model.layers.{layer_of(name)}.", 1)[-1] if layer_of(name) != "global" else name
            expected = expected_suffix.get(suffix)
            logger.log(f"STEP {seq}/{len(selected)} BEGIN | layer={layer_of(name)} | tensor={name} | shard={shard.name} | expected_shape={expected}")

            if shard_name not in shard_cache:
                logger.log(f"SHARD DISCOVERY BEGIN | shard={shard.name}")
                with safe_open(str(shard), framework="pt", device="cpu") as handle:
                    shard_cache[shard_name] = list(handle.keys())
                logger.log(f"SHARD DISCOVERY OK | shard={shard.name} | tensor_count={len(shard_cache[shard_name])}")

            tensor = load_one_tensor_cpu(logger, shard, name, expected)

            if args.stage in {"cuda", "dequant", "full"}:
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA stage requested but CUDA is unavailable")
                logger.log(f"CUDA COPY ABOUT_TO_START | tensor={name} | bytes={tensor_nbytes(tensor)} | {memory_line()}")
                t0 = perf_counter()
                gpu_tensor = tensor.to(device="cuda", non_blocking=False)
                logger.log(f"CUDA COPY ENQUEUED | tensor={name} | elapsed_ms={(perf_counter()-t0)*1000:.2f} | {memory_line()}")
                logger.log(f"CUDA SYNC ABOUT_TO_START | tensor={name} | {memory_line()}")
                t0 = perf_counter()
                torch.cuda.synchronize()
                logger.log(f"CUDA SYNC OK | tensor={name} | elapsed_ms={(perf_counter()-t0)*1000:.2f} | gpu_shape={tuple(gpu_tensor.shape)} | gpu_dtype={gpu_tensor.dtype} | {memory_line()}")
            else:
                gpu_tensor = None

            if args.stage in {"dequant", "full"} and tensor.ndim == 2 and tensor.dtype == torch.float8_e4m3fn:
                scale_name = name + "_scale_inv"
                scale_shard_name = weight_map.get(scale_name)
                if scale_shard_name is None:
                    raise KeyError(f"missing FP8 scale in index: {scale_name}")
                scale_shard = root / scale_shard_name
                logger.log(f"FP8 SCALE LOAD ABOUT_TO_START | tensor={scale_name} | shard={scale_shard.name}")
                scale = load_one_tensor_cpu(logger, scale_shard, scale_name, None)
                if args.stage == "full":
                    logger.log(f"FP8 DEQUANT ABOUT_TO_START | weight={name} | scale={scale_name} | device=cuda | {memory_line()}")
                    if dequantize_fp8_blockwise is None:
                        raise RuntimeError("could not import dequantize_fp8_blockwise")
                    scale_gpu = scale.to("cuda")
                    torch.cuda.synchronize()
                    t0 = perf_counter()
                    out = dequantize_fp8_blockwise(gpu_tensor, scale_gpu)
                    logger.log(f"FP8 DEQUANT KERNEL ENQUEUED | tensor={name} | elapsed_ms={(perf_counter()-t0)*1000:.2f} | out_shape={tuple(out.shape)} | {memory_line()}")
                    logger.log(f"FP8 DEQUANT SYNC ABOUT_TO_START | tensor={name}")
                    torch.cuda.synchronize()
                    logger.log(f"FP8 DEQUANT OK | tensor={name} | out_dtype={out.dtype} | {memory_line()}")
                    del out, scale_gpu
                del scale

            completed += 1
            logger.log(f"STEP {seq}/{len(selected)} COMPLETE | tensor={name} | completed={completed} | {memory_line()}")
            del tensor, gpu_tensor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.log(f"CLEANUP COMPLETE | tensor={name} | {memory_line()}")

        logger.log(f"TRACE COMPLETE | completed={completed}/{len(selected)} | {memory_line()}")
        logger.log("No crash occurred during the requested load trace.")
        logger.log("=" * 100)
        return 0
    except BaseException as exc:
        logger.log(f"FATAL EXCEPTION | {type(exc).__name__}: {exc} | {memory_line()}")
        logger.log(traceback.format_exc())
        logger.log("PROCESS EXITED AFTER PYTHON EXCEPTION; inspect the last GET_TENSOR/CUDA/DEQUANT line.")
        return 1
    finally:
        logger.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Detailed crash-safe Qwen3.6 Safetensors load tracer")
    p.add_argument("--model", required=True, help="model directory containing model.safetensors.index.json")
    p.add_argument("--layers", default="0:39", help="layer range, e.g. 0:39 or 0")
    p.add_argument("--stage", choices=("cpu", "cuda", "dequant", "full"), default="cpu",
                   help="cpu=SSD->RAM; cuda=plus RAM->VRAM; dequant=plus FP8 scale load; full=plus CUDA FP8 dequant")
    p.add_argument("--limit", type=int, default=None, help="maximum number of selected tensors")
    p.add_argument("--include-global", action="store_true", help="also trace global tensors outside model.layers")
    p.add_argument("--log", default=None, help="log file path; default is timestamped file in current directory")
    return p


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
