from __future__ import annotations

"""CUDA-first single-token Qwen3.6 runner with preflight/load diagnostics."""

import argparse
import ctypes
import gc
import json
import os
from collections import OrderedDict
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from safetensors import safe_open

from .qwen36_gated_norm_probe import gated_rmsnorm
from .qwen36_op_probe import BLOCK, dequantize_fp8_blockwise, load_embedding_row, load_tensor, rmsnorm
from .qwen36_router import route

HIDDEN = 2048
LINEAR_NUM_K_HEADS = 16
LINEAR_NUM_V_HEADS = 32
LINEAR_KEY_DIM = LINEAR_NUM_K_HEADS * 128
LINEAR_VALUE_DIM = LINEAR_NUM_V_HEADS * 128
FULL_NUM_HEADS = 16
FULL_NUM_KV_HEADS = 2
FULL_HEAD_DIM = 256
FULL_Q_DIM = FULL_NUM_HEADS * FULL_HEAD_DIM
FULL_NUM_KV_GROUPS = FULL_NUM_HEADS // FULL_NUM_KV_HEADS
EPS = 1e-6
DEFAULT_LAYERS = 40
DEFAULT_CACHE_MIB = 256.0
SAFE_CACHE_LIMIT_MIB = 512.0

EXPECTED_SUFFIX_SHAPES = {
    "input_layernorm.weight": (HIDDEN,),
    "post_attention_layernorm.weight": (HIDDEN,),
    "mlp.gate.weight": (256, HIDDEN),
    "mlp.shared_expert_gate.weight": (1, HIDDEN),
    "linear_attn.in_proj_qkv.weight": (8192, HIDDEN),
    "linear_attn.conv1d.weight": (8192, 1, 4),
    "linear_attn.A_log": (32,),
    "linear_attn.dt_bias": (32,),
    "linear_attn.in_proj_a.weight": (32, HIDDEN),
    "linear_attn.in_proj_b.weight": (32, HIDDEN),
    "linear_attn.in_proj_z.weight": (4096, HIDDEN),
    "linear_attn.out_proj.weight": (HIDDEN, 4096),
    "self_attn.q_proj.weight": (8192, HIDDEN),
    "self_attn.k_proj.weight": (512, HIDDEN),
    "self_attn.v_proj.weight": (512, HIDDEN),
    "self_attn.q_norm.weight": (256,),
    "self_attn.k_norm.weight": (256,),
    "self_attn.o_proj.weight": (HIDDEN, 4096),
    "mlp.experts.0.gate_proj.weight": (512, HIDDEN),
    "mlp.experts.0.up_proj.weight": (512, HIDDEN),
    "mlp.experts.0.down_proj.weight": (HIDDEN, 512),
    "mlp.shared_expert.gate_proj.weight": (512, HIDDEN),
    "mlp.shared_expert.up_proj.weight": (512, HIDDEN),
    "mlp.shared_expert.down_proj.weight": (HIDDEN, 512),
}


def available_system_memory_bytes() -> int | None:
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        status = MEMORYSTATUSEX(); status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys)
        return None
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):
        return None


def tensor_bytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def memory_snapshot(device: str = "cuda") -> str:
    parts = []
    ram = available_system_memory_bytes()
    if ram is not None:
        parts.append(f"RAM livre={ram / 1024**2:.1f} MiB")
    if device == "cuda" and torch.cuda.is_available():
        parts.append(f"VRAM alloc={torch.cuda.memory_allocated()/1024**2:.1f} MiB")
        parts.append(f"VRAM reserv={torch.cuda.memory_reserved()/1024**2:.1f} MiB")
    return " | ".join(parts)


def validate_loaded_tensor(name: str, tensor: torch.Tensor, *, expected_shape: tuple[int, ...] | None = None,
                           finite: bool = True) -> None:
    if tensor.numel() == 0:
        raise ValueError(f"empty tensor: {name}")
    if expected_shape is not None and tuple(tensor.shape) != expected_shape:
        raise ValueError(f"shape mismatch: {name}: got {tuple(tensor.shape)}, expected {expected_shape}")
    if finite and (tensor.is_floating_point() or tensor.is_complex()):
        if not torch.isfinite(tensor.float()).all().item():
            raise ValueError(f"non-finite values: {name}")


def validate_scale_for_weight(name: str, weight: torch.Tensor, scale: torch.Tensor) -> None:
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError(f"scale rank mismatch: {name}: weight={tuple(weight.shape)} scale={tuple(scale.shape)}")
    expected = ((weight.shape[0] + BLOCK - 1)//BLOCK, (weight.shape[1] + BLOCK - 1)//BLOCK)
    if tuple(scale.shape) != expected:
        raise ValueError(f"scale shape mismatch: {name}: got {tuple(scale.shape)}, expected {expected}")
    validate_loaded_tensor(name + ".weight_scale_inv", scale)


def suffix_for_name(name: str) -> str:
    marker = ".layers."
    if marker not in name:
        return name
    tail = name.split(marker, 1)[1]
    if "." in tail:
        tail = tail.split(".", 1)[1]
    parts = tail.split(".")
    if len(parts) >= 4 and parts[0] == "mlp" and parts[1] == "experts":
        return ".".join(parts[:2] + ["0"] + parts[3:])
    return tail


def preflight_model(root: Path, start_layer: int, end_layer: int) -> dict[int, dict[str, object]]:
    """Validate index, real shard membership and tensor shapes without reading tensor data."""
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing index: {index_path}")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("model.safetensors.index.json has no valid weight_map")

    print("=" * 72)
    print("PREFLIGHT: INDEX + SHARD + SHAPE VALIDATION")
    print(f"metadata tensors: {len(weight_map)}")
    print(f"requested layers: {start_layer}..{end_layer}")
    print(f"RAM before preflight: {memory_snapshot('cpu')}")

    errors: list[str] = []
    layer_map: dict[int, dict[str, object]] = {}
    for layer in range(start_layer, end_layer + 1):
        prefix = f"model.language_model.layers.{layer}."
        names = sorted(n for n in weight_map if n.startswith(prefix))
        shard_names = sorted({weight_map[n] for n in names})
        real_names: set[str] = set()
        layer_errors: list[str] = []
        for shard_name in shard_names:
            shard = root / shard_name
            if not shard.is_file():
                layer_errors.append(f"missing shard: {shard_name}")
                continue
            try:
                with safe_open(str(shard), framework="pt", device="cpu") as handle:
                    shard_keys = set(handle.keys())
                    for name in names:
                        if name not in shard_keys:
                            continue
                        real_names.add(name)
                        expected = EXPECTED_SUFFIX_SHAPES.get(suffix_for_name(name))
                        try:
                            view = handle.get_slice(name)
                            shape = tuple(int(v) for v in view.get_shape())
                            if expected is not None and shape != expected:
                                layer_errors.append(f"{name}: shape {shape}, expected {expected}")
                        except Exception as exc:
                            layer_errors.append(f"{name}: metadata read failed: {type(exc).__name__}: {exc}")
            except Exception as exc:
                layer_errors.append(f"cannot open {shard_name}: {type(exc).__name__}: {exc}")

        missing = sorted(set(names) - real_names)
        if missing:
            layer_errors.extend(f"index points to absent tensor: {n}" for n in missing)

        required_any = ("linear_attn.in_proj_qkv.weight", "self_attn.q_proj.weight")
        if not any(prefix + x in real_names for x in required_any):
            layer_errors.append("no recognized attention block")
        required_common = ("input_layernorm.weight", "post_attention_layernorm.weight", "mlp.gate.weight")
        for suffix in required_common:
            if prefix + suffix not in real_names:
                layer_errors.append(f"missing required tensor: {prefix + suffix}")

        kind = "linear" if prefix + "linear_attn.in_proj_qkv.weight" in real_names else \
               "full" if prefix + "self_attn.q_proj.weight" in real_names else "UNKNOWN"
        status = "OK" if not layer_errors else "FAIL"
        print(f"layer {layer:02d}: {status} | index={len(names)} real={len(real_names)} shards={len(shard_names)} type={kind}")
        for err in layer_errors[:12]:
            print(f"  [ERROR] {err}")
        if len(layer_errors) > 12:
            print(f"  ... +{len(layer_errors)-12} more errors")
        errors.extend(f"layer {layer}: {e}" for e in layer_errors)
        layer_map[layer] = {"index_names": names, "real_names": real_names, "missing": missing,
                            "extra": sorted(real_names - set(names)), "shards": shard_names, "kind": kind}

    print(f"RAM after preflight: {memory_snapshot('cpu')}")
    if errors:
        print("PREFLIGHT RESULT: FAIL")
        for e in errors[:80]: print(f"  [ERROR] {e}")
        raise SystemExit("Preflight failed: model/index/shard/shape inconsistency")
    print("PREFLIGHT RESULT: OK")
    print("No execution has started yet. Any later crash is outside this validation stage.")
    print("=" * 72)
    return layer_map


class CPUWeightCache:
    def __init__(self, budget_mib: float) -> None:
        if budget_mib < 0: raise ValueError("cache budget must be non-negative")
        self.budget_bytes = int(budget_mib * 1024**2); self.used_bytes = 0
        self.items: OrderedDict[str, tuple[torch.Tensor, torch.Tensor | None, int]] = OrderedDict()
        self.hits = self.misses = self.evictions = 0
    @staticmethod
    def nbytes(tensor: torch.Tensor) -> int: return tensor_bytes(tensor)
    @classmethod
    def item_nbytes(cls, weight, scale): return cls.nbytes(weight) + (cls.nbytes(scale) if scale is not None else 0)
    def get(self, key):
        item = self.items.get(key)
        if item is None: self.misses += 1; return None
        weight, scale, _ = item; self.items.move_to_end(key); self.hits += 1; return weight, scale
    def put(self, key, weight, scale):
        size = self.item_nbytes(weight, scale)
        if self.budget_bytes == 0 or size > self.budget_bytes: return
        old = self.items.pop(key, None)
        if old: self.used_bytes -= old[2]
        while self.items and self.used_bytes + size > self.budget_bytes:
            _, (_, _, old_size) = self.items.popitem(last=False); self.used_bytes -= old_size; self.evictions += 1
        self.items[key] = (weight, scale, size); self.used_bytes += size
    def stats(self): return self.used_bytes, len(self.items), self.hits, self.misses
    def clear(self): self.items.clear(); self.used_bytes = 0

_CPU_CACHE: CPUWeightCache | None = None

def set_cpu_cache(cache):
    global _CPU_CACHE; _CPU_CACHE = cache

def should_cache_projection(prefix): return ".mlp.experts." in prefix or ".mlp.shared_expert." in prefix

def layer_prefix(layer): return f"model.language_model.layers.{layer}."

def load_layer_weight(root, layer, suffix, device): return load_tensor(root, layer_prefix(layer) + suffix, device=device)

def load_optional_tensor(root, name, device):
    try: return load_tensor(root, name, device=device)
    except KeyError: return None

def attention_type(root, layer):
    prefix = layer_prefix(layer)
    linear = load_optional_tensor(root, prefix + "linear_attn.in_proj_qkv.weight", "cpu")
    if linear is not None: del linear; return "linear_attention"
    full = load_optional_tensor(root, prefix + "self_attn.q_proj.weight", "cpu")
    if full is not None: del full; return "full_attention"
    raise KeyError(f"Could not identify attention type for layer {layer}")

def dequantize_fp8_target(weight, scale_inv, device):
    validate_scale_for_weight("dequant", weight, scale_inv)
    if device != "cuda": return dequantize_fp8_blockwise(weight, scale_inv)
    w = weight.to(device=device, non_blocking=True); s = scale_inv.to(device=device, non_blocking=True).float()
    expanded = s.repeat_interleave(BLOCK, dim=0).repeat_interleave(BLOCK, dim=1)
    out = w.float() * expanded[:w.shape[0], :w.shape[1]]
    del w, s, expanded
    validate_loaded_tensor("dequant output", out)
    return out

def load_fp8_projection_target(root, prefix, device):
    cache_key = prefix if should_cache_projection(prefix) and device == "cuda" else None
    try:
        if cache_key is not None and _CPU_CACHE is not None:
            cached = _CPU_CACHE.get(cache_key)
            if cached is not None:
                weight, scale = cached
                validate_loaded_tensor(prefix + ".weight[cached]", weight, expected_shape=EXPECTED_SUFFIX_SHAPES.get(suffix_for_name(prefix + ".weight")))
                if weight.dtype == torch.float8_e4m3fn:
                    if scale is None: raise ValueError(f"missing scale for FP8 tensor {prefix}")
                    return dequantize_fp8_target(weight, scale, device)
                return weight.float().to(device)
        weight = load_tensor(root, prefix + ".weight", device="cpu")
        expected = EXPECTED_SUFFIX_SHAPES.get(suffix_for_name(prefix + ".weight"))
        validate_loaded_tensor(prefix + ".weight", weight, expected_shape=expected)
        scale = None
        if weight.dtype == torch.float8_e4m3fn:
            scale = load_tensor(root, prefix + ".weight_scale_inv", device="cpu")
            validate_scale_for_weight(prefix, weight, scale)
            out = dequantize_fp8_target(weight, scale, device)
        else:
            out = weight.float().to(device)
        if cache_key is not None and _CPU_CACHE is not None: _CPU_CACHE.put(cache_key, weight, scale)
        else: del weight, scale
        return out
    except Exception as exc:
        raise RuntimeError(f"LOAD FAILED: {prefix} | {type(exc).__name__}: {exc} | {memory_snapshot(device)}") from exc

def load_moe_projection_target(root, layer, expert, kind, device):
    return load_fp8_projection_target(root, f"{layer_prefix(layer)}mlp.experts.{expert}.{kind}", device)

def load_shared_projection_target(root, layer, kind, device):
    return load_fp8_projection_target(root, f"{layer_prefix(layer)}mlp.shared_expert.{kind}", device)

def linear_attention_step(root, layer, x0, device):
    prefix = layer_prefix(layer); input_norm = load_layer_weight(root, layer, "input_layernorm.weight", device); h = rmsnorm(x0, input_norm)
    qkv_w = load_tensor(root, prefix + "linear_attn.in_proj_qkv.weight", device="cpu"); qkv_scale = load_tensor(root, prefix + "linear_attn.in_proj_qkv.weight_scale_inv", device="cpu")
    validate_loaded_tensor(prefix+"qkv", qkv_w, expected_shape=(8192, HIDDEN)); validate_scale_for_weight(prefix+"qkv", qkv_w, qkv_scale)
    qkv_w = dequantize_fp8_target(qkv_w, qkv_scale, device)
    mixed = F.linear(h.float(), qkv_w.float()).reshape(1, 8192); conv_w = load_layer_weight(root, layer, "linear_attn.conv1d.weight", device).float()
    mixed = F.silu(mixed * conv_w[:, 0, -1].reshape(1, -1)); q,k,v = torch.split(mixed,[LINEAR_KEY_DIM,LINEAR_KEY_DIM,LINEAR_VALUE_DIM],dim=-1)
    q=q.reshape(1,16,128).repeat_interleave(2,dim=1); k=k.reshape(1,16,128).repeat_interleave(2,dim=1); v=v.reshape(1,32,128)
    a_w=load_fp8_projection_target(root,prefix+"linear_attn.in_proj_a",device); b_w=load_fp8_projection_target(root,prefix+"linear_attn.in_proj_b",device)
    a_log=load_layer_weight(root,layer,"linear_attn.A_log",device).float().reshape(1,32); dt_bias=load_layer_weight(root,layer,"linear_attn.dt_bias",device).float().reshape(1,32)
    a_raw=F.linear(h.float(),a_w.float()).reshape(1,32); b_raw=F.linear(h.float(),b_w.float()).reshape(1,32); beta=torch.sigmoid(b_raw); g=-torch.exp(a_log)*F.softplus(a_raw+dt_bias); decay=torch.exp(g)
    qn=F.normalize(q.float(),dim=-1,eps=EPS)*(128**-0.5); kn=F.normalize(k.float(),dim=-1,eps=EPS)
    state=torch.zeros(1,32,128,128,device=device,dtype=torch.float32); state=state*decay.unsqueeze(-1).unsqueeze(-1); retrieved=torch.einsum("bhkd,bhk->bhd",state,kn); delta=(v.float()-retrieved)*beta.unsqueeze(-1); state=state+kn.unsqueeze(-1)*delta.unsqueeze(-2); attn=torch.einsum("bhkd,bhk->bhd",state,qn)
    z_w=load_fp8_projection_target(root,prefix+"linear_attn.in_proj_z",device); z=F.linear(h.float(),z_w.float()).reshape(1,32,128); norm_w=load_layer_weight(root,layer,"linear_attn.norm.weight",device); gated,_,_=gated_rmsnorm(attn,z,norm_w); out_w=load_fp8_projection_target(root,prefix+"linear_attn.out_proj",device); attn_projected=F.linear(gated.reshape(1,4096),out_w.float()); residual=x0.reshape(1,HIDDEN)+attn_projected
    del input_norm,h,qkv_w,qkv_scale,mixed,conv_w,q,k,v,a_w,b_w,a_log,dt_bias,a_raw,b_raw,beta,g,decay,qn,kn,state,retrieved,delta,attn,z_w,z,norm_w,gated,out_w,attn_projected
    return residual

def full_attention_step(root,layer,x0,device):
    prefix=layer_prefix(layer); input_norm=load_layer_weight(root,layer,"input_layernorm.weight",device); h=rmsnorm(x0,input_norm)
    q_w=load_tensor(root,prefix+"self_attn.q_proj.weight",device="cpu"); q_scale=load_tensor(root,prefix+"self_attn.q_proj.weight_scale_inv",device="cpu"); validate_loaded_tensor(prefix+"q",q_w,expected_shape=(8192,HIDDEN)); validate_scale_for_weight(prefix+"q",q_w,q_scale); q_w=dequantize_fp8_target(q_w,q_scale,device)
    k_w=load_tensor(root,prefix+"self_attn.k_proj.weight",device="cpu"); k_scale=load_tensor(root,prefix+"self_attn.k_proj.weight_scale_inv",device="cpu"); validate_loaded_tensor(prefix+"k",k_w,expected_shape=(512,HIDDEN)); validate_scale_for_weight(prefix+"k",k_w,k_scale); k_w=dequantize_fp8_target(k_w,k_scale,device)
    v_w=load_tensor(root,prefix+"self_attn.v_proj.weight",device="cpu"); v_scale=load_tensor(root,prefix+"self_attn.v_proj.weight_scale_inv",device="cpu"); validate_loaded_tensor(prefix+"v",v_w,expected_shape=(512,HIDDEN)); validate_scale_for_weight(prefix+"v",v_w,v_scale); v_w=dequantize_fp8_target(v_w,v_scale,device)
    q_gate=F.linear(h.float(),q_w.float()).reshape(1,16,512); q,gate=torch.chunk(q_gate,2,dim=-1); k=F.linear(h.float(),k_w.float()).reshape(1,2,256); v=F.linear(h.float(),v_w.float()).reshape(1,2,256); q_norm_w=load_layer_weight(root,layer,"self_attn.q_norm.weight",device); k_norm_w=load_layer_weight(root,layer,"self_attn.k_norm.weight",device); q=rmsnorm(q,q_norm_w).float(); k=rmsnorm(k,k_norm_w).float(); k=k.repeat_interleave(8,dim=1); v=v.repeat_interleave(8,dim=1)
    scores=torch.matmul(q.unsqueeze(2),k.transpose(-1,-2)).squeeze(-2)*(256**-0.5); attn_weights=torch.softmax(scores.float(),dim=-1); attn=torch.matmul(attn_weights.unsqueeze(-2),v).squeeze(-2); attn=attn*torch.sigmoid(gate); attn_flat=attn.reshape(1,4096)
    out_w=load_tensor(root,prefix+"self_attn.o_proj.weight",device="cpu"); out_scale=load_tensor(root,prefix+"self_attn.o_proj.weight_scale_inv",device="cpu"); validate_loaded_tensor(prefix+"o",out_w,expected_shape=(HIDDEN,4096)); validate_scale_for_weight(prefix+"o",out_w,out_scale); out_w=dequantize_fp8_target(out_w,out_scale,device); attn_projected=F.linear(attn_flat,out_w.float()); residual=x0.reshape(1,HIDDEN)+attn_projected
    del input_norm,h,q_w,q_scale,k_w,k_scale,v_w,v_scale,q_gate,q,gate,k,v,q_norm_w,k_norm_w,scores,attn_weights,attn,attn_flat,out_w,out_scale,attn_projected
    return residual

def run_routed_expert(root,layer,expert,x,device):
    gate_w=load_moe_projection_target(root,layer,expert,"gate_proj",device); up_w=load_moe_projection_target(root,layer,expert,"up_proj",device); down_w=load_moe_projection_target(root,layer,expert,"down_proj",device); gate=F.linear(x,gate_w.float()); up=F.linear(x,up_w.float()); hidden=F.silu(gate)*up; out=F.linear(hidden,down_w.float()); del gate,up,hidden; return out

def run_shared_expert(root,layer,x,device):
    gate_w=load_shared_projection_target(root,layer,"gate_proj",device); up_w=load_shared_projection_target(root,layer,"up_proj",device); down_w=load_shared_projection_target(root,layer,"down_proj",device); shared_gate_w=load_layer_weight(root,layer,"mlp.shared_expert_gate.weight",device).float(); gate=torch.sigmoid(F.linear(x,shared_gate_w)); hidden_gate=F.linear(x,gate_w.float()); up=F.linear(x,up_w.float()); hidden=F.silu(hidden_gate)*up; raw=F.linear(hidden,down_w.float()); out=raw*gate; gate_value=float(gate.item()); del gate_w,up_w,down_w,shared_gate_w,gate,hidden_gate,up,hidden,raw; return out,gate_value

def moe_step(root,layer,residual,top_k,device):
    post_norm=load_layer_weight(root,layer,"post_attention_layernorm.weight",device); moe_in=rmsnorm(residual,post_norm).reshape(1,HIDDEN).float(); router_w=load_layer_weight(root,layer,"mlp.gate.weight",device).float(); routed=route(moe_in.reshape(-1),router_w,top_k=top_k); expert_ids=[int(v) for v in routed.expert_ids.detach().cpu().tolist()]; weights=[float(v) for v in routed.weights.detach().cpu().tolist()]; routed_sum=torch.zeros_like(moe_in)
    for expert_id,weight in zip(expert_ids,weights):
        out=run_routed_expert(root,layer,expert_id,moe_in,device); routed_sum.add_(out,alpha=weight); del out
    shared_out,shared_gate=run_shared_expert(root,layer,moe_in,device); moe_out=routed_sum+shared_out; layer_out=residual+moe_out; moe_input_norm=float(torch.linalg.vector_norm(moe_in).item()); del post_norm,moe_in,router_w,routed,routed_sum,shared_out,moe_out; gc.collect(); return layer_out,expert_ids,weights,shared_gate,moe_input_norm

def main():
    parser=argparse.ArgumentParser(description="CUDA-first Qwen3.6 single-token loop with loader diagnostics"); parser.add_argument("root",type=Path); parser.add_argument("--token-id",type=int,default=0); parser.add_argument("--start-layer",type=int,default=0); parser.add_argument("--end-layer",type=int,default=39); parser.add_argument("--top-k",type=int,default=8); parser.add_argument("--cache-mib",type=float,default=DEFAULT_CACHE_MIB); parser.add_argument("--allow-large-cache",action="store_true"); parser.add_argument("--skip-preflight",action="store_true",help="skip index/shard validation (not recommended for diagnostics)"); args=parser.parse_args()
    if not torch.cuda.is_available(): raise SystemExit("CUDA unavailable")
    if not 0<=args.start_layer<=args.end_layer<DEFAULT_LAYERS: raise SystemExit(f"layer range must be inside 0..{DEFAULT_LAYERS-1}")
    if args.cache_mib<0: raise SystemExit("cache-mib must be non-negative")
    if args.cache_mib>SAFE_CACHE_LIMIT_MIB and not args.allow_large_cache: raise SystemExit(f"cache-mib={args.cache_mib:.1f} is above safe limit {SAFE_CACHE_LIMIT_MIB:.0f} MiB")
    root=args.root.resolve()
    if not args.skip_preflight: preflight_model(root,args.start_layer,args.end_layer)
    available=available_system_memory_bytes(); requested_bytes=int(args.cache_mib*1024**2)
    if available is not None and requested_bytes>0:
        limit=int(available*0.20)
        if requested_bytes>limit and not args.allow_large_cache: args.cache_mib=max(0,limit/1024**2)
    cache=CPUWeightCache(args.cache_mib); set_cpu_cache(cache); device="cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}"); print(f"PyTorch CUDA: {torch.version.cuda}"); print(f"CPU FP8 cache budget: {args.cache_mib:.1f} MiB"); print(f"Before execution: {memory_snapshot(device)}")
    try:
        x=load_embedding_row(root,args.token_id).reshape(1,HIDDEN).to(device).float(); validate_loaded_tensor("embedding",x,expected_shape=(1,HIDDEN)); torch.cuda.synchronize(); start_total=perf_counter(); print("EXECUTION START — preflight passed")
        for layer in range(args.start_layer,args.end_layer+1):
            start_layer=perf_counter(); kind="unknown"
            try:
                kind=attention_type(root,layer); residual=linear_attention_step(root,layer,x,device) if kind=="linear_attention" else full_attention_step(root,layer,x,device); x,expert_ids,weights,shared_gate,moe_input_norm=moe_step(root,layer,residual,args.top_k,device); torch.cuda.synchronize()
            except Exception as exc:
                print(f"[LOAD/EXEC ERROR] layer {layer} ({kind}): {type(exc).__name__}: {exc}"); print(f"  memory at failure: {memory_snapshot(device)}"); raise
            layer_ms=(perf_counter()-start_layer)*1000; used,entries,hits,misses=cache.stats(); print(f"layer {layer} ({kind}): router top-{args.top_k}: {expert_ids}"); print(f"  router weights: {[round(v,8) for v in weights]}"); print(f"  shared gate: {shared_gate:.8f}"); print(f"  moe input norm: {moe_input_norm:.8f}"); print(f"  output shape: {tuple(x.shape)}"); print(f"  output norm: {torch.linalg.vector_norm(x).item():.8f}"); print(f"  VRAM allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MiB | reserved: {torch.cuda.memory_reserved()/1024**2:.1f} MiB"); print(f"  RAM free: {(available_system_memory_bytes() or 0)/1024**2:.1f} MiB"); print(f"  RAM cache: {used/1024**2:.1f}/{args.cache_mib:.1f} MiB ({entries} tensors), hits={hits}, misses={misses}, evictions={cache.evictions}"); print(f"  time: {layer_ms:.3f} ms"); del residual; gc.collect()
        torch.cuda.synchronize(); total=(perf_counter()-start_total)*1000; print("EXECUTION RESULT: OK"); print(f"final output norm: {torch.linalg.vector_norm(x).item():.8f}"); print(f"total time: {total:.3f} ms"); print(f"final memory: {memory_snapshot(device)}")
    finally:
        if 'x' in locals(): del x
        cache.clear(); set_cpu_cache(None); gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__=="__main__": main()
