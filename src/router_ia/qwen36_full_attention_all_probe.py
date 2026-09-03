from __future__ import annotations

"""Run the full-attention residual fidelity probe over every Qwen3.6 full-attention layer."""

import argparse
from pathlib import Path

from .qwen36_full_attention_residual_probe import run

FULL_ATTENTION_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--tokens", type=int, default=4)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--tolerance", type=float, default=1e-3)
    args = p.parse_args()

    root = args.root.resolve()
    all_pass = True
    print(f"Full-attention layers: {list(FULL_ATTENTION_LAYERS)}")
    print(f"tokens={args.tokens} device={args.device} seed={args.seed} tolerance={args.tolerance}")

    for layer in FULL_ATTENTION_LAYERS:
        print("\n" + "=" * 72)
        print(f"FULL ATTENTION LAYER {layer}")
        print("=" * 72)
        ok = run(root, layer, args.tokens, args.device, args.seed, args.tolerance)
        all_pass &= ok

    print("\n" + "=" * 72)
    print(f"ALL FULL-ATTENTION LAYERS: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 72)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
