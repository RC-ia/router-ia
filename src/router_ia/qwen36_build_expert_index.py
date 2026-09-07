from __future__ import annotations

"""CLI for building the minimal expert-location map kept in RAM."""

import argparse
from pathlib import Path

from .qwen36_compact_index import build_compact_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact Qwen3.6 expert index")
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = build_compact_index(args.model_dir, args.output)
    size = output.stat().st_size
    print(f"expert_index={output}")
    print(f"bytes={size}")
    print(f"ram_kib={size / 1024.0:.2f}")


if __name__ == "__main__":
    main()
