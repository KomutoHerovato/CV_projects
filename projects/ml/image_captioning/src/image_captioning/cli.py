from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_captioning_demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run image captioning demo pipeline.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/image_captioning"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_captioning_demo(out_dir=args.out_dir)
    print("Metrics:", artifacts["metrics"])
    print("Model setup:", artifacts["model_setup"])
    print("Decoded example:", artifacts["decoded_example"])
    print("Artifacts:", args.out_dir.resolve())


if __name__ == "__main__":
    main()
