from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .pipeline import run_detector_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Human vs AI text detection baseline.")
    parser.add_argument("--data", type=Path, default=None, help="CSV with text and label/generated columns.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/ai_text_detector"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.data) if args.data else None
    artifacts = run_detector_experiment(data=data, out_dir=args.out_dir)
    print("Metrics:", artifacts["metrics"])
    print("BERT setup:", artifacts["bert_setup"])
    print("Artifacts:", args.out_dir.resolve())


if __name__ == "__main__":
    main()
