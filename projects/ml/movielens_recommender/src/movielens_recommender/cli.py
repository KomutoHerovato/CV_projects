from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import RecommenderConfig, run_recommender_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MovieLens recommendation experiments.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory containing ml-1m/")
    parser.add_argument("--download", action="store_true", help="Download MovieLens 1M into --data-dir.")
    parser.add_argument("--synthetic", action="store_true", help="Run a tiny generated dataset.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/movielens_recommender"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    synthetic = args.synthetic or args.data_dir is None
    config = RecommenderConfig(
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        k=args.k,
        device=args.device,
        download=args.download,
    )
    artifacts = run_recommender_experiment(args.data_dir, synthetic=synthetic, config=config, out_dir=args.out_dir)
    print(artifacts["metrics"].to_string(index=False))
    print("Matrix factorization:", artifacts["mf_status"])
    print("Artifacts:", args.out_dir.resolve())


if __name__ == "__main__":
    main()
