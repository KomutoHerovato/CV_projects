from __future__ import annotations

import argparse
from pathlib import Path

from .features import TopicFeatureConfig
from .pipeline import LentaExperimentConfig, run_lenta_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Lenta.ru publication year.")
    parser.add_argument("--data", type=Path, default=None, help="Path to lenta-ru-news.csv or .csv.gz")
    parser.add_argument("--demo", action="store_true", help="Run on a generated tiny dataset.")
    parser.add_argument("--sample-per-year", type=int, default=700)
    parser.add_argument("--min-year", type=int, default=2006)
    parser.add_argument("--max-year", type=int, default=2019)
    parser.add_argument("--representation", choices=["nmf", "svd", "nmf_svd", "enhanced"], default="enhanced")
    parser.add_argument("--n-topics", type=int, default=60)
    parser.add_argument("--n-word-svd", type=int, default=250)
    parser.add_argument("--n-char-svd", type=int, default=60)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/lenta_year_classifier"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = args.demo or args.data is None
    feature_config = TopicFeatureConfig(
        n_topics=args.n_topics,
        n_word_svd=args.n_word_svd,
        n_char_svd=args.n_char_svd,
        min_df=1 if demo else args.min_df,
    )
    config = LentaExperimentConfig(
        sample_per_year=args.sample_per_year,
        min_year=args.min_year,
        max_year=args.max_year,
        representation=args.representation,
        feature_config=feature_config,
    )
    artifacts = run_lenta_experiment(args.data, config=config, demo=demo, out_dir=args.out_dir)
    print("Best topic experiment:", artifacts["best_experiment"])
    print(artifacts["modal_results"].to_string(index=False))
    print("Sparse TF-IDF metrics:", artifacts["sparse_metrics"])
    print("Artifacts:", args.out_dir.resolve())


if __name__ == "__main__":
    main()
