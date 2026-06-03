from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import ExperimentConfig, run_comparison
from .plotting import plot_metric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RiverSwim PSRL vs Q-learning.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--n-intermediate", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/riverswim_psrl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        n_intermediate=args.n_intermediate,
        horizon=args.horizon,
        episodes=args.episodes,
        trials=args.trials,
    )
    history, summary = run_comparison(config)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(args.out_dir / "history.csv", index=False)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    for metric in ["reward", "right_visits", "left_visits", "swim_share", "regret"]:
        plot_metric(history, metric, args.out_dir / f"{metric}.png")

    print(summary.to_string(index=False))
    print(json.dumps({"out_dir": str(args.out_dir.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
