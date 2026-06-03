from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_metric(history: pd.DataFrame, metric: str, out_path: Path, window: int = 30) -> None:
    smoothed = (
        history.sort_values(["algorithm", "trial", "episode"])
        .assign(
            value=lambda df: df.groupby(["algorithm", "trial"])[metric].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
        )
        .groupby(["algorithm", "episode"], as_index=False)["value"]
        .mean()
    )

    plt.figure(figsize=(10, 5))
    for algorithm, part in smoothed.groupby("algorithm"):
        plt.plot(part["episode"], part["value"], label=algorithm)
    plt.title(f"RiverSwim: {metric}")
    plt.xlabel("Episode")
    plt.ylabel(metric)
    plt.grid(alpha=0.25)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
