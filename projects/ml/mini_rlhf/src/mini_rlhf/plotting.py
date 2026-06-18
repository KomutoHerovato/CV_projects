from __future__ import annotations

from pathlib import Path
from typing import Dict


def plot_training_curves(out_dir: Path, sft_hist: Dict, rm_out: Dict, rl_hist: Dict) -> None:
    """Рисует кривые обучения трёх этапов в один PNG (если есть matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    axes[0, 0].plot(sft_hist["sft_loss"])
    axes[0, 0].set_title("SFT: cross-entropy loss")
    axes[0, 0].set_xlabel("step")

    axes[0, 1].plot(rm_out["rm_loss"], label="loss")
    axes[0, 1].plot(rm_out["rm_train_acc"], label="pref acc")
    axes[0, 1].set_title("Reward model: Bradley-Terry")
    axes[0, 1].set_xlabel("step")
    axes[0, 1].legend()

    axes[1, 0].plot(rl_hist["gt_reward"], label="ground-truth reward")
    axes[1, 0].plot(rl_hist["rm_score"], label="reward-model score")
    axes[1, 0].set_title("RL: alignment progress")
    axes[1, 0].set_xlabel("iter")
    axes[1, 0].legend()

    axes[1, 1].plot(rl_hist["kl"])
    axes[1, 1].set_title("RL: KL(policy || reference)")
    axes[1, 1].set_xlabel("iter")

    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=120)
    plt.close(fig)
