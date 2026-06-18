"""Ablation по силе KL-штрафа (beta = kl_coef).

Главный демонстрационный эксперимент проекта: показывает компромисс
reward vs KL. При слабом KL политика выжимает из reward-модели максимум, но
улетает далеко от reference и скатывается в reward hacking; при сильном KL
почти не двигается с SFT. Запускается по нескольким сидам для устойчивости.

    PYTHONPATH=src python -m mini_rlhf.ablation
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

from .config import RLHFConfig
from .pipeline import run_pipeline


def run_kl_ablation(
    betas: List[float],
    seeds: List[int],
    device: str = "cpu",
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for beta in betas:
        rewards, kls = [], []
        for seed in seeds:
            cfg = RLHFConfig.demo()
            cfg.kl_coef = beta
            cfg.seed = seed
            cfg.device = device
            # ablation-прогоны мусорить артефактами не нужно — пишем во временную папку
            res = run_pipeline(cfg, out_dir=Path("outputs") / "_ablation_tmp")
            m = res["metrics"]
            rewards.append(m["rl_gt_reward"])
            kls.append(m["final_kl_to_ref"])
        rows.append(
            {
                "kl_coef": beta,
                "reward_mean": mean(rewards),
                "reward_std": pstdev(rewards) if len(rewards) > 1 else 0.0,
                "kl_mean": mean(kls),
                "kl_std": pstdev(kls) if len(kls) > 1 else 0.0,
            }
        )
        print(
            f"beta={beta:<5} reward={rows[-1]['reward_mean']:+.3f}"
            f"±{rows[-1]['reward_std']:.3f}  KL={rows[-1]['kl_mean']:.2f}"
        )
    return rows


def _save(rows: List[Dict[str, float]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "kl_ablation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    betas = [r["kl_coef"] for r in rows]
    rew = [r["reward_mean"] for r in rows]
    rew_s = [r["reward_std"] for r in rows]
    kls = [r["kl_mean"] for r in rows]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    ax[0].errorbar(betas, rew, yerr=rew_s, marker="o", capsize=3)
    ax[0].set_xscale("symlog", linthresh=0.01)
    ax[0].set_xlabel("kl_coef (beta)")
    ax[0].set_ylabel("ground-truth reward после RL")
    ax[0].set_title("Сильный KL придерживает reward")
    ax[0].grid(alpha=0.3)

    sc = ax[1].scatter(kls, rew, c=betas, cmap="viridis", s=80)
    for r in rows:
        ax[1].annotate(f"b={r['kl_coef']}", (r["kl_mean"], r["reward_mean"]),
                       textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax[1].set_xlabel("итоговый KL(policy || reference)")
    ax[1].set_ylabel("ground-truth reward")
    ax[1].set_title("Компромисс reward vs KL")
    ax[1].grid(alpha=0.3)
    fig.colorbar(sc, ax=ax[1], label="beta")

    fig.tight_layout()
    fig.savefig(out_dir / "kl_ablation.png", dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="KL-coefficient ablation для mini RLHF.")
    p.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.1, 0.3])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--out-dir", type=Path, default=Path("assets"))
    args = p.parse_args()

    rows = run_kl_ablation(args.betas, args.seeds, args.device)
    _save(rows, args.out_dir)
    print(f"\nсохранено в {args.out_dir.resolve()} (kl_ablation.png, kl_ablation.csv)")


if __name__ == "__main__":
    main()
