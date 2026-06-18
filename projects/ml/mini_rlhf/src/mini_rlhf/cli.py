from __future__ import annotations

import argparse
from pathlib import Path

from .config import RLHFConfig
from .pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="mini RLHF: SFT -> Reward Model (Bradley-Terry) -> RL (PPO/REINFORCE)."
    )
    p.add_argument("--algo", choices=["ppo", "reinforce"], default="ppo")
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu",
                   help="на M1 используйте mps для full-режима")
    p.add_argument("--rl-iters", type=int, default=None, help="число RL-итераций")
    p.add_argument("--kl-coef", type=float, default=None, help="beta: вес KL-штрафа")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/mini_rlhf"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RLHFConfig.demo()
    cfg.rl_algo = args.algo
    cfg.device = args.device
    cfg.seed = args.seed
    if args.rl_iters is not None:
        cfg.rl_iters = args.rl_iters
    if args.kl_coef is not None:
        cfg.kl_coef = args.kl_coef

    result = run_pipeline(cfg, args.out_dir)
    m = result["metrics"]
    print("=== mini RLHF ===")
    print(f"device           : {m['device']}")
    print(f"RL algo          : {m['rl_algo']}")
    print(f"RM pref accuracy : {m['rm_final_pref_acc']:.3f}")
    print("ground-truth reward (выше = позитивнее текст):")
    print(f"  base : {m['base_gt_reward']:+.3f}")
    print(f"  SFT  : {m['sft_gt_reward']:+.3f}")
    print(f"  RL   : {m['rl_gt_reward']:+.3f}")
    print(f"final KL(policy||ref): {m['final_kl_to_ref']:.3f}")
    print("\nпримеры после RL:")
    for s in result["samples"]["rl"][:5]:
        print(f"  {s}")
    print(f"\nартефакты: {result['out_dir']}")


if __name__ == "__main__":
    main()
