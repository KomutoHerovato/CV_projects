from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch

from .config import RLHFConfig
from .model import PolicyWithValue
from .reward import programmatic_reward
from .reward_model import train_reward_model
from .rl import clone_reference, train_rl
from .sft import train_sft
from .tokenizer import ToyTokenizer


def resolve_device(name: str) -> torch.device:
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate_policy(
    policy: PolicyWithValue,
    tok: ToyTokenizer,
    cfg: RLHFConfig,
    device: torch.device,
    n_samples: int = 256,
) -> Dict[str, object]:
    """Средняя ground-truth награда политики + примеры генераций."""
    policy.eval()
    prompts = tok.prompt_ids()
    g = torch.Generator(device="cpu").manual_seed(cfg.seed + 7)
    rewards: List[float] = []
    samples: List[str] = []
    for i in range(n_samples):
        p = prompts[torch.randint(0, len(prompts), (1,), generator=g).item()]
        idx = torch.tensor([p], device=device)
        out = policy.generate(idx, cfg.max_new_tokens, cfg.temperature)
        ids = out[0].tolist()
        rewards.append(programmatic_reward(ids, tok))
        if len(samples) < 8:
            samples.append(tok.decode(ids))
    mean_r = float(sum(rewards) / len(rewards))
    return {"mean_gt_reward": mean_r, "samples": samples}


def run_pipeline(cfg: RLHFConfig, out_dir: Path) -> Dict[str, object]:
    """Полный RLHF-прогон: SFT -> Reward Model -> RL. Пишет артефакты в out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    tok = ToyTokenizer()
    policy = PolicyWithValue(tok.vocab_size, cfg.model).to(device)

    # 0) Базовая (необученная) политика — точка отсчёта.
    eval_base = evaluate_policy(policy, tok, cfg, device)

    # 1) SFT
    sft_hist = train_sft(policy, tok, cfg, device)
    eval_sft = evaluate_policy(policy, tok, cfg, device)

    # reference = замороженная SFT-политика (для KL-штрафа)
    reference = clone_reference(policy, cfg, device)

    # 2) Reward model (Bradley-Terry на парах предпочтений)
    rm_out = train_reward_model(tok, cfg, device)
    rm = rm_out["reward_model"]

    # 3) RL (PPO / REINFORCE) с KL-контролем
    rl_hist = train_rl(policy, reference, rm, tok, cfg, device)
    eval_rl = evaluate_policy(policy, tok, cfg, device)

    metrics = {
        "device": str(device),
        "rl_algo": cfg.rl_algo,
        "base_gt_reward": eval_base["mean_gt_reward"],
        "sft_gt_reward": eval_sft["mean_gt_reward"],
        "rl_gt_reward": eval_rl["mean_gt_reward"],
        "rm_final_pref_acc": rm_out["rm_final_pref_acc"],
        "final_kl_to_ref": rl_hist["kl"][-1] if rl_hist["kl"] else None,
    }

    _write_artifacts(out_dir, cfg, metrics, sft_hist, rm_out, rl_hist, eval_base, eval_sft, eval_rl)

    return {
        "metrics": metrics,
        "samples": {
            "base": eval_base["samples"],
            "sft": eval_sft["samples"],
            "rl": eval_rl["samples"],
        },
        "rl_history": rl_hist,
        "out_dir": str(out_dir.resolve()),
    }


def _write_artifacts(out_dir, cfg, metrics, sft_hist, rm_out, rl_hist, eb, es, er) -> None:
    import csv
    import json

    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # история RL по итерациям
    with open(out_dir / "rl_history.csv", "w", newline="") as f:
        w = csv.writer(f)
        keys = list(rl_hist.keys())
        w.writerow(["iter"] + keys)
        for i in range(len(rl_hist[keys[0]])):
            w.writerow([i] + [rl_hist[k][i] for k in keys])

    # примеры генераций на каждой стадии
    with open(out_dir / "samples.txt", "w") as f:
        for name, ev in [("BASE", eb), ("SFT", es), ("RL", er)]:
            f.write(f"=== {name} (mean gt reward = {ev['mean_gt_reward']:.3f}) ===\n")
            for s in ev["samples"]:
                f.write(f"  {s}\n")
            f.write("\n")

    # попытка нарисовать кривые обучения (matplotlib опционален)
    try:
        from .plotting import plot_training_curves

        plot_training_curves(out_dir, sft_hist, rm_out, rl_hist)
    except Exception:
        pass
