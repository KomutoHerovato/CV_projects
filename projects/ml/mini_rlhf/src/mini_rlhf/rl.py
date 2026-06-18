from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from .config import RLHFConfig
from .model import PolicyWithValue, RewardModel
from .reward import programmatic_reward
from .tokenizer import ToyTokenizer


# --- мелкие хелперы для rollout'а и подсчёта преимуществ ---

def _logprobs_and_values(
    policy: PolicyWithValue, seq: torch.Tensor, prompt_len: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token log-prob'ы выбранных токенов, энтропия и value на отклике.

    Для токена на позиции t распределение берётся из logits на позиции t-1.
    Возвращаются срезы только по сгенерированному отклику (длина G).
    """
    logits, values = policy(seq)                       # (B, T, V), (B, T)
    logp_all = F.log_softmax(logits[:, :-1, :], dim=-1)  # предсказание токенов 1..T-1
    labels = seq[:, 1:]
    token_logp = logp_all.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    entropy_all = -(logp_all.exp() * logp_all).sum(-1)                   # (B, T-1)

    resp = slice(prompt_len - 1, seq.size(1) - 1)  # позиции, относящиеся к отклику
    return token_logp[:, resp], entropy_all[:, resp], values[:, resp]


def _reward_scores(rm: RewardModel, seq: torch.Tensor, tok: ToyTokenizer) -> torch.Tensor:
    lengths = (seq != tok.pad_id).sum(dim=1)
    return rm(seq, lengths)


def _ground_truth_reward(seq: torch.Tensor, tok: ToyTokenizer) -> float:
    """Средняя истинная награда батча (метрика качества выравнивания)."""
    vals = [programmatic_reward(seq[i].tolist(), tok) for i in range(seq.size(0))]
    return float(sum(vals) / len(vals))


def _gae(
    rewards: torch.Tensor, values: torch.Tensor, gamma: float, lam: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation по токенам отклика."""
    B, L = rewards.shape
    adv = torch.zeros_like(rewards)
    last = torch.zeros(B, device=rewards.device)
    for t in reversed(range(L)):
        next_v = values[:, t + 1] if t + 1 < L else torch.zeros(B, device=rewards.device)
        delta = rewards[:, t] + gamma * next_v - values[:, t]
        last = delta + gamma * lam * last
        adv[:, t] = last
    returns = adv + values
    return adv, returns


# --- сам RL-цикл: rollout -> PPO/REINFORCE update ---

def _rollout(
    policy: PolicyWithValue,
    reference: PolicyWithValue,
    rm: RewardModel,
    tok: ToyTokenizer,
    cfg: RLHFConfig,
    device: torch.device,
    g: torch.Generator,
) -> Dict[str, torch.Tensor]:
    """Сэмплируем батч откликов и считаем награды/KL/преимущества."""
    prompts = tok.prompt_ids()
    prompt_len = len(prompts[0])
    pidx = torch.randint(0, len(prompts), (cfg.rl_batch_size,), generator=g)
    prompt_batch = torch.tensor([prompts[i] for i in pidx.tolist()], device=device)

    with torch.no_grad():
        seq = policy.generate(prompt_batch, cfg.max_new_tokens, cfg.temperature)
        old_logp, _, old_values = _logprobs_and_values(policy, seq, prompt_len)
        ref_logp, _, _ = _logprobs_and_values(reference, seq, prompt_len)
        rm_score = _reward_scores(rm, seq, tok)

    # Собираем наградной сигнал. На каждый токен вешаем штраф за KL к reference,
    # а скор reward-модели добавляем только на последний токен (награда за весь
    # отклик целиком). Если убрать KL или сделать kl_coef слишком маленьким,
    # политика быстро скатывается в повтор позитивных слов - см. README.
    kl = old_logp - ref_logp                       # (B, L)
    token_rewards = -cfg.kl_coef * kl
    # rm_score нормируем по батчу, иначе масштаб награды скачет от итерации
    # к итерации и PPO становится нестабильным.
    rm_norm = (rm_score - rm_score.mean()) / (rm_score.std() + 1e-6)
    token_rewards[:, -1] = token_rewards[:, -1] + rm_norm

    adv, returns = _gae(token_rewards, old_values, cfg.gamma, cfg.gae_lambda)
    adv = (adv - adv.mean()) / (adv.std() + 1e-6)

    return {
        "seq": seq,
        "prompt_len": torch.tensor(prompt_len),
        "old_logp": old_logp,
        "old_values": old_values,
        "advantages": adv,
        "returns": returns,
        "rm_score": rm_score,
        "kl": kl,
    }


def _ppo_update(
    policy: PolicyWithValue,
    opt: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    cfg: RLHFConfig,
) -> Dict[str, float]:
    seq = batch["seq"]
    prompt_len = int(batch["prompt_len"])
    old_logp = batch["old_logp"]
    advantages = batch["advantages"]
    returns = batch["returns"]
    old_values = batch["old_values"]

    stats = {"pg_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clipfrac": 0.0}
    for _ in range(cfg.ppo_epochs):
        new_logp, entropy, values = _logprobs_and_values(policy, seq, prompt_len)
        ratio = torch.exp(new_logp - old_logp)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - cfg.ppo_clip, 1 + cfg.ppo_clip) * advantages
        pg_loss = -torch.min(surr1, surr2).mean()

        # clipped value loss (как в реализациях PPO для LLM)
        v_clipped = old_values + (values - old_values).clamp(-cfg.ppo_clip, cfg.ppo_clip)
        v_loss = torch.max((values - returns) ** 2, (v_clipped - returns) ** 2)
        value_loss = 0.5 * v_loss.mean()

        ent = entropy.mean()
        loss = pg_loss + cfg.value_coef * value_loss - cfg.entropy_coef * ent

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
        opt.step()

        with torch.no_grad():
            clipfrac = ((ratio - 1.0).abs() > cfg.ppo_clip).float().mean().item()
        stats["pg_loss"] = float(pg_loss.item())
        stats["value_loss"] = float(value_loss.item())
        stats["entropy"] = float(ent.item())
        stats["clipfrac"] = float(clipfrac)
    return stats


def _reinforce_update(
    policy: PolicyWithValue,
    opt: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    cfg: RLHFConfig,
    baseline: float,
) -> Tuple[Dict[str, float], float]:
    """REINFORCE с скользящим baseline — для сравнения с PPO.

    Возвращает статистику и обновлённый baseline.
    """
    seq = batch["seq"]
    prompt_len = int(batch["prompt_len"])
    returns = batch["returns"][:, 0]  # суммарный дисконт-возврат с первого токена

    new_logp, entropy, _ = _logprobs_and_values(policy, seq, prompt_len)
    seq_logp = new_logp.sum(dim=1)
    advantage = returns - baseline
    loss = -(advantage.detach() * seq_logp).mean() - cfg.entropy_coef * entropy.mean()

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
    opt.step()

    new_baseline = 0.9 * baseline + 0.1 * float(returns.mean().item())
    return {"pg_loss": float(loss.item()), "entropy": float(entropy.mean().item())}, new_baseline


def train_rl(
    policy: PolicyWithValue,
    reference: PolicyWithValue,
    rm: RewardModel,
    tok: ToyTokenizer,
    cfg: RLHFConfig,
    device: torch.device,
) -> Dict[str, List[float]]:
    """RL-этап RLHF: выравнивание политики по reward-модели с KL-контролем.

    Цель: max E[ r_RM(x) ] - beta * KL(policy || reference). Reward-модель даёт
    обучаемый сигнал, KL не даёт политике уйти от SFT-reference (защита от
    reward hacking и деградации языка).
    """
    reference = reference.to(device).eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    rm = rm.to(device).eval()

    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.rl_lr)
    g = torch.Generator(device="cpu").manual_seed(cfg.seed + 3)

    hist: Dict[str, List[float]] = {
        "gt_reward": [], "rm_score": [], "kl": [], "pg_loss": [], "entropy": [],
    }
    baseline = 0.0
    policy.train()
    for it in range(cfg.rl_iters):
        batch = _rollout(policy, reference, rm, tok, cfg, device, g)

        if cfg.rl_algo == "ppo":
            stats = _ppo_update(policy, opt, batch, cfg)
        elif cfg.rl_algo == "reinforce":
            stats, baseline = _reinforce_update(policy, opt, batch, cfg, baseline)
        else:
            raise ValueError(f"unknown rl_algo: {cfg.rl_algo}")

        hist["gt_reward"].append(_ground_truth_reward(batch["seq"], tok))
        hist["rm_score"].append(float(batch["rm_score"].mean().item()))
        hist["kl"].append(float(batch["kl"].sum(dim=1).mean().item()))
        hist["pg_loss"].append(stats["pg_loss"])
        hist["entropy"].append(stats.get("entropy", 0.0))
    return hist


def clone_reference(policy: PolicyWithValue, cfg: RLHFConfig, device: torch.device) -> PolicyWithValue:
    """Замороженная копия SFT-политики — reference для KL-штрафа."""
    ref = PolicyWithValue(policy.lm_head.out_features, cfg.model).to(device)
    ref.load_state_dict(copy.deepcopy(policy.state_dict()))
    ref.eval()
    return ref
