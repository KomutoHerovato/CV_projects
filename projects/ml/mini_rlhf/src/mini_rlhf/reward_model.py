from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from .config import RLHFConfig
from .data import make_preference_pairs
from .model import RewardModel
from .tokenizer import ToyTokenizer


def train_reward_model(
    tok: ToyTokenizer,
    cfg: RLHFConfig,
    device: torch.device,
) -> Dict[str, object]:
    """Обучение reward-модели на парах предпочтений (Bradley-Terry loss).

    Модель Брэдли-Терри: P(chosen > rejected) = sigmoid(r_c - r_r).
    Максимизация log-правдоподобия предпочтений эквивалентна минимизации
    loss = -log sigmoid(r_c - r_r). Это ровно тот reward-моделинг, что лежит
    в основе RLHF — здесь предпочтения синтетические, но механизм настоящий.
    """
    rm = RewardModel(tok.vocab_size, cfg.model).to(device)
    opt = torch.optim.AdamW(rm.parameters(), lr=cfg.rm_lr)

    ch, ch_len, rj, rj_len = make_preference_pairs(tok, cfg, n=cfg.rm_n_pairs)
    ch, ch_len = ch.to(device), ch_len.to(device)
    rj, rj_len = rj.to(device), rj_len.to(device)

    n = ch.size(0)
    g = torch.Generator(device="cpu").manual_seed(cfg.seed + 2)

    loss_hist: List[float] = []
    acc_hist: List[float] = []
    rm.train()
    for step in range(cfg.rm_steps):
        sel = torch.randint(0, n, (min(cfg.rm_batch_size, n),), generator=g)
        r_c = rm(ch[sel], ch_len[sel])
        r_r = rm(rj[sel], rj_len[sel])
        loss = -F.logsigmoid(r_c - r_r).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_hist.append(float(loss.item()))
        acc_hist.append(float((r_c > r_r).float().mean().item()))

    # финальная accuracy на всём наборе пар (доля верно упорядоченных)
    rm.eval()
    with torch.no_grad():
        final_acc = float((rm(ch, ch_len) > rm(rj, rj_len)).float().mean().item())

    return {
        "reward_model": rm,
        "rm_loss": loss_hist,
        "rm_train_acc": acc_hist,
        "rm_final_pref_acc": final_acc,
    }
