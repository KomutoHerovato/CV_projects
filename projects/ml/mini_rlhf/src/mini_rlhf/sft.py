from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from .config import RLHFConfig
from .data import make_sft_dataset
from .model import PolicyWithValue
from .tokenizer import ToyTokenizer


def train_sft(
    policy: PolicyWithValue,
    tok: ToyTokenizer,
    cfg: RLHFConfig,
    device: torch.device,
) -> Dict[str, List[float]]:
    """Supervised fine-tuning: next-token prediction на демонстрациях.

    Это первый этап RLHF: модель учится формату/«грамматике» задачи. Loss —
    обычная causal-LM кросс-энтропия со сдвигом, паддинг игнорируется.
    """
    data = make_sft_dataset(tok, cfg, n=1024).to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.sft_lr)
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)

    history: List[float] = []
    policy.train()
    for step in range(cfg.sft_steps):
        idx = torch.randint(0, data.size(0), (cfg.sft_batch_size,), generator=g)
        batch = data[idx]
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        logits = policy.logits_only(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=tok.pad_id,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
    return {"sft_loss": history}
