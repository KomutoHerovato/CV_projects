from __future__ import annotations

from typing import List, Tuple

import torch

from .config import RLHFConfig
from .reward import programmatic_reward
from .tokenizer import (
    ToyTokenizer,
    NEUTRAL_WORDS,
    POSITIVE_WORDS,
    NEGATIVE_WORDS,
)


def _pad(seqs: List[List[int]], pad_id: int, block_size: int) -> torch.Tensor:
    maxlen = min(block_size, max(len(s) for s in seqs))
    out = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        s = s[:maxlen]
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def _sample_completion(
    tok: ToyTokenizer,
    gen: torch.Generator,
    n_words: int,
    pos_p: float = 1 / 3,
    neg_p: float = 1 / 3,
) -> List[int]:
    """Случайное продолжение из словаря с заданными долями pos/neg/neutral слов."""
    neutral_ids = [tok.stoi[w] for w in NEUTRAL_WORDS]
    pos_ids = [tok.stoi[w] for w in POSITIVE_WORDS]
    neg_ids = [tok.stoi[w] for w in NEGATIVE_WORDS]
    probs = torch.tensor([pos_p, neg_p, max(0.0, 1.0 - pos_p - neg_p)])
    out: List[int] = []
    for _ in range(n_words):
        bucket = torch.multinomial(probs, 1, generator=gen).item()
        pool = [pos_ids, neg_ids, neutral_ids][bucket]
        j = torch.randint(0, len(pool), (1,), generator=gen).item()
        out.append(pool[j])
    return out


def make_sft_dataset(tok: ToyTokenizer, cfg: RLHFConfig, n: int = 512) -> torch.Tensor:
    """Демонстрации для SFT: prompt + сбалансированное (нейтральное) продолжение.

    Reference-политика после SFT владеет «грамматикой» задачи, но в среднем
    нейтральна по тональности — выравнивание к позитиву делает уже RL.
    """
    gen = torch.Generator().manual_seed(cfg.seed)
    prompts = tok.prompt_ids()
    seqs: List[List[int]] = []
    for i in range(n):
        prompt = prompts[i % len(prompts)]
        comp = _sample_completion(tok, gen, n_words=cfg.max_new_tokens)
        seq = prompt + comp + [tok.eos_id]
        seqs.append(seq[: cfg.model.block_size])
    return _pad(seqs, tok.pad_id, cfg.model.block_size)


def make_preference_pairs(
    tok: ToyTokenizer, cfg: RLHFConfig, n: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Пары (chosen, rejected) для обучения reward-модели.

    Для каждой пары берём два случайных продолжения одного промпта и назначаем
    «победителя» по ground-truth программной reward-функции — синтетический
    аналог разметки человеческих предпочтений.
    """
    gen = torch.Generator().manual_seed(cfg.seed + 1)
    prompts = tok.prompt_ids()
    chosen: List[List[int]] = []
    rejected: List[List[int]] = []
    tries = 0
    while len(chosen) < n and tries < n * 20:
        tries += 1
        prompt = prompts[torch.randint(0, len(prompts), (1,), generator=gen).item()]
        a = prompt + _sample_completion(tok, gen, cfg.max_new_tokens) + [tok.eos_id]
        b = prompt + _sample_completion(tok, gen, cfg.max_new_tokens) + [tok.eos_id]
        ra = programmatic_reward(a, tok)
        rb = programmatic_reward(b, tok)
        if ra == rb:
            continue  # пропускаем неинформативные пары
        if ra > rb:
            chosen.append(a[: cfg.model.block_size])
            rejected.append(b[: cfg.model.block_size])
        else:
            chosen.append(b[: cfg.model.block_size])
            rejected.append(a[: cfg.model.block_size])

    ch = _pad(chosen, tok.pad_id, cfg.model.block_size)
    rj = _pad(rejected, tok.pad_id, cfg.model.block_size)
    ch_len = (ch != tok.pad_id).sum(dim=1)
    rj_len = (rj != tok.pad_id).sum(dim=1)
    return ch, ch_len, rj, rj_len
