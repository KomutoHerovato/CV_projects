from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Многоголовое причинное self-attention (как в GPT)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """Маленький decoder-only трансформер. Общий «ствол» для политики и reward."""

    def __init__(self, vocab_size: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.tok_emb = nn.Embedding(vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.block_size, f"sequence {T} > block_size {self.block_size}"
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)  # (B, T, n_embd) — скрытые состояния


class PolicyWithValue(nn.Module):
    """Языковая модель (policy) с LM-головой и value-головой для PPO."""

    def __init__(self, vocab_size: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.backbone = TinyGPT(vocab_size, cfg)
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size, bias=False)
        self.value_head = nn.Linear(cfg.n_embd, 1)
        self.block_size = cfg.block_size

    def forward(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(idx)
        logits = self.lm_head(hidden)            # (B, T, vocab)
        values = self.value_head(hidden).squeeze(-1)  # (B, T)
        return logits, values

    def logits_only(self, idx: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.backbone(idx))

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Авторегрессионная генерация. Возвращает prompt + продолжение."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self.logits_only(idx_cond)[:, -1, :]
            probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


class RewardModel(nn.Module):
    """Reward-модель: трансформерный ствол + скалярная голова на последнем токене."""

    def __init__(self, vocab_size: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.backbone = TinyGPT(vocab_size, cfg)
        self.score_head = nn.Linear(cfg.n_embd, 1)

    def forward(self, idx: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Возвращает скалярную награду на последовательность (B,).

        lengths — индекс последнего непаддингового токена (если есть паддинг).
        """
        hidden = self.backbone(idx)            # (B, T, n_embd)
        scores = self.score_head(hidden).squeeze(-1)  # (B, T)
        if lengths is None:
            return scores[:, -1]
        idx_last = (lengths - 1).clamp(min=0)
        return scores.gather(1, idx_last.view(-1, 1)).squeeze(1)
