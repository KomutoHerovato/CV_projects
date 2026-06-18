from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict


@dataclass
class ModelConfig:
    """Гиперпараметры маленького decoder-only трансформера."""

    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 2
    block_size: int = 24
    dropout: float = 0.0


@dataclass
class RLHFConfig:
    """Единый конфиг всего пайплайна.

    Значения по умолчанию подобраны под demo: весь SFT -> RM -> PPO прогон
    на CPU занимает считанные секунды. Для более «настоящего» прогона
    увеличьте число шагов (см. README, раздел про full-режим).
    """

    seed: int = 0
    device: str = "cpu"  # "cpu" | "mps" | "cuda"; resolve_device() уточнит

    model: ModelConfig = field(default_factory=ModelConfig)

    # --- генерация ---
    max_new_tokens: int = 8
    temperature: float = 1.0

    # --- SFT ---
    sft_steps: int = 120
    sft_batch_size: int = 32
    sft_lr: float = 3e-3

    # --- Reward model ---
    rm_steps: int = 150
    rm_batch_size: int = 32
    rm_lr: float = 3e-3
    rm_n_pairs: int = 512

    # --- RL (PPO) ---
    rl_algo: str = "ppo"  # "ppo" | "reinforce"
    rl_iters: int = 40
    rl_batch_size: int = 32
    rl_lr: float = 1e-3
    kl_coef: float = 0.05         # beta: штраф за отклонение от reference-политики
    ppo_epochs: int = 4
    ppo_clip: float = 0.2
    gae_lambda: float = 0.95
    gamma: float = 1.0
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def demo(cls) -> "RLHFConfig":
        """Быстрый воспроизводимый прогон (по умолчанию)."""
        return cls()

    @classmethod
    def smoke(cls) -> "RLHFConfig":
        """Минимальный конфиг для тестов: пара шагов на каждом этапе."""
        cfg = cls()
        cfg.model = ModelConfig(n_embd=32, n_head=2, n_layer=1, block_size=24)
        cfg.sft_steps = 8
        cfg.rm_steps = 8
        cfg.rm_n_pairs = 64
        cfg.rl_iters = 3
        cfg.rl_batch_size = 16
        cfg.ppo_epochs = 2
        return cfg
