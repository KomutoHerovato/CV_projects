"""mini_rlhf — компактный, но честный RLHF-триплет на маленькой GPT-модели.

Пакет реализует полный alignment-пайплайн на синтетической задаче:

1. SFT          — supervised fine-tuning базовой языковой модели на демонстрациях.
2. Reward Model — обучение reward-головы на парах предпочтений (Bradley-Terry loss).
3. RL           — оптимизация политики через PPO с per-token KL-штрафом к reference
                  (+ более простой REINFORCE-with-baseline для сравнения).

Всё считается на маленьком decoder-only трансформере, написанном с нуля на
PyTorch, поэтому demo и тесты гоняются на CPU за секунды и без скачивания весов.
"""

from .config import RLHFConfig
from .tokenizer import ToyTokenizer
from .model import TinyGPT, PolicyWithValue, RewardModel
from .reward import programmatic_reward

__all__ = [
    "RLHFConfig",
    "ToyTokenizer",
    "TinyGPT",
    "PolicyWithValue",
    "RewardModel",
    "programmatic_reward",
]
