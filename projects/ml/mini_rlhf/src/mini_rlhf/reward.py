from __future__ import annotations

from typing import List

from .tokenizer import ToyTokenizer


def programmatic_reward(token_ids: List[int], tokenizer: ToyTokenizer) -> float:
    """«Истинная» (ground-truth) reward-функция задачи.

    Это *синтетический* аналог человеческих предпочтений: чем больше в тексте
    позитивных слов и чем меньше негативных, тем выше награда. В реальном RLHF
    эту функцию заменяют живые разметчики, а reward-модель учится её
    аппроксимировать. Здесь она нужна, чтобы:

      * генерировать пары предпочтений для обучения reward-модели;
      * честно измерять качество выравнивания политики (ground-truth метрика).

    Награда нормирована примерно в [-1, 1] по числу сгенерированных слов.
    """
    pos = sum(1 for t in token_ids if t in tokenizer.positive_ids)
    neg = sum(1 for t in token_ids if t in tokenizer.negative_ids)
    n_words = sum(
        1 for t in token_ids
        if t not in (tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id)
    )
    if n_words == 0:
        return 0.0
    return (pos - neg) / n_words
