from __future__ import annotations

from typing import List

# --- Синтетический «язык отзывов» ----------------------------------------
# Маленький word-level словарь. Слова осмысленно разбиты на группы, чтобы
# можно было задать прозрачную программную reward-функцию (см. reward.py) и
# при этом текст оставался читаемым человеком.

POSITIVE_WORDS = [
    "good", "great", "love", "happy", "wonderful", "excellent", "amazing", "nice",
]
NEGATIVE_WORDS = [
    "bad", "terrible", "hate", "sad", "awful", "horrible", "ugly", "boring",
]
NEUTRAL_WORDS = [
    "the", "movie", "film", "was", "is", "this", "it", "and", "really", "very",
]

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]

# Промпты, с которых начинается генерация (заданы как последовательности слов).
# Все промпты одной длины (3 слова) — это упрощает выравнивание per-token
# log-prob'ов в PPO (фиксированная длина префикса во всём батче).
PROMPTS: List[List[str]] = [
    ["this", "movie", "was"],
    ["the", "film", "is"],
    ["it", "was", "really"],
    ["this", "film", "very"],
]


class ToyTokenizer:
    """Минимальный word-level токенизатор с фиксированным словарём."""

    def __init__(self) -> None:
        vocab = list(SPECIAL_TOKENS) + NEUTRAL_WORDS + POSITIVE_WORDS + NEGATIVE_WORDS
        # убираем возможные дубли, сохраняя порядок
        seen = {}
        self.itos: List[str] = []
        for tok in vocab:
            if tok not in seen:
                seen[tok] = len(self.itos)
                self.itos.append(tok)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]

        self.positive_ids = {self.stoi[w] for w in POSITIVE_WORDS}
        self.negative_ids = {self.stoi[w] for w in NEGATIVE_WORDS}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, words: List[str], add_bos: bool = True) -> List[int]:
        ids = [self.stoi[w] for w in words]
        if add_bos:
            ids = [self.bos_id] + ids
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        out = []
        for i in ids:
            tok = self.itos[int(i)]
            if skip_special and tok in SPECIAL_TOKENS:
                continue
            out.append(tok)
        return " ".join(out)

    def prompt_ids(self) -> List[List[int]]:
        return [self.encode(p, add_bos=True) for p in PROMPTS]
