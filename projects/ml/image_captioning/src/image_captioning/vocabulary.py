from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


def tokenize(text: str) -> list[str]:
    return [token.strip(".,!?;:()[]\"'").lower() for token in text.split() if token.strip()]


@dataclass
class Vocabulary:
    min_freq: int = 2
    pad_token: str = "<PAD>"
    start_token: str = "<START>"
    end_token: str = "<END>"
    unk_token: str = "<UNK>"
    word_to_idx: dict[str, int] = field(init=False)
    idx_to_word: dict[int, str] = field(init=False)

    def __post_init__(self) -> None:
        self.word_to_idx = {
            self.pad_token: 0,
            self.start_token: 1,
            self.end_token: 2,
            self.unk_token: 3,
        }
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

    def build(self, captions: list[str]) -> "Vocabulary":
        counts = Counter()
        for caption in captions:
            counts.update(tokenize(caption))
        for word, count in sorted(counts.items()):
            if count >= self.min_freq and word not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
        return self

    def encode(self, caption: str, max_length: int = 25) -> list[int]:
        tokens = [self.start_token] + tokenize(caption)[: max_length - 2] + [self.end_token]
        ids = [self.word_to_idx.get(token, self.word_to_idx[self.unk_token]) for token in tokens]
        if len(ids) < max_length:
            ids.extend([self.word_to_idx[self.pad_token]] * (max_length - len(ids)))
        return ids

    def decode(self, ids: list[int]) -> str:
        words = []
        for idx in ids:
            word = self.idx_to_word.get(int(idx), self.unk_token)
            if word == self.end_token:
                break
            if word not in {self.start_token, self.pad_token}:
                words.append(word)
        return " ".join(words)

    def __len__(self) -> int:
        return len(self.word_to_idx)
