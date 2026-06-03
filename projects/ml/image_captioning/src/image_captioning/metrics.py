from __future__ import annotations

from collections import Counter

from .vocabulary import tokenize


def unigram_precision(reference: str, prediction: str) -> float:
    ref_counts = Counter(tokenize(reference))
    pred_tokens = tokenize(prediction)
    if not pred_tokens:
        return 0.0
    hits = 0
    for token in pred_tokens:
        if ref_counts[token] > 0:
            hits += 1
            ref_counts[token] -= 1
    return hits / len(pred_tokens)


def caption_overlap_score(references: list[str], predictions: list[str]) -> float:
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have the same length")
    if not references:
        return 0.0
    return sum(unigram_precision(ref, pred) for ref, pred in zip(references, predictions)) / len(references)
