from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class BaselineConfig:
    max_features: int = 50_000
    word_ngram_max: int = 2
    char_ngram_min: int = 3
    char_ngram_max: int = 5
    c: float = 2.0


def build_tfidf_baseline(config: BaselineConfig | None = None) -> Pipeline:
    config = config or BaselineConfig()
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=config.max_features,
                    ngram_range=(1, config.word_ngram_max),
                    analyzer="word",
                    sublinear_tf=True,
                    min_df=1,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=config.c,
                    max_iter=2_000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
