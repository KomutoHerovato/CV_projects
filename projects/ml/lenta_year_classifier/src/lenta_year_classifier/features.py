from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class TopicFeatureConfig:
    n_topics: int = 60
    n_word_svd: int = 250
    n_char_svd: int = 60
    max_word_features: int = 70_000
    max_char_features: int = 50_000
    min_df: int = 2
    max_df: float = 0.92
    random_state: int = 42


def top_words_from_model(model: object, vectorizer: TfidfVectorizer, n_words: int = 12) -> pd.DataFrame:
    terms = np.array(vectorizer.get_feature_names_out())
    if not hasattr(model, "components_"):
        return pd.DataFrame()
    rows = []
    for topic_idx, component in enumerate(model.components_):
        top = terms[np.argsort(component)[::-1][:n_words]]
        rows.append({"topic": topic_idx, "top_words": ", ".join(top)})
    return pd.DataFrame(rows)


def safe_normalize_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    denom = values.sum(axis=1, keepdims=True)
    return values / np.maximum(denom, eps)


def _bounded_components(requested: int, shape: tuple[int, int], *, svd: bool) -> int:
    limit = min(shape)
    if svd:
        limit -= 1
    return max(1, min(requested, limit))


def fit_nmf_safely(
    x_train,
    x_test,
    n_topics: int,
    random_state: int,
    reg_alpha: float = 0.0,
    l1_ratio: float = 0.0,
):
    n_components = _bounded_components(n_topics, x_train.shape, svd=False)
    alpha_candidates = [0.0] if reg_alpha <= 0 else [reg_alpha, reg_alpha / 10, reg_alpha / 100, 0.0]
    last_error: Exception | None = None
    for alpha in alpha_candidates:
        try:
            model = NMF(
                n_components=n_components,
                init="nndsvda",
                random_state=random_state,
                max_iter=500,
                solver="cd",
                beta_loss="frobenius",
                alpha_W=alpha,
                alpha_H=alpha,
                l1_ratio=l1_ratio,
                tol=1e-4,
            )
            theta_train = model.fit_transform(x_train)
            if np.max(theta_train) <= 1e-12:
                raise ValueError("NMF produced a near-zero document-topic matrix")
            theta_test = model.transform(x_test)
            model.used_alpha = alpha
            return theta_train, theta_test, model
        except Exception as exc:  # pragma: no cover - branch depends on sklearn numerics
            last_error = exc
    raise RuntimeError(f"NMF failed for all alpha candidates. Last error: {last_error!r}")


def build_topic_features(
    train_texts,
    test_texts,
    representation: str = "enhanced",
    config: TopicFeatureConfig | None = None,
    reg_alpha: float = 0.0,
    l1_ratio: float = 0.0,
):
    config = config or TopicFeatureConfig()
    word_vectorizer = TfidfVectorizer(
        max_features=config.max_word_features,
        min_df=config.min_df,
        max_df=config.max_df,
        ngram_range=(1, 2),
        sublinear_tf=True,
        norm="l2",
    )
    x_train_word = word_vectorizer.fit_transform(train_texts)
    x_test_word = word_vectorizer.transform(test_texts)

    parts_train: list[np.ndarray] = []
    parts_test: list[np.ndarray] = []
    models: dict[str, object] = {}
    vectorizers: dict[str, object] = {"word_tfidf": word_vectorizer}

    if representation in {"nmf", "nmf_svd", "enhanced"}:
        theta_train, theta_test, nmf_model = fit_nmf_safely(
            x_train_word,
            x_test_word,
            n_topics=config.n_topics,
            random_state=config.random_state,
            reg_alpha=reg_alpha,
            l1_ratio=l1_ratio,
        )
        parts_train.append(safe_normalize_rows(theta_train))
        parts_test.append(safe_normalize_rows(theta_test))
        models["nmf"] = nmf_model

    if representation in {"svd", "nmf_svd", "enhanced"}:
        n_components = _bounded_components(config.n_word_svd, x_train_word.shape, svd=True)
        svd_model = TruncatedSVD(n_components=n_components, random_state=config.random_state)
        parts_train.append(svd_model.fit_transform(x_train_word))
        parts_test.append(svd_model.transform(x_test_word))
        models["word_svd"] = svd_model

    if representation == "enhanced":
        char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=config.max_char_features,
            min_df=config.min_df,
            sublinear_tf=True,
        )
        x_train_char = char_vectorizer.fit_transform(train_texts)
        x_test_char = char_vectorizer.transform(test_texts)
        n_components = _bounded_components(config.n_char_svd, x_train_char.shape, svd=True)
        char_svd = TruncatedSVD(n_components=n_components, random_state=config.random_state)
        parts_train.append(char_svd.fit_transform(x_train_char))
        parts_test.append(char_svd.transform(x_test_char))
        models["char_svd"] = char_svd
        vectorizers["char_tfidf"] = char_vectorizer

    if not parts_train:
        raise ValueError(f"Unknown representation: {representation}")

    theta_train = np.hstack(parts_train)
    theta_test = np.hstack(parts_test)
    scaler = StandardScaler()
    theta_train = scaler.fit_transform(theta_train)
    theta_test = scaler.transform(theta_test)
    models["scaler"] = scaler
    return theta_train, theta_test, models, vectorizers


def build_sparse_tfidf(train_texts, test_texts, max_word_features: int = 120_000, max_char_features: int = 80_000):
    word_vec = TfidfVectorizer(
        max_features=max_word_features,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=max_char_features,
        min_df=2,
        sublinear_tf=True,
    )
    xw_train = word_vec.fit_transform(train_texts)
    xw_test = word_vec.transform(test_texts)
    xc_train = char_vec.fit_transform(train_texts)
    xc_test = char_vec.transform(test_texts)
    return hstack([xw_train, xc_train], format="csr"), hstack([xw_test, xc_test], format="csr"), {
        "word_tfidf": word_vec,
        "char_tfidf": char_vec,
    }
