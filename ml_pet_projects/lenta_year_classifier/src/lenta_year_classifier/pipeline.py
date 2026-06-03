from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from .data import make_demo_lenta_dataset, read_balanced_lenta
from .features import TopicFeatureConfig, build_sparse_tfidf, build_topic_features, top_words_from_model
from .text import build_document_text


@dataclass(frozen=True)
class LentaExperimentConfig:
    sample_per_year: int = 700
    min_year: int = 2006
    max_year: int = 2019
    test_size: float = 0.25
    representation: str = "enhanced"
    random_state: int = 42
    feature_config: TopicFeatureConfig = TopicFeatureConfig()


def _metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def _fit_classifier(x_train, x_test, y_train, y_test) -> tuple[LinearSVC, object, dict[str, float]]:
    clf = LinearSVC(C=0.8, class_weight="balanced", random_state=42, max_iter=6000)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return clf, pred, _metrics(y_test, pred)


def load_dataset(path: str | Path | None, config: LentaExperimentConfig, demo: bool = False) -> pd.DataFrame:
    if demo or path is None:
        years = range(config.min_year, config.max_year + 1)
        return make_demo_lenta_dataset(years=years, docs_per_year=max(12, min(config.sample_per_year, 60)))
    return read_balanced_lenta(
        path,
        sample_per_year=config.sample_per_year,
        min_year=config.min_year,
        max_year=config.max_year,
        random_state=config.random_state,
    )


def run_lenta_experiment(
    path: str | Path | None = None,
    config: LentaExperimentConfig | None = None,
    demo: bool = False,
    out_dir: str | Path | None = None,
) -> dict[str, object]:
    config = config or LentaExperimentConfig()
    df = load_dataset(path, config, demo=demo)

    stratify = df["year"] if df["year"].value_counts().min() >= 2 else None
    train_df, test_df = train_test_split(
        df,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["year"])
    y_test = label_encoder.transform(test_df["year"])

    modal_experiments = {
        "text": ["text"],
        "title_text": ["title", "text"],
        "title_topic_text": ["title", "topic", "text"],
        "title_topic_tags_text": ["title", "topic", "tags", "text"],
    }
    modal_rows: list[dict[str, object]] = []
    saved: dict[str, dict[str, object]] = {}

    for name, modalities in modal_experiments.items():
        train_text = build_document_text(train_df, modalities)
        test_text = build_document_text(test_df, modalities)
        x_train, x_test, models, vectorizers = build_topic_features(
            train_text,
            test_text,
            representation=config.representation,
            config=config.feature_config,
        )
        clf, pred, metrics = _fit_classifier(x_train, x_test, y_train, y_test)
        row = {
            "experiment": name,
            "modalities": "+".join(modalities),
            "representation": config.representation,
            **metrics,
        }
        modal_rows.append(row)
        saved[name] = {
            "modalities": modalities,
            "models": models,
            "vectorizers": vectorizers,
            "classifier": clf,
            "prediction": pred,
            "metrics": metrics,
        }

    modal_results = pd.DataFrame(modal_rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    best_name = str(modal_results.iloc[0]["experiment"])
    best_modalities = saved[best_name]["modalities"]

    train_text = build_document_text(train_df, best_modalities)
    test_text = build_document_text(test_df, best_modalities)
    x_sparse_train, x_sparse_test, sparse_vectorizers = build_sparse_tfidf(train_text, test_text)
    sparse_clf, sparse_pred, sparse_metrics = _fit_classifier(x_sparse_train, x_sparse_test, y_train, y_test)

    report = classification_report(
        y_test,
        saved[best_name]["prediction"],
        target_names=[str(x) for x in label_encoder.classes_],
        zero_division=0,
    )
    topic_words = pd.DataFrame()
    nmf_model = saved[best_name]["models"].get("nmf")
    word_vec = saved[best_name]["vectorizers"].get("word_tfidf")
    if nmf_model is not None and word_vec is not None:
        topic_words = top_words_from_model(nmf_model, word_vec, n_words=12)

    artifacts = {
        "data": df,
        "modal_results": modal_results,
        "sparse_metrics": sparse_metrics,
        "classification_report": report,
        "topic_words": topic_words,
        "best_experiment": best_name,
        "label_encoder": label_encoder,
        "sparse_classifier": sparse_clf,
        "sparse_vectorizers": sparse_vectorizers,
        "sparse_prediction": sparse_pred,
    }

    if out_dir is not None:
        save_lenta_artifacts(artifacts, out_dir)
    return artifacts


def save_lenta_artifacts(artifacts: dict[str, object], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifacts["modal_results"].to_csv(out / "modal_results.csv", index=False)
    artifacts["topic_words"].to_csv(out / "topic_words.csv", index=False)
    pd.DataFrame([artifacts["sparse_metrics"]]).to_csv(out / "sparse_tfidf_metrics.csv", index=False)
    (out / "classification_report.txt").write_text(str(artifacts["classification_report"]), encoding="utf-8")
