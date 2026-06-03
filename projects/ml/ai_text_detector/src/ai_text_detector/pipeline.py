from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from .baseline import BaselineConfig, build_tfidf_baseline
from .bert import BertFineTuneConfig, describe_bert_setup
from .data import make_demo_dataset


@dataclass(frozen=True)
class DetectorConfig:
    test_size: float = 0.25
    random_state: int = 42
    baseline: BaselineConfig = BaselineConfig()
    bert: BertFineTuneConfig = BertFineTuneConfig()


def run_detector_experiment(
    data: pd.DataFrame | None = None,
    config: DetectorConfig | None = None,
    out_dir: str | Path | None = None,
) -> dict[str, object]:
    config = config or DetectorConfig()
    df = data.copy() if data is not None else make_demo_dataset()
    df = df.rename(columns={"generated": "label"})
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)

    train_df, test_df = train_test_split(
        df,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=df["label"],
    )
    model = build_tfidf_baseline(config.baseline)
    model.fit(train_df["text"], train_df["label"])
    pred = model.predict(test_df["text"])
    metrics = {
        "accuracy": accuracy_score(test_df["label"], pred),
        "macro_f1": f1_score(test_df["label"], pred, average="macro"),
        "weighted_f1": f1_score(test_df["label"], pred, average="weighted"),
    }
    report = classification_report(
        test_df["label"],
        pred,
        target_names=["Human", "Machine"],
        zero_division=0,
    )
    artifacts = {
        "metrics": metrics,
        "classification_report": report,
        "bert_setup": describe_bert_setup(config.bert),
        "model": model,
        "train_size": len(train_df),
        "test_size": len(test_df),
    }
    if out_dir is not None:
        save_detector_artifacts(artifacts, out_dir)
    return artifacts


def save_detector_artifacts(artifacts: dict[str, object], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([artifacts["metrics"]]).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame([artifacts["bert_setup"]]).to_csv(out / "bert_setup.csv", index=False)
    (out / "classification_report.txt").write_text(
        str(artifacts["classification_report"]),
        encoding="utf-8",
    )
