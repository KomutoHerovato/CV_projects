from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TextSample:
    text: str
    label: int


def make_demo_dataset(samples_per_class: int = 80, seed: int = 42) -> pd.DataFrame:
    """Create a deterministic toy dataset for smoke tests and local demos.

    Label convention follows the notebook: 0 = human, 1 = machine.
    """
    rng = random.Random(seed)
    human_templates = [
        "I changed my mind twice while writing this because the story felt too neat.",
        "The meeting ran late, so I grabbed coffee and rewrote the paragraph on the train.",
        "What surprised me most was not the result, but how ordinary the day around it felt.",
        "I left a note in the margin because that sentence sounded true but unfinished.",
    ]
    machine_templates = [
        "In conclusion, this topic demonstrates a multifaceted relationship between innovation and society.",
        "The generated analysis provides a comprehensive overview with clear implications for future development.",
        "This response highlights key considerations, practical benefits, and potential limitations in a balanced manner.",
        "Overall, the system can optimize workflows, enhance productivity, and support data-driven decisions.",
    ]
    rows: list[dict[str, object]] = []
    for idx in range(samples_per_class):
        human = rng.choice(human_templates)
        machine = rng.choice(machine_templates)
        rows.append({"text": f"{human} Example {idx}: {rng.choice(human_templates)}", "label": 0})
        rows.append({"text": f"{machine} Example {idx}: {rng.choice(machine_templates)}", "label": 1})
    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def prepare_balanced_dataframe(
    rows: list[dict[str, object]] | pd.DataFrame,
    text_col: str = "text",
    label_col: str = "generated",
    max_train: int = 4_000,
    max_val: int = 2_000,
    min_words: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Balance a HuggingFace-style AI-human dataset into a simple DataFrame."""
    df = pd.DataFrame(rows).copy()
    if text_col not in df.columns:
        raise ValueError(f"Missing text column: {text_col}")
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df[df["text"].str.split().str.len() >= min_words]

    balanced_parts = []
    target = min(df["label"].value_counts().min(), (max_train + max_val) // 2)
    for label, part in df.groupby("label"):
        balanced_parts.append(part.sample(n=target, random_state=seed))
    return pd.concat(balanced_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
