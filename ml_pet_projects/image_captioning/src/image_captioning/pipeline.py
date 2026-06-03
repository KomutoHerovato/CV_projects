from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .data import make_demo_captions, train_val_split
from .metrics import caption_overlap_score
from .model import ResNetLSTMConfig, describe_model
from .vocabulary import Vocabulary


@dataclass(frozen=True)
class CaptioningConfig:
    min_word_freq: int = 1
    max_caption_length: int = 25
    val_fraction: float = 0.2
    model: ResNetLSTMConfig = ResNetLSTMConfig()


def _scene_template_baseline(train_df: pd.DataFrame, val_df: pd.DataFrame) -> list[str]:
    templates = train_df.groupby("scene")["caption"].agg(lambda x: x.value_counts().index[0]).to_dict()
    fallback = train_df["caption"].value_counts().index[0]
    return [templates.get(scene, fallback) for scene in val_df["scene"]]


def run_captioning_demo(
    data: pd.DataFrame | None = None,
    config: CaptioningConfig | None = None,
    out_dir: str | Path | None = None,
) -> dict[str, object]:
    config = config or CaptioningConfig()
    df = data.copy() if data is not None else make_demo_captions()
    train_df, val_df = train_val_split(df, val_fraction=config.val_fraction)

    vocabulary = Vocabulary(min_freq=config.min_word_freq).build(train_df["caption"].tolist())
    predictions = _scene_template_baseline(train_df, val_df)
    score = caption_overlap_score(val_df["caption"].tolist(), predictions)
    encoded_example = vocabulary.encode(train_df.iloc[0]["caption"], max_length=config.max_caption_length)

    artifacts = {
        "metrics": {
            "demo_unigram_precision": score,
            "vocab_size": len(vocabulary),
            "train_size": len(train_df),
            "val_size": len(val_df),
        },
        "model_setup": describe_model(config.model),
        "encoded_example": encoded_example,
        "decoded_example": vocabulary.decode(encoded_example),
        "predictions": pd.DataFrame(
            {
                "image_id": val_df["image_id"],
                "reference": val_df["caption"],
                "prediction": predictions,
            }
        ),
    }
    if out_dir is not None:
        save_captioning_artifacts(artifacts, out_dir)
    return artifacts


def save_captioning_artifacts(artifacts: dict[str, object], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([artifacts["metrics"]]).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame([artifacts["model_setup"]]).to_csv(out / "model_setup.csv", index=False)
    artifacts["predictions"].to_csv(out / "demo_predictions.csv", index=False)
