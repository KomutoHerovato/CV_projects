from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CaptionSample:
    image_id: str
    caption: str


def make_demo_captions(samples_per_scene: int = 20, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    scenes = {
        "dog": [
            "a dog runs through grass",
            "a brown dog plays outside",
            "a dog jumps near a field",
        ],
        "kitchen": [
            "a person prepares food in a kitchen",
            "fresh vegetables sit on a table",
            "a kitchen counter holds bowls and plates",
        ],
        "street": [
            "people walk along a busy street",
            "cars move through the city road",
            "a cyclist rides past buildings",
        ],
        "beach": [
            "waves reach the sandy beach",
            "people stand near the ocean",
            "a child plays beside the sea",
        ],
    }
    rows = []
    for scene, captions in scenes.items():
        for idx in range(samples_per_scene):
            rows.append(
                {
                    "image_id": f"{scene}_{idx:03d}.jpg",
                    "scene": scene,
                    "caption": rng.choice(captions),
                }
            )
    return pd.DataFrame(rows)


def train_val_split(df: pd.DataFrame, val_fraction: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    cut = max(1, int(len(shuffled) * (1 - val_fraction)))
    return shuffled.iloc[:cut].reset_index(drop=True), shuffled.iloc[cut:].reset_index(drop=True)
