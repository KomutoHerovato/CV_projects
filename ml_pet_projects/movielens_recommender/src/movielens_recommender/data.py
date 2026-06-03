from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def download_movielens_1m(out_dir: str | Path, url: str = MOVIELENS_1M_URL) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    marker = out / "ml-1m"
    if marker.exists():
        return marker
    with urllib.request.urlopen(url) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(out)
    return marker


def load_movielens_1m(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(data_dir)
    ratings_path = root / "ml-1m" / "ratings.dat"
    movies_path = root / "ml-1m" / "movies.dat"
    if not ratings_path.exists() or not movies_path.exists():
        raise FileNotFoundError("Expected ml-1m/ratings.dat and ml-1m/movies.dat")
    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        names=["user_id", "movie_id", "rating", "timestamp"],
        engine="python",
        encoding="latin-1",
    )
    movies = pd.read_csv(
        movies_path,
        sep="::",
        names=["movie_id", "title", "genres"],
        engine="python",
        encoding="latin-1",
    )
    return ratings, movies


def make_synthetic_ratings(n_users: int = 80, n_items: int = 120, ratings_per_user: int = 25) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    user_factors = rng.normal(size=(n_users, 8))
    item_factors = rng.normal(size=(n_items, 8))
    rows = []
    timestamp = 1_000_000_000
    for user_id in range(n_users):
        items = rng.choice(n_items, size=min(ratings_per_user, n_items), replace=False)
        for item_id in items:
            score = 3.5 + 0.45 * np.dot(user_factors[user_id], item_factors[item_id]) / 8
            rating = float(np.clip(np.round(score + rng.normal(scale=0.35)), 1, 5))
            rows.append(
                {
                    "user_id": user_id,
                    "movie_id": item_id,
                    "rating": rating,
                    "timestamp": timestamp,
                }
            )
            timestamp += int(rng.integers(1, 500))
    ratings = pd.DataFrame(rows)
    movies = pd.DataFrame(
        {
            "movie_id": np.arange(n_items),
            "title": [f"Demo Movie {i}" for i in range(n_items)],
            "genres": [rng.choice(["Drama", "Comedy", "Sci-Fi", "Action"]) for _ in range(n_items)],
        }
    )
    return ratings, movies


def filter_and_encode(
    ratings: pd.DataFrame,
    min_user_ratings: int = 20,
    min_item_ratings: int = 20,
) -> tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    df = ratings.copy()
    active_users = df["user_id"].value_counts()
    df = df[df["user_id"].isin(active_users[active_users >= min_user_ratings].index)]
    active_items = df["movie_id"].value_counts()
    df = df[df["movie_id"].isin(active_items[active_items >= min_item_ratings].index)]

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    df["user_idx"] = user_encoder.fit_transform(df["user_id"])
    df["item_idx"] = item_encoder.fit_transform(df["movie_id"])
    return df.reset_index(drop=True), user_encoder, item_encoder


def temporal_split(
    ratings: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split each user's history by timestamp into train/val/test."""
    train_parts = []
    val_parts = []
    test_parts = []
    for _, group in ratings.sort_values("timestamp").groupby("user_idx"):
        n = len(group)
        n_test = max(1, int(round(n * test_size)))
        n_val = max(1, int(round(n * val_size)))
        if n - n_test - n_val < 1:
            train_parts.append(group.iloc[: max(1, n - 2)])
            val_parts.append(group.iloc[max(1, n - 2) : max(1, n - 1)])
            test_parts.append(group.iloc[max(1, n - 1) :])
        else:
            train_parts.append(group.iloc[: n - n_test - n_val])
            val_parts.append(group.iloc[n - n_test - n_val : n - n_test])
            test_parts.append(group.iloc[n - n_test :])
    return (
        pd.concat(train_parts).reset_index(drop=True),
        pd.concat(val_parts).reset_index(drop=True),
        pd.concat(test_parts).reset_index(drop=True),
    )
