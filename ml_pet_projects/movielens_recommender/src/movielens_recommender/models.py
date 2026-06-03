from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

try:  # Torch is optional for import-time friendliness.
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover - depends on local environment
    torch = None
    nn = None
    optim = None
    DataLoader = None
    Dataset = object


class MostPopularModel:
    def __init__(self, min_votes: int = 50) -> None:
        self.min_votes = min_votes
        self.global_mean = 0.0
        self.item_scores: pd.Series | None = None

    def fit(self, train_df: pd.DataFrame) -> "MostPopularModel":
        stats = train_df.groupby("item_idx")["rating"].agg(["mean", "count"])
        self.global_mean = float(train_df["rating"].mean())
        stats["score"] = (
            stats["count"] / (stats["count"] + self.min_votes) * stats["mean"]
            + self.min_votes / (stats["count"] + self.min_votes) * self.global_mean
        )
        self.item_scores = stats["score"].sort_values(ascending=False)
        return self

    def predict(self, item_idx: int) -> float:
        if self.item_scores is None:
            raise RuntimeError("Model is not fitted")
        return float(self.item_scores.get(item_idx, self.global_mean))

    def recommend(self, seen: set[int] | None = None, k: int = 10) -> list[int]:
        if self.item_scores is None:
            raise RuntimeError("Model is not fitted")
        seen = seen or set()
        return [int(i) for i in self.item_scores.index if int(i) not in seen][:k]


class CollaborativeFiltering:
    @staticmethod
    def build_matrix(df: pd.DataFrame, n_users: int, n_items: int) -> np.ndarray:
        matrix = np.zeros((n_users, n_items), dtype=np.float32)
        for row in df.itertuples():
            matrix[int(row.user_idx), int(row.item_idx)] = float(row.rating)
        return matrix

    @staticmethod
    def similarities(matrix: np.ndarray, target: str = "user") -> np.ndarray:
        return cosine_similarity(matrix if target == "user" else matrix.T)

    @staticmethod
    def user_based_predict(user_idx: int, item_idx: int, matrix: np.ndarray, user_sim: np.ndarray, k: int = 20) -> float:
        rated = np.where(matrix[:, item_idx] > 0)[0]
        if len(rated) == 0:
            user_ratings = matrix[user_idx][matrix[user_idx] > 0]
            return float(user_ratings.mean()) if len(user_ratings) else 3.5
        sims = user_sim[user_idx, rated]
        top = rated[np.argsort(sims)[-k:]]
        weights = user_sim[user_idx, top]
        ratings = matrix[top, item_idx]
        if np.abs(weights).sum() <= 1e-12:
            return float(ratings.mean())
        return float(np.dot(weights, ratings) / np.abs(weights).sum())

    @staticmethod
    def item_based_predict(user_idx: int, item_idx: int, matrix: np.ndarray, item_sim: np.ndarray, k: int = 20) -> float:
        rated = np.where(matrix[user_idx] > 0)[0]
        if len(rated) == 0:
            return 3.5
        sims = item_sim[item_idx, rated]
        top = rated[np.argsort(sims)[-k:]]
        weights = item_sim[item_idx, top]
        ratings = matrix[user_idx, top]
        if np.abs(weights).sum() <= 1e-12:
            return float(ratings.mean())
        return float(np.dot(weights, ratings) / np.abs(weights).sum())


if torch is not None:

    class RatingDataset(Dataset):
        def __init__(self, df: pd.DataFrame) -> None:
            self.users = torch.tensor(df["user_idx"].values, dtype=torch.long)
            self.items = torch.tensor(df["item_idx"].values, dtype=torch.long)
            self.ratings = torch.tensor(df["rating"].values, dtype=torch.float32)

        def __len__(self) -> int:
            return len(self.ratings)

        def __getitem__(self, idx: int):
            return self.users[idx], self.items[idx], self.ratings[idx]


    class MatrixFactorization(nn.Module):
        def __init__(self, n_users: int, n_items: int, latent_dim: int = 50, global_mean: float = 3.5) -> None:
            super().__init__()
            self.user_factors = nn.Embedding(n_users, latent_dim)
            self.item_factors = nn.Embedding(n_items, latent_dim)
            self.user_bias = nn.Embedding(n_users, 1)
            self.item_bias = nn.Embedding(n_items, 1)
            self.register_buffer("global_mean", torch.tensor(float(global_mean)))
            nn.init.normal_(self.user_factors.weight, std=0.05)
            nn.init.normal_(self.item_factors.weight, std=0.05)
            nn.init.zeros_(self.user_bias.weight)
            nn.init.zeros_(self.item_bias.weight)

        def forward(self, users, items):
            dot = (self.user_factors(users) * self.item_factors(items)).sum(dim=1)
            bias = self.user_bias(users).squeeze(-1) + self.item_bias(items).squeeze(-1)
            return self.global_mean + dot + bias


else:

    class RatingDataset:  # type: ignore[no-redef]
        def __init__(self, df: pd.DataFrame) -> None:
            raise ImportError("Install torch to use RatingDataset")


    class MatrixFactorization:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("Install torch to use MatrixFactorization")


@dataclass(frozen=True)
class TrainingConfig:
    latent_dim: int = 32
    epochs: int = 5
    batch_size: int = 1024
    learning_rate: float = 5e-3
    weight_decay: float = 1e-4
    device: str = "cpu"


def train_matrix_factorization(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_users: int,
    n_items: int,
    config: TrainingConfig,
) -> tuple[object, list[dict[str, float]]]:
    if torch is None:
        raise ImportError("Install torch to train matrix factorization")
    device = torch.device(config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = MatrixFactorization(n_users, n_items, config.latent_dim, global_mean=float(train_df["rating"].mean())).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.MSELoss()
    train_loader = DataLoader(RatingDataset(train_df), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(RatingDataset(val_df), batch_size=config.batch_size, shuffle=False)
    history = []

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        for users, items, ratings in train_loader:
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)
            optimizer.zero_grad()
            loss = criterion(model(users, items), ratings)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(ratings)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for users, items, ratings in val_loader:
                users, items, ratings = users.to(device), items.to(device), ratings.to(device)
                val_loss += float(criterion(model(users, items), ratings).item()) * len(ratings)
        history.append(
            {
                "epoch": epoch + 1,
                "train_rmse": float(np.sqrt(train_loss / len(train_df))),
                "val_rmse": float(np.sqrt(val_loss / len(val_df))),
            }
        )
    return model, history
