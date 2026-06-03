from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def dcg_at_k(relevance: list[int], k: int) -> float:
    values = np.asarray(relevance[:k], dtype=float)
    if len(values) == 0:
        return 0.0
    return float(np.sum(values / np.log2(np.arange(2, len(values) + 2))))


def ndcg_at_k(relevance: list[int], k: int) -> float:
    ideal = dcg_at_k(sorted(relevance, reverse=True), k)
    return 0.0 if ideal == 0 else dcg_at_k(relevance, k) / ideal


def ranking_metrics(recommended: list[int], relevant: set[int], k: int = 10) -> dict[str, float]:
    recommended = recommended[:k]
    hits = [1 if item in relevant else 0 for item in recommended]
    precision = sum(hits) / k
    recall = sum(hits) / max(1, len(relevant))
    ap = 0.0
    hit_count = 0
    for idx, hit in enumerate(hits, start=1):
        if hit:
            hit_count += 1
            ap += hit_count / idx
    ap /= max(1, min(len(relevant), k))
    return {"precision": precision, "recall": recall, "map": ap, "ndcg": ndcg_at_k(hits, k)}


def evaluate_static_recommender(train_df: pd.DataFrame, test_df: pd.DataFrame, recommend_fn, k: int = 10) -> dict[str, float]:
    train_items = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    test_items = test_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    rows = []
    all_recommended = set()
    for user_idx, relevant in test_items.items():
        recs = recommend_fn(int(user_idx), train_items.get(user_idx, set()), k)
        all_recommended.update(recs)
        rows.append(ranking_metrics(recs, relevant, k=k))
    metrics = pd.DataFrame(rows).mean().to_dict()
    metrics["coverage"] = len(all_recommended) / max(1, train_df["item_idx"].nunique())
    return {key: float(value) for key, value in metrics.items()}


def recommend_with_mf(model, user_idx: int, seen: set[int], n_items: int, k: int = 10, device: str = "cpu") -> list[int]:
    if torch is None:
        raise ImportError("Install torch to evaluate matrix factorization")
    model.eval()
    dev = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    items = torch.arange(n_items, dtype=torch.long, device=dev)
    users = torch.full((n_items,), int(user_idx), dtype=torch.long, device=dev)
    with torch.no_grad():
        scores = model(users, items).detach().cpu().numpy()
    if seen:
        scores[list(seen)] = -np.inf
    return [int(i) for i in np.argsort(scores)[::-1][:k]]


def predict_mf(model, df: pd.DataFrame, device: str = "cpu") -> np.ndarray:
    if torch is None:
        raise ImportError("Install torch to evaluate matrix factorization")
    dev = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    users = torch.tensor(df["user_idx"].values, dtype=torch.long, device=dev)
    items = torch.tensor(df["item_idx"].values, dtype=torch.long, device=dev)
    model.eval()
    with torch.no_grad():
        return model(users, items).detach().cpu().numpy()
