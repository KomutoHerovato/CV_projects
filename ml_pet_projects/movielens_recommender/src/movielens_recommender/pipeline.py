from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .data import download_movielens_1m, filter_and_encode, load_movielens_1m, make_synthetic_ratings, temporal_split
from .metrics import evaluate_static_recommender, predict_mf, recommend_with_mf, rmse
from .models import CollaborativeFiltering, MostPopularModel, TrainingConfig, train_matrix_factorization


@dataclass(frozen=True)
class RecommenderConfig:
    min_user_ratings: int = 20
    min_item_ratings: int = 20
    latent_dim: int = 32
    epochs: int = 5
    k: int = 10
    device: str = "auto"
    download: bool = False


def load_recommender_data(data_dir: str | Path | None, synthetic: bool, config: RecommenderConfig):
    if synthetic:
        ratings, movies = make_synthetic_ratings()
        min_user, min_item = 5, 5
    else:
        if data_dir is None:
            raise ValueError("data_dir is required unless synthetic=True")
        root = Path(data_dir)
        if config.download:
            download_movielens_1m(root)
        ratings, movies = load_movielens_1m(root)
        min_user, min_item = config.min_user_ratings, config.min_item_ratings
    encoded, user_encoder, item_encoder = filter_and_encode(ratings, min_user, min_item)
    return encoded, movies, user_encoder, item_encoder


def run_recommender_experiment(
    data_dir: str | Path | None = None,
    synthetic: bool = False,
    config: RecommenderConfig | None = None,
    out_dir: str | Path | None = None,
) -> dict[str, object]:
    config = config or RecommenderConfig()
    ratings, movies, user_encoder, item_encoder = load_recommender_data(data_dir, synthetic, config)
    train_df, val_df, test_df = temporal_split(ratings)
    n_users = int(ratings["user_idx"].nunique())
    n_items = int(ratings["item_idx"].nunique())

    popular = MostPopularModel(min_votes=50).fit(train_df)
    popular_rmse = rmse(test_df["rating"], [popular.predict(int(i)) for i in test_df["item_idx"]])
    popular_ranking = evaluate_static_recommender(
        train_df,
        test_df,
        lambda user, seen, k: popular.recommend(seen, k),
        k=config.k,
    )

    matrix = CollaborativeFiltering.build_matrix(train_df, n_users, n_items)
    user_sim = CollaborativeFiltering.similarities(matrix, "user")
    item_sim = CollaborativeFiltering.similarities(matrix, "item")
    eval_sample = test_df.head(min(len(test_df), 2_000))
    user_cf_pred = [
        CollaborativeFiltering.user_based_predict(int(r.user_idx), int(r.item_idx), matrix, user_sim)
        for r in eval_sample.itertuples()
    ]
    item_cf_pred = [
        CollaborativeFiltering.item_based_predict(int(r.user_idx), int(r.item_idx), matrix, item_sim)
        for r in eval_sample.itertuples()
    ]

    metrics_rows = [
        {"model": "Most Popular", "rmse": popular_rmse, **popular_ranking},
        {"model": "User-CF", "rmse": rmse(eval_sample["rating"], user_cf_pred)},
        {"model": "Item-CF", "rmse": rmse(eval_sample["rating"], item_cf_pred)},
    ]

    training_history = pd.DataFrame()
    mf_status = "trained"
    try:
        training_config = TrainingConfig(
            latent_dim=config.latent_dim,
            epochs=config.epochs,
            device=config.device,
        )
        mf_model, history = train_matrix_factorization(train_df, val_df, n_users, n_items, training_config)
        training_history = pd.DataFrame(history)
        mf_pred = predict_mf(mf_model, test_df, device=config.device)
        mf_ranking = evaluate_static_recommender(
            train_df,
            test_df,
            lambda user, seen, k: recommend_with_mf(mf_model, user, seen, n_items, k=k, device=config.device),
            k=config.k,
        )
        metrics_rows.append({"model": "Matrix Factorization", "rmse": rmse(test_df["rating"], mf_pred), **mf_ranking})
    except Exception as exc:  # pragma: no cover - useful when torch is not installed
        mf_status = f"skipped: {exc}"

    metrics = pd.DataFrame(metrics_rows)
    artifacts = {
        "metrics": metrics,
        "training_history": training_history,
        "mf_status": mf_status,
        "n_users": n_users,
        "n_items": n_items,
        "n_ratings": len(ratings),
        "movies": movies,
        "user_encoder": user_encoder,
        "item_encoder": item_encoder,
    }
    if out_dir is not None:
        save_recommender_artifacts(artifacts, out_dir)
    return artifacts


def save_recommender_artifacts(artifacts: dict[str, object], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifacts["metrics"].to_csv(out / "metrics.csv", index=False)
    artifacts["training_history"].to_csv(out / "training_history.csv", index=False)
    (out / "summary.txt").write_text(
        "\n".join(
            [
                f"users={artifacts['n_users']}",
                f"items={artifacts['n_items']}",
                f"ratings={artifacts['n_ratings']}",
                f"matrix_factorization={artifacts['mf_status']}",
            ]
        ),
        encoding="utf-8",
    )
