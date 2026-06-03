from movielens_recommender.pipeline import RecommenderConfig, run_recommender_experiment


def test_recommender_synthetic_smoke(tmp_path) -> None:
    config = RecommenderConfig(latent_dim=4, epochs=1, k=5, device="cpu")
    result = run_recommender_experiment(synthetic=True, config=config, out_dir=tmp_path)
    assert not result["metrics"].empty
    assert result["n_users"] > 0
    assert (tmp_path / "metrics.csv").exists()
