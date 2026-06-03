from lenta_year_classifier.features import TopicFeatureConfig
from lenta_year_classifier.pipeline import LentaExperimentConfig, run_lenta_experiment


def test_lenta_demo_smoke(tmp_path) -> None:
    config = LentaExperimentConfig(
        sample_per_year=16,
        min_year=2016,
        max_year=2019,
        representation="enhanced",
        feature_config=TopicFeatureConfig(n_topics=4, n_word_svd=4, n_char_svd=3, min_df=1),
    )
    result = run_lenta_experiment(config=config, demo=True, out_dir=tmp_path)
    assert not result["modal_results"].empty
    assert (tmp_path / "modal_results.csv").exists()
    assert "macro_f1" in result["sparse_metrics"]
