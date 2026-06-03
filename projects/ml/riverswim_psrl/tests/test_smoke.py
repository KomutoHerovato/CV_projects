from riverswim_psrl.experiment import ExperimentConfig, run_comparison


def test_riverswim_smoke() -> None:
    config = ExperimentConfig(episodes=4, trials=1, horizon=16, n_intermediate=3)
    history, summary = run_comparison(config)
    assert set(history["algorithm"]) == {"Q-learning", "PSRL"}
    assert len(history) == 8
    assert {"reward", "right_visits", "regret"}.issubset(summary.columns)
