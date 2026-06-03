from ai_text_detector.data import make_demo_dataset
from ai_text_detector.pipeline import run_detector_experiment


def test_ai_text_detector_smoke(tmp_path) -> None:
    df = make_demo_dataset(samples_per_class=20)
    result = run_detector_experiment(df, out_dir=tmp_path)
    assert result["metrics"]["macro_f1"] >= 0.8
    assert (tmp_path / "metrics.csv").exists()
    assert result["bert_setup"]["model_name"] == "bert-base-uncased"
