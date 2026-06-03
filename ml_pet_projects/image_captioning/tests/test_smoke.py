from image_captioning.pipeline import run_captioning_demo
from image_captioning.vocabulary import Vocabulary


def test_vocabulary_roundtrip() -> None:
    vocab = Vocabulary(min_freq=1).build(["a dog runs", "a dog jumps"])
    encoded = vocab.encode("a dog runs", max_length=8)
    assert vocab.decode(encoded) == "a dog runs"


def test_captioning_demo_smoke(tmp_path) -> None:
    result = run_captioning_demo(out_dir=tmp_path)
    assert result["metrics"]["vocab_size"] > 4
    assert result["metrics"]["demo_unigram_precision"] > 0.3
    assert (tmp_path / "metrics.csv").exists()
