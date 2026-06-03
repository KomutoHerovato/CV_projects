from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BertFineTuneConfig:
    model_name: str = "bert-base-uncased"
    max_length: int = 256
    batch_size: int = 32
    epochs: int = 3
    learning_rate: float = 2e-5
    freeze_layers: int = 10


def describe_bert_setup(config: BertFineTuneConfig | None = None) -> dict[str, object]:
    """Return the BERT setup used by the original notebook.

    The actual fine-tuning path intentionally lives outside smoke tests because
    it downloads a transformer checkpoint and benefits from GPU. The project
    still documents the architecture in importable code for reproducibility.
    """
    config = config or BertFineTuneConfig()
    trainable_layers = max(0, 12 - config.freeze_layers)
    return {
        "model_name": config.model_name,
        "max_length": config.max_length,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "frozen_encoder_layers": config.freeze_layers,
        "trainable_encoder_layers": trainable_layers,
        "head": "Dropout + Linear classifier for Human/Machine labels",
    }
