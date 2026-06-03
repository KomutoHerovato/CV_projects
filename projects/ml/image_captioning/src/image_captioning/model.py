from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResNetLSTMConfig:
    encoder_name: str = "resnet152"
    encoder_feature_dim: int = 2048
    embed_dim: int = 256
    hidden_dim: int = 512
    num_lstm_layers: int = 1
    dropout: float = 0.5
    max_caption_length: int = 25
    fine_tune_encoder: bool = False


def describe_model(config: ResNetLSTMConfig | None = None) -> dict[str, object]:
    config = config or ResNetLSTMConfig()
    return {
        "encoder": f"{config.encoder_name} without classifier head",
        "encoder_feature_dim": config.encoder_feature_dim,
        "decoder": "Embedding + LSTM + Linear vocabulary projection",
        "embed_dim": config.embed_dim,
        "hidden_dim": config.hidden_dim,
        "num_lstm_layers": config.num_lstm_layers,
        "dropout": config.dropout,
        "max_caption_length": config.max_caption_length,
        "fine_tune_encoder": config.fine_tune_encoder,
    }
