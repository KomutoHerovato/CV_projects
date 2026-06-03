from __future__ import annotations

import re

import pandas as pd


RU_STOPWORDS = set(
    "и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только "
    "ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть "
    "был него до вас нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть надо "
    "ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот "
    "того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем "
    "всех никогда можно при наконец два об другой хоть после над больше тот через эти нас про всего них "
    "какая много разве три эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им "
    "более всегда конечно всю между".split()
)
TOKEN_RE = re.compile(r"[а-яёa-z0-9]{2,}", flags=re.IGNORECASE)
MODALITY_ALIASES = {
    "text_clean": "text",
    "title_clean": "title",
    "topic_clean": "topic",
    "tags_clean": "tags",
}


def clean_text(value: object, max_chars: int = 6000) -> str:
    text = str(value).lower().replace("ё", "е")[:max_chars]
    tokens = TOKEN_RE.findall(text)
    return " ".join(token for token in tokens if token not in RU_STOPWORDS)


def _safe_series(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col in df.columns:
        return df[col].fillna(default).astype(str)
    return pd.Series([default] * len(df), index=df.index, dtype="object")


def build_document_text(
    df: pd.DataFrame,
    modalities: list[str] | tuple[str, ...] | str,
    max_chars: int = 6000,
) -> pd.Series:
    """Combine document modalities into one weighted text field."""
    if isinstance(modalities, str):
        modalities = [modalities]

    parts: list[pd.Series] = []
    for modality in modalities:
        base = MODALITY_ALIASES.get(modality, modality)
        if base == "text":
            parts.append(_safe_series(df, "text").map(lambda x: clean_text(x, max_chars=max_chars)))
        elif base == "title":
            title = _safe_series(df, "title").map(lambda x: clean_text(x, max_chars=max_chars))
            parts.append((title + " ") * 4)
        elif base == "topic":
            topic = _safe_series(df, "topic", "unknown").str.replace(r"\s+", "_", regex=True)
            parts.append((" topic_" + topic + " ") * 6)
        elif base == "tags":
            tags = _safe_series(df, "tags", "unknown").str.replace(r"\s+", "_", regex=True)
            parts.append((" tags_" + tags + " ") * 4)
        elif base in df.columns:
            parts.append(_safe_series(df, base).map(lambda x: clean_text(x, max_chars=max_chars)))

    if not parts:
        raise ValueError(f"No usable modalities found: {modalities}")
    out = parts[0].copy()
    for part in parts[1:]:
        out = out.str.cat(part, sep=" ")
    return out
