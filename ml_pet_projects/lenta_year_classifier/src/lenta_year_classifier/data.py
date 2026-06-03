from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def get_year_column(df: pd.DataFrame) -> pd.Series:
    for col in ["date", "datetime", "time"]:
        if col in df.columns:
            return pd.to_datetime(df[col], errors="coerce").dt.year
    if "url" in df.columns:
        extracted = df["url"].astype(str).str.extract(r"/(19\d{2}|20\d{2})/")[0]
        return pd.to_numeric(extracted, errors="coerce")
    if "year" in df.columns:
        return pd.to_numeric(df["year"], errors="coerce")
    raise ValueError("Could not find date/time/year column or year inside url")


def _ensure_modal_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["title", "text", "topic", "tags"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    return df


def read_balanced_lenta(
    path: str | Path,
    sample_per_year: int = 700,
    min_year: int = 2006,
    max_year: int = 2019,
    chunksize: int = 50_000,
    random_state: int = 42,
    min_text_len: int = 250,
) -> pd.DataFrame:
    """Read a balanced year sample from a Lenta.ru CSV or CSV.GZ file."""
    path = Path(path)
    compression = "gzip" if path.suffix == ".gz" else None
    buckets: dict[int, list[pd.DataFrame]] = defaultdict(list)
    need_years = set(range(min_year, max_year + 1))

    reader = pd.read_csv(path, compression=compression, chunksize=chunksize)
    for chunk in reader:
        chunk = normalize_columns(chunk)
        chunk["year"] = get_year_column(chunk)
        chunk = chunk.dropna(subset=["year"])
        chunk["year"] = chunk["year"].astype(int)
        chunk = chunk[chunk["year"].isin(need_years)]
        chunk = _ensure_modal_columns(chunk)
        chunk = chunk[chunk["text"].str.len() >= min_text_len]

        for year, part in chunk.groupby("year"):
            current = sum(len(x) for x in buckets[year])
            if current < sample_per_year:
                take = min(sample_per_year - current, len(part))
                buckets[year].append(part.sample(take, random_state=random_state))

        if all(sum(len(x) for x in buckets[y]) >= sample_per_year for y in need_years):
            break

    frames = [pd.concat(buckets[y], ignore_index=True) for y in sorted(need_years) if buckets[y]]
    if not frames:
        raise RuntimeError("No documents were collected. Check dataset path and year range.")

    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df[["title", "text", "topic", "tags", "year"]]


def make_demo_lenta_dataset(years: Iterable[int] = range(2016, 2020), docs_per_year: int = 45) -> pd.DataFrame:
    """Create a tiny deterministic dataset for smoke tests and GitHub demos."""
    rng = np.random.default_rng(42)
    templates = {
        2016: ("economy", "markets oil currency export bank inflation budget"),
        2017: ("sports", "football championship olympic coach player match team"),
        2018: ("technology", "startup robot smartphone neural network platform data"),
        2019: ("culture", "film festival museum theatre actor director premiere"),
    }
    fallback_topics = [
        ("world", "election parliament government diplomacy minister reform"),
        ("science", "space satellite physics research laboratory experiment"),
    ]
    rows: list[dict[str, object]] = []
    for year in years:
        topic, vocab = templates.get(year, fallback_topics[year % len(fallback_topics)])
        tokens = vocab.split()
        for idx in range(docs_per_year):
            noise = rng.choice(["moscow", "expert", "report", "agency", "today"], size=8, replace=True)
            body = " ".join(tokens * 8 + noise.tolist())
            rows.append(
                {
                    "title": f"{topic.title()} update {idx}",
                    "text": body,
                    "topic": topic,
                    "tags": f"{topic} archive",
                    "year": year,
                }
            )
    return pd.DataFrame(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)
