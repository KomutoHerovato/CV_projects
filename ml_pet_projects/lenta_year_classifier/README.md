# Lenta Year Classifier

Pet project based on `task9_lenta_year_bigartm_v4_final(1).ipynb`.

The model predicts the publication year of a Lenta.ru article from text and
metadata. The original notebook experimented with topic representations and a
BigARTM-compatible task framing. This project keeps the robust, reproducible
path: sklearn NMF, LSA/SVD, char-SVD, modality ablations, and a strong sparse
TF-IDF + LinearSVC baseline.

## What is inside

- Balanced loading from the public Lenta.ru CSV/CSV.GZ dataset.
- Modalities: `title`, `topic`, `tags`, `text`.
- Topic features: `NMF`, `SVD`, `NMF + SVD`, `NMF + word-SVD + char-SVD`.
- Year classifier with Accuracy, Macro-F1, Weighted-F1.
- Demo mode that runs without downloading the large dataset.
- Saved reports for GitHub screenshots and project writeups.

## Quick demo

```bash
cd /Users/damrorus/CV_projects/ml_pet_projects/lenta_year_classifier
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m lenta_year_classifier.cli --demo --min-year 2016 --max-year 2019 --sample-per-year 24 --n-topics 6 --n-word-svd 6 --n-char-svd 4
```

## Full dataset run

Download `lenta-ru-news.csv.gz` from the public
`yutkin/Lenta.Ru-News-Dataset` release, then run:

```bash
PYTHONPATH=src python -m lenta_year_classifier.cli \
  --data /path/to/lenta-ru-news.csv.gz \
  --sample-per-year 700 \
  --min-year 2006 \
  --max-year 2019
```

Artifacts are saved to `outputs/lenta_year_classifier`:

- `modal_results.csv`
- `sparse_tfidf_metrics.csv`
- `topic_words.csv`
- `classification_report.txt`

## Test

```bash
PYTHONPATH=src pytest -q
```

## BigARTM note

BigARTM is intentionally optional. Its Python wheels can be fragile on modern
Python/macOS setups, so the reproducible default uses sklearn topic models while
preserving the core experiment design from the notebook.
