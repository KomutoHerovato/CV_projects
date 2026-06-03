# MovieLens Recommender

Pet project based on `untilted_2.ipynb` and `8.ipynb`.

The project implements a recommendation benchmark on MovieLens-style explicit
ratings: Most Popular, User-CF, Item-CF, and matrix factorization with user/item
biases. It reports both rating prediction quality and top-K ranking quality.

## What is inside

- MovieLens 1M loader and downloader.
- Synthetic demo dataset for quick GitHub/local checks.
- Per-user temporal train/validation/test split.
- Most Popular weighted baseline.
- User-based and item-based collaborative filtering.
- PyTorch matrix factorization with biases.
- Metrics: RMSE, Precision@K, Recall@K, MAP@K, NDCG@K, Coverage.

## Quick demo

```bash
cd /Users/damrorus/CV_projects/ml_pet_projects/movielens_recommender
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m movielens_recommender.cli --synthetic --epochs 3 --latent-dim 16
```

## Full MovieLens 1M run

```bash
PYTHONPATH=src python -m movielens_recommender.cli \
  --data-dir data \
  --download \
  --epochs 10 \
  --latent-dim 50
```

Artifacts are saved to `outputs/movielens_recommender`:

- `metrics.csv`
- `training_history.csv`
- `summary.txt`

## Test

```bash
PYTHONPATH=src pytest -q
```
