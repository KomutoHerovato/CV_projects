# MovieLens Recommender

Проект по мотивам ноутбуков `untilted_2.ipynb` и `8.ipynb`.

Это мини-бенчмарк рекомендательных систем на explicit ratings в стиле
MovieLens: сравниваются Most Popular, User-CF, Item-CF и matrix factorization
с user/item bias. Проект считает не только ошибку предсказания рейтинга, но и
качество top-K рекомендаций.

## Что внутри

- Загрузчик MovieLens 1M и синтетический demo-датасет.
- Временное train/validation/test разбиение по каждому пользователю.
- Weighted Most Popular baseline.
- User-based и item-based collaborative filtering.
- PyTorch matrix factorization с bias-термами.
- Метрики: RMSE, Precision@K, Recall@K, MAP@K, NDCG@K, Coverage.

## Быстрый demo-запуск

```bash
cd /Users/damrorus/CV_projects/projects/ml/movielens_recommender
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m movielens_recommender.cli --synthetic --epochs 3 --latent-dim 16
```

## Полный запуск на MovieLens 1M

```bash
PYTHONPATH=src python -m movielens_recommender.cli \
  --data-dir data \
  --download \
  --epochs 10 \
  --latent-dim 50
```

Результаты сохраняются в `outputs/movielens_recommender`:

- `metrics.csv`
- `training_history.csv`
- `summary.txt`

## Тесты

```bash
PYTHONPATH=src pytest -q
```
