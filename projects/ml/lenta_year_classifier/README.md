# Lenta Year Classifier

Проект по мотивам ноутбука `task9_lenta_year_bigartm_v4_final(1).ipynb`.

Цель — предсказать год публикации новости Lenta.ru по тексту и метаданным.
В ноутбуке была учебная постановка через тематические модели и BigARTM-like
эксперименты; в проекте оставлен устойчивый воспроизводимый путь на `sklearn`:
NMF, LSA/SVD, char-SVD, сравнение модальностей и сильный sparse TF-IDF baseline.

## Что внутри

- Балансировка выборки по годам из CSV/CSV.GZ датасета Lenta.ru.
- Модальности: `title`, `topic`, `tags`, `text`.
- Тематические признаки: `NMF`, `SVD`, `NMF + SVD`,
  `NMF + word-SVD + char-SVD`.
- Классификатор года на `LinearSVC`.
- Метрики: Accuracy, Macro-F1, Weighted-F1.
- Demo-режим без скачивания большого датасета.

## Быстрый demo-запуск

```bash
cd /Users/damrorus/CV_projects/projects/ml/lenta_year_classifier
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m lenta_year_classifier.cli --demo --min-year 2016 --max-year 2019 --sample-per-year 24 --n-topics 6 --n-word-svd 6 --n-char-svd 4
```

## Запуск на полном датасете

Скачайте `lenta-ru-news.csv.gz` из публичного релиза
`yutkin/Lenta.Ru-News-Dataset`, затем выполните:

```bash
PYTHONPATH=src python -m lenta_year_classifier.cli \
  --data /path/to/lenta-ru-news.csv.gz \
  --sample-per-year 700 \
  --min-year 2006 \
  --max-year 2019
```

Результаты сохраняются в `outputs/lenta_year_classifier`:

- `modal_results.csv`
- `sparse_tfidf_metrics.csv`
- `topic_words.csv`
- `classification_report.txt`

## Тесты

```bash
PYTHONPATH=src pytest -q
```

## Примечание про BigARTM

BigARTM оставлен как опциональная идея из ноутбука. На современных macOS/Python
окружениях его установка часто нестабильна, поэтому основной воспроизводимый
проект построен на тематических моделях `sklearn`.
