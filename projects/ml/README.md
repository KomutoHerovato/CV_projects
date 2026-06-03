# ML pet-проекты

Эта папка превращает последние ноутбуки в полноценные проекты для GitHub и CV.
Исходные ноутбуки сохранены в `notebooks/`, а основной воспроизводимый код
вынесен в Python-пакеты внутри `src/`.

## Проекты

| Проект | Исходные ноутбуки | Что показывает |
| --- | --- | --- |
| [RiverSwim PSRL](./riverswim_psrl) | `river_swim_psrl_colab(1).ipynb` | Байесовское reinforcement learning, posterior sampling, динамическое программирование, Q-learning baseline. |
| [Lenta Year Classifier](./lenta_year_classifier) | `task9_lenta_year_bigartm_v4_final(1).ipynb` | NLP для русских новостей, тематические признаки, сравнение модальностей, классификация года, TF-IDF baseline. |
| [MovieLens Recommender](./movielens_recommender) | `untilted_2.ipynb`, `8.ipynb` | Рекомендательные системы, collaborative filtering, matrix factorization, ranking-метрики. |
| [AI Text Detector](./ai_text_detector) | `task7.ipynb` | Классификация Human/Machine текста, BERT fine-tuning setup, быстрый TF-IDF baseline. |
| [Image Captioning](./image_captioning) | `task6(3).ipynb` | Multimodal learning, ResNet152 encoder, LSTM decoder, COCO-style preprocessing. |

## Быстрая проверка

Каждый проект проверяется независимо:

```bash
cd /Users/damrorus/CV_projects/projects/ml/riverswim_psrl
PYTHONPATH=src python -m riverswim_psrl.cli --episodes 20 --trials 1
PYTHONPATH=src pytest -q
```

```bash
cd /Users/damrorus/CV_projects/projects/ml/lenta_year_classifier
PYTHONPATH=src python -m lenta_year_classifier.cli --demo --min-year 2016 --max-year 2019 --sample-per-year 20 --n-topics 4 --n-word-svd 4 --n-char-svd 3
PYTHONPATH=src pytest -q
```

```bash
cd /Users/damrorus/CV_projects/projects/ml/movielens_recommender
PYTHONPATH=src python -m movielens_recommender.cli --synthetic --epochs 1 --latent-dim 4
PYTHONPATH=src pytest -q
```

```bash
cd /Users/damrorus/CV_projects/projects/ml/ai_text_detector
PYTHONPATH=src python -m ai_text_detector.cli
PYTHONPATH=src pytest -q
```

```bash
cd /Users/damrorus/CV_projects/projects/ml/image_captioning
PYTHONPATH=src python -m image_captioning.cli
PYTHONPATH=src pytest -q
```

## Принцип оформления

- `src/` — код проекта, который можно импортировать и переиспользовать.
- `tests/` — быстрые smoke-тесты без тяжёлых скачиваний и GPU.
- `notebooks/` — исходные ноутбуки как исследовательская версия проекта.
- `outputs/` — локальные результаты запусков; папка игнорируется Git.
