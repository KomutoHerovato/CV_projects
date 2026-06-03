# AI Text Detector

Проект по мотивам ноутбука `task7.ipynb`.

Задача — определить, написан текст человеком или сгенерирован моделью. В
ноутбуке использовался fine-tuning `bert-base-uncased` с заморозкой нижних
слоёв; в репозитории проект оформлен так, чтобы его можно было быстро проверить
локально: есть TF-IDF baseline, сохранение отчётов и отдельное описание
BERT-настройки для полного эксперимента.

## Почему проект выглядит полезно для портфолио

- Понятная современная NLP-задача: Human vs Machine text classification.
- Быстрый baseline, который запускается без GPU и скачивания BERT.
- BERT fine-tuning setup из исходного ноутбука:
  замороженные embeddings и первые 10 encoder layers, trainable верхние слои,
  dropout + linear classification head.
- Балансировка данных с конвенцией меток `0 = Human`, `1 = Machine`.
- Метрики и classification report сохраняются как артефакты.

## Быстрый запуск

```bash
cd /Users/damrorus/CV_projects/projects/ml/ai_text_detector
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m ai_text_detector.cli
PYTHONPATH=src pytest -q
```

## Запуск на своих данных

CLI принимает CSV с колонками:

- `text`
- `label` или `generated`, где `0 = Human`, `1 = Machine`

```bash
PYTHONPATH=src python -m ai_text_detector.cli --data data/ai_human_text_sample.csv
```

Результаты сохраняются в `outputs/ai_text_detector`:

- `metrics.csv`
- `bert_setup.csv`
- `classification_report.txt`

## Полный эксперимент из ноутбука

Исходный ноутбук использует HuggingFace dataset
`andythetechnerd03/AI-human-text`, токенизирует тексты через
`bert-base-uncased`, замораживает первые 10 BERT-слоёв и обучает бинарный
классификатор 3 эпохи. Этот путь лучше запускать с GPU, поэтому он не включён в
smoke-тесты.
