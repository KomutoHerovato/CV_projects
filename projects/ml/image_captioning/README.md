# Image Captioning

Проект по мотивам ноутбука `task6(3).ipynb`.

Исходный ноутбук обучает модель генерации описаний изображений на подвыборке
COCO Captions: замороженный ResNet152 извлекает признаки изображения, а LSTM
decoder генерирует caption по токенам. В проекте эта работа оформлена как
самостоятельный модуль, готовый для GitHub: с утилитами для словаря и данных, описанием
архитектуры, smoke-тестами и demo-артефактами.

## Почему проект выглядит полезно для портфолио

- Классическая multimodal deep learning задача.
- Понятная архитектура: `ResNet152 -> projection -> LSTM decoder -> vocabulary`.
- COCO-style preprocessing: captions, vocabulary, max length, специальные
  токены `<PAD>`, `<START>`, `<END>`, `<UNK>`.
- TensorBoard/checkpoint workflow сохранён в описании из ноутбука.
- Лёгкий demo pipeline проверяет код без скачивания COCO.

## Быстрый запуск

```bash
cd /Users/damrorus/CV_projects/projects/ml/image_captioning
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m image_captioning.cli
PYTHONPATH=src pytest -q
```

Результаты сохраняются в `outputs/image_captioning`:

- `metrics.csv`
- `model_setup.csv`
- `demo_predictions.csv`

## Полный эксперимент из ноутбука

Для полного запуска на COCO:

1. Скачать COCO 2017 captions и подвыборку изображений.
2. Построить словарь с `<PAD>`, `<START>`, `<END>`, `<UNK>`.
3. Применить ImageNet preprocessing для входов ResNet152.
4. Заморозить ResNet backbone и обучать projection layer + LSTM decoder.
5. Логировать loss/perplexity и примеры генерации в TensorBoard.

Исходный ноутбук сохранён в `notebooks/task6_image_captioning.ipynb`.
