# CV_projects

Репозиторий с учебными проектами, pet-проектами и задачами, которые можно
использовать как портфолио для CV.

## Структура

- [projects/ml](./projects/ml) — оформленные ML pet-проекты с README, CLI,
  тестами, исходными ноутбуками и воспроизводимыми demo-режимами.
- [C++](./C++) — учебные и практические задачи на C++.
- [Flex_analysis](./Flex_analysis) — проект с лексическим анализатором на Flex.
- [Test_task](./Test_task) — тестовое задание с Docker, API и тестами.
- [ML_lessons](./ML_lessons) — учебные ноутбуки по машинному обучению.
- [CVision](./CVision), [MolGan](./MolGan), [cifar10_distillation_svd](./cifar10_distillation_svd) —
  отдельные ML/CV-эксперименты и ноутбуки.

## Витринные ML-проекты

| Проект | Кратко |
| --- | --- |
| [RiverSwim PSRL](./projects/ml/riverswim_psrl) | Байесовское RL: PSRL против Q-learning в среде RiverSwim. |
| [Lenta Year Classifier](./projects/ml/lenta_year_classifier) | Классификация года новости Lenta.ru по тексту и метаданным. |
| [MovieLens Recommender](./projects/ml/movielens_recommender) | Рекомендательная система: Popular, User-CF, Item-CF и matrix factorization. |
| [AI Text Detector](./projects/ml/ai_text_detector) | Детектор Human/Machine текста: быстрый TF-IDF baseline и BERT fine-tuning setup. |
| [Image Captioning](./projects/ml/image_captioning) | Генерация описаний изображений: ResNet152 encoder + LSTM decoder. |

Каждый проект внутри `projects/ml` можно запускать отдельно: в папке проекта
есть `requirements.txt`, `src/`, `tests/`, `notebooks/` и подробный README.
