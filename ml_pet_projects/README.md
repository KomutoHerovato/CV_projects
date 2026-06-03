# ML Pet Projects

This folder turns the latest notebooks from Downloads into three clean,
reproducible portfolio projects. The original notebooks are kept under each
project's `notebooks/` directory for reference; the production-style entrypoint
is the Python package in `src/`.

## Projects

| Project | Source notebooks | What it demonstrates |
| --- | --- | --- |
| [RiverSwim PSRL](./riverswim_psrl) | `river_swim_psrl_colab(1).ipynb` | Bayesian reinforcement learning, posterior sampling, finite-horizon dynamic programming, Q-learning baseline |
| [Lenta Year Classifier](./lenta_year_classifier) | `task9_lenta_year_bigartm_v4_final(1).ipynb` | Russian NLP, topic features, modality ablations, year classification, TF-IDF baseline |
| [MovieLens Recommender](./movielens_recommender) | `untilted_2.ipynb`, `8.ipynb` | Recommendation systems, collaborative filtering, matrix factorization, ranking metrics |
| [AI Text Detector](./ai_text_detector) | `task7.ipynb` | Human-vs-machine text classification, BERT fine-tuning setup, robust TF-IDF baseline |
| [Image Captioning](./image_captioning) | `task6(3).ipynb` | Multimodal learning, ResNet152 encoder, LSTM decoder, COCO-style caption preprocessing |

## Quick Smoke Run

Each project can be tested independently:

```bash
cd /Users/damrorus/CV_projects/ml_pet_projects/riverswim_psrl
PYTHONPATH=src python -m riverswim_psrl.cli --episodes 20 --trials 1
PYTHONPATH=src pytest -q
```

```bash
cd /Users/damrorus/CV_projects/ml_pet_projects/lenta_year_classifier
PYTHONPATH=src python -m lenta_year_classifier.cli --demo --min-year 2016 --max-year 2019 --sample-per-year 20 --n-topics 4 --n-word-svd 4 --n-char-svd 3
PYTHONPATH=src pytest -q
```

```bash
cd /Users/damrorus/CV_projects/ml_pet_projects/movielens_recommender
PYTHONPATH=src python -m movielens_recommender.cli --synthetic --epochs 1 --latent-dim 4
PYTHONPATH=src pytest -q
```

```bash
cd /Users/damrorus/CV_projects/ml_pet_projects/ai_text_detector
PYTHONPATH=src python -m ai_text_detector.cli
PYTHONPATH=src pytest -q
```

```bash
cd /Users/damrorus/CV_projects/ml_pet_projects/image_captioning
PYTHONPATH=src python -m image_captioning.cli
PYTHONPATH=src pytest -q
```

The Lenta and MovieLens projects also support full dataset runs; see their
project READMEs for commands.
