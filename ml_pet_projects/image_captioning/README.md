# Image Captioning

Pet project based on `task6(3).ipynb`.

The original notebook trains an image captioning model on a COCO Captions
subset: a frozen ResNet152 encoder extracts image features, and an LSTM decoder
generates captions token by token. This repository version turns that notebook
into a clean project with reproducible vocabulary/data utilities, documented
model setup, smoke tests, and GitHub-friendly artifacts.

## Why it is portfolio-ready

- Classic multimodal deep learning task: vision encoder + language decoder.
- Clear architecture: `ResNet152 -> projection -> LSTM decoder -> vocabulary`.
- COCO-oriented preprocessing: captions, vocabulary, max length, special tokens.
- TensorBoard/checkpoint workflow described from the notebook.
- Lightweight demo pipeline verifies project health without downloading COCO.

## Quick demo

```bash
cd /Users/damrorus/CV_projects/ml_pet_projects/image_captioning
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m image_captioning.cli
PYTHONPATH=src pytest -q
```

Artifacts are saved to `outputs/image_captioning`:

- `metrics.csv`
- `model_setup.csv`
- `demo_predictions.csv`

## Full experiment direction

For the full notebook-scale experiment:

1. Download COCO 2017 captions and a manageable image subset.
2. Build a vocabulary with `<PAD>`, `<START>`, `<END>`, `<UNK>`.
3. Use ImageNet preprocessing for ResNet152 inputs.
4. Freeze the ResNet backbone and train the projection layer plus LSTM decoder.
5. Track loss/perplexity and generated examples in TensorBoard.

The notebook reference is preserved in `notebooks/task6_image_captioning.ipynb`.
