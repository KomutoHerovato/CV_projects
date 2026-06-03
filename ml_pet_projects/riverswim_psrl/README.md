# RiverSwim PSRL

Pet project based on the notebook `river_swim_psrl_colab(1).ipynb`.

The project compares tabular Q-learning with Posterior Sampling for
Reinforcement Learning (PSRL) on the RiverSwim environment. RiverSwim is a
small MDP where shallow random exploration often stays near a small reward,
while PSRL commits to sampled optimistic dynamics and reaches the large reward
on the right more reliably.

## What is inside

- Finite-horizon RiverSwim environment.
- Backward dynamic programming solver for known MDPs.
- Q-learning baseline.
- PSRL with Dirichlet transition posterior and Beta reward posterior.
- Reproducible CLI that saves metrics and plots.
- Smoke test for quick CI/local verification.

## Quick start

```bash
cd /Users/damrorus/CV_projects/ml_pet_projects/riverswim_psrl
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m riverswim_psrl.cli --episodes 200 --trials 3
```

Artifacts are written to `outputs/riverswim_psrl`:

- `history.csv`
- `summary.csv`
- `reward.png`
- `right_visits.png`
- `left_visits.png`
- `swim_share.png`
- `regret.png`

## Test

```bash
PYTHONPATH=src pytest -q
```
