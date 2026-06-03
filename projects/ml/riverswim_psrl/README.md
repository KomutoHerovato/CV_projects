# RiverSwim PSRL

Проект по мотивам ноутбука `river_swim_psrl_colab(1).ipynb`.

Задача показывает, почему обычный tabular Q-learning часто застревает около
маленькой награды слева в среде RiverSwim, а Posterior Sampling for
Reinforcement Learning (PSRL) чаще находит большую награду справа за счёт
эпизодической глубокой разведки.

## Что внутри

- Finite-horizon среда RiverSwim.
- Решение известного MDP обратным динамическим программированием.
- Baseline на Q-learning.
- PSRL с Dirichlet posterior для переходов и Beta posterior для наград.
- CLI, который сохраняет метрики и графики.
- Быстрый smoke-тест для проверки проекта.

## Быстрый запуск

```bash
cd /Users/damrorus/CV_projects/projects/ml/riverswim_psrl
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m riverswim_psrl.cli --episodes 200 --trials 3
```

Результаты сохраняются в `outputs/riverswim_psrl`:

- `history.csv`
- `summary.csv`
- `reward.png`
- `right_visits.png`
- `left_visits.png`
- `swim_share.png`
- `regret.png`

## Тесты

```bash
PYTHONPATH=src pytest -q
```
