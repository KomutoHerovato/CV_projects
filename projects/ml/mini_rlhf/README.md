# mini RLHF

Компактный, но **честный** RLHF-пайплайн на маленькой языковой модели:
полный триплет **SFT → Reward Model → RL (PPO)** реализован с нуля на PyTorch
и запускается на CPU за секунды, а на MacBook M1 (MPS) — в полном объёме.

Проект сделан как демонстрация понимания механики выравнивания (alignment) LLM,
а не как обёртка над готовой библиотекой: каждый этап — это читаемый код, где
видно, что именно происходит. Для сравнения добавлен и «промышленный» вариант
на HuggingFace TRL.

## Что демонстрируется

| Этап | Файл | Суть |
| --- | --- | --- |
| **SFT** | [sft.py](src/mini_rlhf/sft.py) | Supervised fine-tuning базовой модели на демонстрациях (causal-LM кросс-энтропия). Получаем reference-политику. |
| **Reward Model** | [reward_model.py](src/mini_rlhf/reward_model.py) | Обучение reward-головы на парах предпочтений через **Bradley-Terry loss** `-log σ(r_chosen − r_rejected)`. |
| **RL** | [rl.py](src/mini_rlhf/rl.py) | **PPO** (clipped objective + value head + GAE) с **per-token KL-штрафом к reference**. Альтернатива — REINFORCE с baseline. |

Каноническая RLHF-цель, которую оптимизирует RL-этап:

```
maximize  E[ r_RM(x) ]  −  β · KL( π_policy ‖ π_reference )
```

reward-модель даёт обучаемый сигнал, а KL-штраф удерживает политику рядом с
SFT-reference — это и есть защита от деградации языка и от reward hacking.

## Задача (синтетическая, без скачиваний)

Маленький word-level «язык отзывов»: модель продолжает промпт вроде
`this movie was ...`. «Истинная» награда — позитивность текста (позитивные слова
минус негативные, см. [reward.py](src/mini_rlhf/reward.py)). Это *синтетический
аналог человеческих предпочтений*: пары `(chosen, rejected)` для reward-модели
размечаются этой функцией, но **механизм обучения reward-модели — настоящий**.
Благодаря синтетике весь прогон полностью воспроизводим и не требует интернета.

## Быстрый запуск

```bash
cd /Users/damrorus/CV_projects/projects/ml/mini_rlhf
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

PYTHONPATH=src python -m mini_rlhf.cli              # PPO, CPU, ~10-20 c
PYTHONPATH=src python -m mini_rlhf.cli --algo reinforce
PYTHONPATH=src pytest -q
```

Пример вывода (demo, CPU):

```
RM pref accuracy : 0.998
ground-truth reward (выше = позитивнее текст):
  base : -0.008      # необученная модель — нейтральна
  SFT  : +0.014      # после SFT всё ещё нейтральна (учила формат)
  RL   : +0.652      # после RL — устойчиво позитивный текст
final KL(policy||ref): 14.586

примеры после RL:
  this film very good great amazing great amazing great great great
  the film is it great amazing amazing amazing this great good
```

Видно ключевое: **SFT задаёт формат, а позитивность появляется именно на
RL-этапе** — выравнивание по reward-модели.

## Артефакты

После запуска в `outputs/mini_rlhf/`:

- `metrics.json` — сводные метрики (reward на стадиях base/SFT/RL, accuracy RM, финальный KL);
- `rl_history.csv` — по итерациям: ground-truth reward, скор reward-модели, KL, loss;
- `samples.txt` — примеры генераций на каждой стадии;
- `training_curves.png` — кривые обучения всех трёх этапов;
- `config.json` — полный конфиг прогона.

## Что здесь можно обсудить на собеседовании по alignment

- **Bradley-Terry / preference modeling** — почему reward учат из *парных*
  сравнений, а не из абсолютных оценок.
- **KL-контроль и reference policy** — зачем нужен `β·KL`, что происходит при
  слишком маленьком/большом `β` (попробуйте `--kl-coef 0.5` против `0.01`).
- **Reward hacking / mode collapse** — при слабом KL политика вырождается в
  повторение позитивных слов (`great amazing great ...`): reward растёт, а текст
  деградирует. Это видно прямо в `samples.txt` — наглядный учебный пример.
- **PPO vs REINFORCE** — clipping и value-baseline против простого
  REINFORCE-with-baseline; сравните KL и стабильность (`--algo`).
- **Reward whitening, GAE, advantage normalization** — приёмы стабилизации,
  реализованные в [rl.py](src/mini_rlhf/rl.py).

## Параметры

```bash
PYTHONPATH=src python -m mini_rlhf.cli \
    --algo ppo --device mps \      # на M1 для более долгого прогона
    --rl-iters 200 --kl-coef 0.1 \
    --out-dir outputs/mini_rlhf
```

## Full-режим: тот же RLHF на реальной LM через TRL

[trl_variant.py](src/mini_rlhf/trl_variant.py) — мост к индустриальному стеку:
`distilgpt2` (82M) + реальная sentiment-модель как reward + `trl.PPOTrainer` на
датасете IMDB. Работает на M1 (MPS), но медленно и требует скачивания весов,
поэтому вынесен отдельно и **не входит в smoke-тесты**.

```bash
pip install -r requirements-trl.txt
PYTHONPATH=src python -m mini_rlhf.trl_variant --steps 20 --device mps
```

## Про железо

- **demo + тесты** — CPU, секунды, без скачиваний. Это основной путь.
- **full-режим (TRL, distilgpt2)** — MacBook M1 (MPS) тянет для нескольких
  десятков PPO-шагов; этого достаточно, чтобы увидеть рост reward.
- **Масштабирование** (модель ≥ GPT-2 medium, длинные последовательности, много
  PPO-шагов) — уже стоит арендовать GPU-сервер: ничего в коде менять не нужно,
  кроме `--device cuda` и числа шагов.

## Структура

```
src/mini_rlhf/
  config.py        # гиперпараметры (demo / smoke)
  tokenizer.py     # синтетический word-level словарь и промпты
  reward.py        # ground-truth программная reward-функция
  data.py          # демонстрации для SFT и пары предпочтений
  model.py         # TinyGPT + policy/value головы + reward-модель
  sft.py           # этап 1: SFT
  reward_model.py  # этап 2: reward model (Bradley-Terry)
  rl.py            # этап 3: PPO / REINFORCE с KL-контролем
  pipeline.py      # оркестрация и артефакты
  cli.py           # точка входа
  trl_variant.py   # опциональный full-режим на HuggingFace TRL
tests/test_smoke.py
notebooks/walkthrough.ipynb
```
