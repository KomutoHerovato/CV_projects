"""Опциональный «настоящий» RLHF на базе HuggingFace TRL (full-режим).

Этот скрипт — мост от учебной from-scratch реализации к индустриальному стеку.
Та же идея (выравнивание малой LM к позитивной тональности через PPO с
KL-контролем), но:

  * политика — реальная `distilgpt2` (82M);
  * reward — реальная sentiment-модель `lvwerra/distilbert-imdb`;
  * PPO — из библиотеки TRL.

Запускается отдельно и НЕ участвует в smoke-тестах (требует скачивания весов и
тяжелее по железу). На MacBook M1 работает через MPS, но медленно — это путь
«посмотреть, как то же самое выглядит в проде».

Зависимости (отдельно от основного requirements.txt):

    pip install -r requirements-trl.txt

Запуск:

    PYTHONPATH=src python -m mini_rlhf.trl_variant --steps 20
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="TRL PPO sentiment alignment (full mode).")
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--reward-model", default="lvwerra/distilbert-imdb")
    parser.add_argument("--dataset", default="stanfordnlp/imdb",
                        help="источник промптов (HF dataset id)")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    # устройство выбирает Accelerate автоматически (на M1 — MPS).
    args = parser.parse_args()

    # Импорты внутри main, чтобы основной пакет не зависел от TRL/transformers.
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, pipeline
    from trl import (
        AutoModelForCausalLMWithValueHead,
        PPOConfig,
        PPOTrainer,
    )
    from trl.core import LengthSampler

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    # IMDB как источник промптов (первые несколько токенов отзыва).
    ds = load_dataset(args.dataset, split="train")
    ds = ds.filter(lambda x: len(x["text"]) > 200, batched=False)
    sampler = LengthSampler(4, 12)

    def tokenize(sample):
        ids = tokenizer.encode(sample["text"])[: sampler()]
        sample["input_ids"] = ids
        sample["query"] = tokenizer.decode(ids)
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")

    model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model)

    # отклики имеют разную длину, поэтому батч собираем без stack — списком
    def collator(data):
        return {key: [d[key] for d in data] for key in data[0]}

    ppo_config = PPOConfig(
        model_name=args.model,
        batch_size=args.batch_size,
        mini_batch_size=args.batch_size,
        learning_rate=1.41e-5,
    )
    ppo_trainer = PPOTrainer(
        ppo_config, model, ref_model, tokenizer, dataset=ds, data_collator=collator
    )
    # устройством управляет Accelerate (на Mac по умолчанию MPS); выравниваем
    # на него и наши тензоры, иначе словим рассинхрон cpu/mps.
    device = ppo_trainer.accelerator.device
    print(f"[trl] device = {device}")

    # reward = логит позитивного класса sentiment-модели
    sentiment = pipeline("sentiment-analysis", model=args.reward_model, device=device)

    gen_kwargs = {"min_length": -1, "top_k": 0.0, "top_p": 1.0, "do_sample": True,
                  "pad_token_id": tokenizer.eos_token_id}
    out_sampler = LengthSampler(8, 16)

    step = 0
    for batch in ppo_trainer.dataloader:
        if step >= args.steps:
            break
        query_tensors = [q.to(device) for q in batch["input_ids"]]
        response_tensors = []
        for q in query_tensors:
            gen_len = out_sampler()
            resp = ppo_trainer.generate(q, max_new_tokens=gen_len, **gen_kwargs)
            response_tensors.append(resp.squeeze()[-gen_len:])
        batch["response"] = [tokenizer.decode(r) for r in response_tensors]

        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_out = sentiment(texts, top_k=None)
        rewards = [
            torch.tensor(next(d["score"] for d in out if d["label"] == "POSITIVE"))
            for out in pipe_out
        ]

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        mean_r = float(torch.stack(rewards).mean())
        print(f"step {step:03d} | mean positive reward = {mean_r:.3f} | "
              f"kl = {stats.get('objective/kl', float('nan')):.3f}")
        step += 1


if __name__ == "__main__":
    main()
