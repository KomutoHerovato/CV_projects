"""Быстрые smoke-тесты: весь пайплайн на CPU за секунды, без скачиваний."""
from __future__ import annotations

import torch

from mini_rlhf.config import RLHFConfig
from mini_rlhf.data import make_preference_pairs, make_sft_dataset
from mini_rlhf.model import PolicyWithValue, RewardModel
from mini_rlhf.pipeline import run_pipeline
from mini_rlhf.reward import programmatic_reward
from mini_rlhf.reward_model import train_reward_model
from mini_rlhf.tokenizer import ToyTokenizer


def test_tokenizer_roundtrip():
    tok = ToyTokenizer()
    ids = tok.encode(["this", "movie", "was", "good"])
    assert ids[0] == tok.bos_id
    assert "good" in tok.decode(ids)


def test_programmatic_reward_sign():
    tok = ToyTokenizer()
    pos = tok.encode(["good", "great", "love"])
    neg = tok.encode(["bad", "awful", "terrible"])
    assert programmatic_reward(pos, tok) > 0
    assert programmatic_reward(neg, tok) < 0


def test_model_shapes():
    cfg = RLHFConfig.smoke()
    tok = ToyTokenizer()
    policy = PolicyWithValue(tok.vocab_size, cfg.model)
    idx = torch.randint(0, tok.vocab_size, (4, 6))
    logits, values = policy(idx)
    assert logits.shape == (4, 6, tok.vocab_size)
    assert values.shape == (4, 6)
    out = policy.generate(idx, max_new_tokens=3)
    assert out.shape == (4, 9)


def test_preference_data():
    cfg = RLHFConfig.smoke()
    tok = ToyTokenizer()
    ch, ch_len, rj, rj_len = make_preference_pairs(tok, cfg, n=32)
    assert ch.shape[0] == rj.shape[0]
    # chosen должен иметь не меньшую истинную награду, чем rejected
    for i in range(ch.shape[0]):
        assert programmatic_reward(ch[i].tolist(), tok) >= programmatic_reward(rj[i].tolist(), tok)


def test_reward_model_learns():
    cfg = RLHFConfig.smoke()
    cfg.rm_steps = 60
    cfg.rm_n_pairs = 256
    tok = ToyTokenizer()
    out = train_reward_model(tok, cfg, torch.device("cpu"))
    # на синтетических предпочтениях RM должна заметно превзойти случай (0.5)
    assert out["rm_final_pref_acc"] > 0.7


def test_full_pipeline_ppo_runs():
    cfg = RLHFConfig.smoke()
    res = run_pipeline(cfg, out_dir="outputs/_smoke_ppo")
    m = res["metrics"]
    assert "rl_gt_reward" in m
    assert len(res["rl_history"]["gt_reward"]) == cfg.rl_iters


def test_full_pipeline_reinforce_runs():
    cfg = RLHFConfig.smoke()
    cfg.rl_algo = "reinforce"
    res = run_pipeline(cfg, out_dir="outputs/_smoke_reinforce")
    assert res["metrics"]["rl_algo"] == "reinforce"
