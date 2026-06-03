from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .agents import PSRLAgent, QLearningAgent
from .env import RiverSwim
from .planning import finite_horizon_dp


AgentFactory = Callable[[int], object]


@dataclass(frozen=True)
class ExperimentConfig:
    n_intermediate: int = 4
    horizon: int = 128
    start_state: int = 1
    episodes: int = 600
    trials: int = 8
    base_seed: int = 10_000

    @property
    def n_states(self) -> int:
        return 2 + self.n_intermediate


def optimal_episode_value(config: ExperimentConfig) -> float:
    env = RiverSwim(
        n_intermediate=config.n_intermediate,
        horizon=config.horizon,
        start_state=config.start_state,
    )
    _, values, _ = finite_horizon_dp(env.P, env.R, horizon=config.horizon, gamma=1.0)
    return float(values[0, config.start_state])


def run_agent(
    agent: object,
    label: str,
    config: ExperimentConfig,
    trial: int,
    optimal_value: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for episode in range(config.episodes):
        env = RiverSwim(
            n_intermediate=config.n_intermediate,
            horizon=config.horizon,
            start_state=config.start_state,
            seed=config.base_seed + trial * 100_000 + episode,
        )
        state = env.reset()
        agent.start_episode()

        total_reward = 0.0
        right_visits = 0
        left_visits = 0
        right_actions = 0
        done = False

        while not done:
            t = env.t
            action = int(agent.act(t, state))
            right_actions += int(action == RiverSwim.RIGHT)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            total_reward += reward
            right_visits += int(next_state == env.S - 1)
            left_visits += int(next_state == 0)
            state = next_state

        rows.append(
            {
                "algorithm": label,
                "trial": trial,
                "episode": episode,
                "reward": total_reward,
                "right_visits": right_visits,
                "left_visits": left_visits,
                "swim_share": right_actions / config.horizon,
                "regret": optimal_value - total_reward,
            }
        )
    return pd.DataFrame(rows)


def q_learning_factory(config: ExperimentConfig) -> AgentFactory:
    return lambda trial: QLearningAgent(
        n_states=config.n_states,
        n_actions=2,
        alpha=0.15,
        gamma=0.98,
        epsilon=0.20,
        epsilon_decay=0.998,
        min_epsilon=0.02,
        seed=trial,
    )


def psrl_factory(config: ExperimentConfig) -> AgentFactory:
    return lambda trial: PSRLAgent(
        n_states=config.n_states,
        n_actions=2,
        horizon=config.horizon,
        gamma=1.0,
        transition_prior=0.5,
        reward_alpha0=1.0,
        reward_beta0=1.0,
        seed=trial,
    )


def run_comparison(config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    opt = optimal_episode_value(config)
    frames = []
    factories = {
        "Q-learning": q_learning_factory(config),
        "PSRL": psrl_factory(config),
    }
    for label, factory in factories.items():
        for trial in range(config.trials):
            frames.append(run_agent(factory(trial), label, config, trial, opt))

    history = pd.concat(frames, ignore_index=True)
    summary = (
        history[history["episode"] >= max(0, config.episodes - 100)]
        .groupby("algorithm", as_index=False)[
            ["reward", "right_visits", "left_visits", "swim_share", "regret"]
        ]
        .mean()
        .sort_values("reward", ascending=False)
    )
    return history, summary
