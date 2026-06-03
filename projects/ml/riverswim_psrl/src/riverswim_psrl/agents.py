from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .planning import finite_horizon_dp


class QLearningAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.15,
        gamma: float = 0.98,
        epsilon: float = 0.20,
        epsilon_decay: float = 0.998,
        min_epsilon: float = 0.02,
        seed: int = 42,
    ) -> None:
        self.S = n_states
        self.A = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros((n_states, n_actions), dtype=float)

    def start_episode(self) -> None:
        return None

    def act(self, t: int, state: int) -> int:
        del t
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.A))
        return int(self.q[state].argmax())

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        target = reward if done else reward + self.gamma * float(self.q[next_state].max())
        self.q[state, action] += self.alpha * (target - self.q[state, action])
        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


class DirichletTransitionPosterior:
    def __init__(self, n_states: int, n_actions: int, prior: float = 0.5, seed: int = 42) -> None:
        self.S = n_states
        self.A = n_actions
        self.alpha = np.full((n_states, n_actions, n_states), prior, dtype=float)
        self.rng = np.random.default_rng(seed)

    def update(self, state: int, action: int, next_state: int) -> None:
        self.alpha[state, action, next_state] += 1.0

    def sample(self) -> np.ndarray:
        out = np.zeros_like(self.alpha)
        for state in range(self.S):
            for action in range(self.A):
                out[state, action] = self.rng.dirichlet(self.alpha[state, action])
        return out

    def mean(self) -> np.ndarray:
        return self.alpha / self.alpha.sum(axis=2, keepdims=True)


class BetaRewardPosterior:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.alpha = np.full((n_states, n_actions), alpha0, dtype=float)
        self.beta = np.full((n_states, n_actions), beta0, dtype=float)
        self.rng = np.random.default_rng(seed)

    def update(self, state: int, action: int, reward: float) -> None:
        reward = float(np.clip(reward, 0.0, 1.0))
        self.alpha[state, action] += reward
        self.beta[state, action] += 1.0 - reward

    def sample(self) -> np.ndarray:
        return self.rng.beta(self.alpha, self.beta)

    def mean(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)


@dataclass
class PSRLPlanningState:
    policy: np.ndarray
    values: np.ndarray
    q_values: np.ndarray


class PSRLAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        horizon: int,
        gamma: float = 1.0,
        transition_prior: float = 0.5,
        reward_alpha0: float = 1.0,
        reward_beta0: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.S = n_states
        self.A = n_actions
        self.horizon = horizon
        self.gamma = gamma
        self.transitions = DirichletTransitionPosterior(
            n_states, n_actions, prior=transition_prior, seed=seed
        )
        self.rewards = BetaRewardPosterior(
            n_states, n_actions, alpha0=reward_alpha0, beta0=reward_beta0, seed=seed + 1
        )
        self.plan: PSRLPlanningState | None = None

    def start_episode(self) -> None:
        sampled_transition = self.transitions.sample()
        sampled_reward = self.rewards.sample()
        policy, values, q_values = finite_horizon_dp(
            sampled_transition,
            sampled_reward,
            horizon=self.horizon,
            gamma=self.gamma,
        )
        self.plan = PSRLPlanningState(policy=policy, values=values, q_values=q_values)

    def act(self, t: int, state: int) -> int:
        if self.plan is None:
            self.start_episode()
        assert self.plan is not None
        return int(self.plan.policy[min(t, self.horizon - 1), state])

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        del done
        self.transitions.update(state, action, next_state)
        self.rewards.update(state, action, reward)

    def get_q_matrix(self) -> np.ndarray:
        if self.plan is None:
            return np.zeros((self.S, self.A), dtype=float)
        return self.plan.q_values[0]

    def get_reward_mean(self) -> np.ndarray:
        return self.rewards.mean()
