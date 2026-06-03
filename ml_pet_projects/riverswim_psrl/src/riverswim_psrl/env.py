from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RiverSwimConfig:
    n_intermediate: int = 4
    horizon: int = 128
    left_reward: float = 5e-3
    right_reward: float = 1.0
    start_state: int = 1
    seed: int = 42


class RiverSwim:
    """Small finite-horizon RiverSwim MDP.

    States form a chain from 0 to S - 1. Action 0 moves left with certainty.
    Action 1 tries to swim right and is intentionally stochastic, which makes
    deep exploration valuable.
    """

    LEFT = 0
    RIGHT = 1

    def __init__(
        self,
        n_intermediate: int = 4,
        horizon: int = 128,
        left_reward: float = 5e-3,
        right_reward: float = 1.0,
        start_state: int = 1,
        seed: int = 42,
    ) -> None:
        self.config = RiverSwimConfig(
            n_intermediate=n_intermediate,
            horizon=horizon,
            left_reward=left_reward,
            right_reward=right_reward,
            start_state=start_state,
            seed=seed,
        )
        self.rng = np.random.default_rng(seed)
        self.n_intermediate = n_intermediate
        self.horizon = horizon
        self.left_reward = left_reward
        self.right_reward = right_reward
        self.start_state = start_state
        self.S = 2 + n_intermediate
        self.A = 2
        if not 0 <= start_state < self.S:
            raise ValueError("start_state must be inside the chain")

        self.P = self._build_transition_matrix()
        self.R = self._build_reward_matrix()
        self.state = self.start_state
        self.t = 0

    def _build_transition_matrix(self) -> np.ndarray:
        p = np.zeros((self.S, self.A, self.S), dtype=float)

        for state in range(self.S):
            p[state, self.LEFT, max(0, state - 1)] = 1.0

        p[0, self.RIGHT, 0] = 0.6
        p[0, self.RIGHT, 1] = 0.4
        for state in range(1, self.S - 1):
            p[state, self.RIGHT, state - 1] = 0.05
            p[state, self.RIGHT, state] = 0.6
            p[state, self.RIGHT, state + 1] = 0.35
        p[self.S - 1, self.RIGHT, self.S - 2] = 0.05
        p[self.S - 1, self.RIGHT, self.S - 1] = 0.95
        return p

    def _build_reward_matrix(self) -> np.ndarray:
        rewards = np.zeros((self.S, self.A), dtype=float)
        rewards[0, self.LEFT] = self.left_reward
        rewards[self.S - 1, self.RIGHT] = self.right_reward
        return rewards

    def reset(self) -> int:
        self.state = self.start_state
        self.t = 0
        return self.state

    def step(self, action: int) -> tuple[int, float, bool, dict[str, int]]:
        if action not in (self.LEFT, self.RIGHT):
            raise ValueError("action must be 0 (left) or 1 (right)")
        state = self.state
        next_state = int(self.rng.choice(self.S, p=self.P[state, action]))
        reward = float(self.R[state, action])
        self.state = next_state
        self.t += 1
        done = self.t >= self.horizon
        return next_state, reward, done, {"t": self.t}
