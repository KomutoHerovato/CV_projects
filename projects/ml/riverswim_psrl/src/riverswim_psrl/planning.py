from __future__ import annotations

import numpy as np


def finite_horizon_dp(
    transition: np.ndarray,
    reward: np.ndarray,
    horizon: int,
    gamma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve a finite-horizon tabular MDP by backward induction."""
    n_states, n_actions, n_next = transition.shape
    if n_states != n_next:
        raise ValueError("transition must have shape (S, A, S)")
    if reward.shape != (n_states, n_actions):
        raise ValueError("reward must have shape (S, A)")

    values = np.zeros((horizon + 1, n_states), dtype=float)
    q_values = np.zeros((horizon, n_states, n_actions), dtype=float)
    policy = np.zeros((horizon, n_states), dtype=int)

    for t in range(horizon - 1, -1, -1):
        q_values[t] = reward + gamma * np.einsum("san,n->sa", transition, values[t + 1])
        values[t] = q_values[t].max(axis=1)
        policy[t] = q_values[t].argmax(axis=1)

    return policy, values, q_values
