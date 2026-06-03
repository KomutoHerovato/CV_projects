"""Posterior Sampling for Reinforcement Learning on RiverSwim."""

from .agents import PSRLAgent, QLearningAgent
from .env import RiverSwim
from .experiment import ExperimentConfig, run_comparison
from .planning import finite_horizon_dp

__all__ = [
    "ExperimentConfig",
    "PSRLAgent",
    "QLearningAgent",
    "RiverSwim",
    "finite_horizon_dp",
    "run_comparison",
]
