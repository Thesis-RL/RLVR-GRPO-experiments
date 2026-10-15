"""Binary and fractional reward maps from test-outcome matrices."""

from __future__ import annotations

import numpy as np


def fractional_rewards(outcomes: np.ndarray) -> np.ndarray:
    """outcomes: (G, K) → rewards in {0, 1/K, ..., 1}."""
    outcomes = np.asarray(outcomes, dtype=np.float64)
    return outcomes.mean(axis=-1)


def binary_rewards(outcomes: np.ndarray) -> np.ndarray:
    """All tests must pass."""
    outcomes = np.asarray(outcomes, dtype=np.int8)
    return (outcomes.min(axis=-1) == 1).astype(np.float64)


def grpo_advantages(rewards: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Within-group standardized advantages (GRPO-style)."""
    rewards = np.asarray(rewards, dtype=np.float64)
    mean = rewards.mean()
    std = rewards.std(ddof=0)
    if std < eps:
        return np.zeros_like(rewards)
    return (rewards - mean) / (std + eps)
