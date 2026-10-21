"""Binary and fractional reward maps from test-outcome matrices."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

RewardFunction = Callable[[np.ndarray], np.ndarray]


def fractional_rewards(outcomes: np.ndarray) -> np.ndarray:
    """outcomes: (G, K) → rewards in {0, 1/K, ..., 1}."""
    outcomes = np.asarray(outcomes, dtype=np.float64)
    return outcomes.mean(axis=-1)


def binary_rewards(outcomes: np.ndarray) -> np.ndarray:
    """All tests must pass."""
    outcomes = np.asarray(outcomes, dtype=np.int8)
    return (outcomes.min(axis=-1) == 1).astype(np.float64)


def grpo_advantages(rewards: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Within-group standardized advantages matching TRL's sample std."""
    rewards = np.asarray(rewards, dtype=np.float64)
    mean = rewards.mean()
    std = rewards.std(ddof=1) if rewards.size > 1 else 0.0
    if std < eps:
        return np.zeros_like(rewards)
    return (rewards - mean) / (std + eps)


REWARD_FUNCTIONS: dict[str, RewardFunction] = {
    "binary": binary_rewards,
    "fractional": fractional_rewards,
}


def get_reward_function(name: str) -> RewardFunction:
    try:
        return REWARD_FUNCTIONS[name]
    except KeyError as exc:
        available = ", ".join(sorted(REWARD_FUNCTIONS))
        raise ValueError(f"Unknown reward {name!r}; choose one of: {available}") from exc


def resolve_reward_functions(names: Sequence[str]) -> dict[str, RewardFunction]:
    if isinstance(names, (str, bytes)):
        raise TypeError("Reward names must be a sequence, not a single string")
    if len(names) != len(set(names)):
        raise ValueError(f"Reward names must be unique, got: {list(names)}")
    return {name: get_reward_function(name) for name in names}
