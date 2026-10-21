"""Shared experiment configuration loading and validation."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rlvr.noise.corruption import NoiseSpec, resolve_noise_specs
from rlvr.rewards.aggregation import get_reward_function


class ConfigError(ValueError):
    """Raised when an experiment config is missing or internally inconsistent."""


@dataclass(frozen=True)
class ModelConfig:
    id: str
    role: str = "main_candidate"
    dtype: str = "bfloat16"


@dataclass(frozen=True)
class ExperimentCondition:
    name: str
    reward: str
    noise: NoiseSpec


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    study: str
    source: Path
    model: ModelConfig
    seeds: tuple[int, ...]
    conditions: tuple[ExperimentCondition, ...]
    values: dict[str, Any]

    def condition(self, name: str) -> ExperimentCondition:
        """Return one configured condition by its stable name."""
        for condition in self.conditions:
            if condition.name == name:
                return condition
        available = ", ".join(item.name for item in self.conditions)
        raise ConfigError(f"Unknown condition {name!r}; choose one of: {available}")

    def require_seed(self, seed: int) -> int:
        """Validate a seed against the predeclared matched-seed protocol."""
        if seed not in self.seeds:
            raise ConfigError(f"Seed {seed} is not configured; choose one of: {list(self.seeds)}")
        return seed


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML, recursively applying relative ``extends`` files."""
    return _load_config(Path(path).resolve(), stack=())


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load a shared online-study config and expand its condition matrix."""
    source = Path(path).resolve()
    values = load_config(source)
    if values.get("schema_version") != 1:
        raise ConfigError("schema_version must be 1")
    experiment = _mapping(values, "experiment")
    matrix = _mapping(values, "matrix")
    training = _mapping(values, "training")
    if str(training.get("algorithm")) != "grpo":
        raise ConfigError("training.algorithm must be grpo")

    name = _required_string(experiment, "name")
    study = _required_string(experiment, "study")
    model = resolve_model_config(values)

    rewards_raw = matrix.get("reward_types")
    if not isinstance(rewards_raw, list) or not rewards_raw:
        raise ConfigError("matrix.reward_types must be a non-empty list")
    rewards = [str(value) for value in rewards_raw]
    try:
        for reward in rewards:
            get_reward_function(reward)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    if len(rewards) != len(set(rewards)):
        raise ConfigError(f"matrix.reward_types contains duplicates: {rewards}")

    noise_raw = matrix.get("noise_specs")
    if not isinstance(noise_raw, list) or not noise_raw:
        raise ConfigError("matrix.noise_specs must be a non-empty list")
    try:
        noise_specs = resolve_noise_specs(noise_raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc

    seeds_raw = training.get("seeds")
    if not isinstance(seeds_raw, list) or not seeds_raw:
        raise ConfigError("training.seeds must be a non-empty list")
    try:
        seeds = tuple(int(seed) for seed in seeds_raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError("training.seeds must contain integers") from exc
    if len(seeds) != len(set(seeds)):
        raise ConfigError(f"training.seeds contains duplicates: {list(seeds)}")

    conditions = tuple(
        ExperimentCondition(
            name=f"{noise.name}__{reward}",
            reward=reward,
            noise=noise,
        )
        for noise in noise_specs
        for reward in rewards
    )
    return ExperimentConfig(
        name=name,
        study=study,
        source=source,
        model=model,
        seeds=seeds,
        conditions=conditions,
        values=values,
    )


def resolve_model_config(
    values: Mapping[str, Any],
    *,
    model_id: str | None = None,
    role: str | None = None,
) -> ModelConfig:
    """Resolve one model from either the shared or legacy offline schema."""
    default_dtype = str(values.get("dtype", "bfloat16"))

    if isinstance(values.get("model"), Mapping):
        raw = dict(values["model"])
        configured_id = _required_string(raw, "id")
        configured_role = str(raw.get("role", "main_candidate"))
        configured_dtype = str(raw.get("dtype", default_dtype))
        if model_id is not None and model_id != configured_id:
            return ModelConfig(id=model_id, role=role or configured_role, dtype=configured_dtype)
        return ModelConfig(
            id=model_id or configured_id,
            role=role or configured_role,
            dtype=configured_dtype,
        )

    raw_models = values.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        if model_id is None:
            raise ConfigError("Config must define model.id or a non-empty models list")
        return ModelConfig(id=model_id, role=role or "main_candidate", dtype=default_dtype)

    candidates = [item for item in raw_models if isinstance(item, Mapping)]
    selected: Mapping[str, Any] | None = None
    if model_id is not None:
        selected = next((item for item in candidates if item.get("id") == model_id), None)
    if selected is None and role is not None:
        selected = next((item for item in candidates if item.get("role") == role), None)
    if selected is None:
        selected = candidates[0] if candidates else None
    if selected is None:
        raise ConfigError("models must contain mappings with at least an id")

    selected_id = model_id or _required_string(selected, "id")
    return ModelConfig(
        id=selected_id,
        role=role or str(selected.get("role", "main_candidate")),
        dtype=str(selected.get("dtype", default_dtype)),
    )


def _load_config(path: Path, *, stack: tuple[Path, ...]) -> dict[str, Any]:
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ConfigError(f"Circular config inheritance: {chain}")
    if not path.exists():
        raise ConfigError(f"Config does not exist: {path}")

    raw = yaml.safe_load(path.read_text())
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Top-level YAML value must be a mapping: {path}")

    extends = raw.pop("extends", None)
    if extends is None:
        return raw
    parents = [extends] if isinstance(extends, str) else extends
    if not isinstance(parents, list) or not all(isinstance(item, str) for item in parents):
        raise ConfigError(f"extends must be a path or list of paths: {path}")

    merged: dict[str, Any] = {}
    next_stack = (*stack, path)
    for parent in parents:
        parent_values = _load_config((path.parent / parent).resolve(), stack=next_stack)
        merged = _deep_merge(merged, parent_values)
    return _deep_merge(merged, raw)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _mapping(values: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = values.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{key} must be a mapping")
    return value


def _required_string(values: Mapping[str, Any], key: str) -> str:
    value = str(values.get(key, "")).strip()
    if not value:
        raise ConfigError(f"Missing non-empty value: {key}")
    return value
