"""Test-level verification noise (Study A / B schedules)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NoiseSpec:
    name: str
    alpha: float = 0.0  # FPR: P(z̃=1 | z=0)
    beta: float = 0.0  # FNR: P(z̃=0 | z=1)
    mode: str = "class_conditional"  # or "independent_flip" / "group_flip"
    flip_p: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in {"class_conditional", "independent_flip", "group_flip"}:
            raise ValueError(f"Unknown noise mode: {self.mode}")
        for field_name, value in (
            ("alpha", self.alpha),
            ("beta", self.beta),
            ("flip_p", self.flip_p),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1, got {value}")


PLANNED_SPECS: dict[str, NoiseSpec] = {
    "clean": NoiseSpec(name="clean"),
    "fp_only": NoiseSpec(name="fp_only", alpha=0.3, beta=0.0),
    "fn_only": NoiseSpec(name="fn_only", alpha=0.0, beta=0.3),
    "fp_heavy": NoiseSpec(name="fp_heavy", alpha=0.3, beta=0.1),
    "fn_heavy": NoiseSpec(name="fn_heavy", alpha=0.1, beta=0.3),
    "ind_flip_10": NoiseSpec(name="ind_flip_10", mode="independent_flip", flip_p=0.1),
    "group_flip_10": NoiseSpec(name="group_flip_10", mode="group_flip", flip_p=0.1),
}


def resolve_noise_spec(value: str | Mapping[str, Any]) -> NoiseSpec:
    """Resolve a named or inline YAML noise definition.

    Named definitions keep existing configs concise. Inline definitions make it
    possible to add a study-specific rate without editing the global registry.
    """
    if isinstance(value, str):
        try:
            return PLANNED_SPECS[value]
        except KeyError as exc:
            available = ", ".join(sorted(PLANNED_SPECS))
            raise ValueError(f"Unknown noise spec {value!r}; choose one of: {available}") from exc

    if not isinstance(value, Mapping):
        raise TypeError("Noise spec must be a name or a mapping")

    name = str(value.get("name", "")).strip()
    if not name:
        raise ValueError("Inline noise spec requires a non-empty 'name'")

    base = PLANNED_SPECS.get(name, NoiseSpec(name=name))
    return NoiseSpec(
        name=name,
        alpha=float(value.get("alpha", base.alpha)),
        beta=float(value.get("beta", base.beta)),
        mode=str(value.get("mode", base.mode)),
        flip_p=float(value.get("flip_p", base.flip_p)),
    )


def resolve_noise_specs(values: Sequence[str | Mapping[str, Any]]) -> list[NoiseSpec]:
    if isinstance(values, (str, bytes)):
        raise TypeError("Noise specs must be a sequence, not a single string")
    specs = [resolve_noise_spec(value) for value in values]
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError(f"Noise spec names must be unique, got: {names}")
    return specs


def apply_noise(
    clean: np.ndarray,
    spec: NoiseSpec,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply noise to a clean outcome matrix of shape (G, K).

    Returns (noisy_outcomes, corruption_mask) where mask is 1 if flipped.
    """
    clean = np.asarray(clean, dtype=np.int8)
    noisy = clean.copy()
    mask = np.zeros_like(clean, dtype=np.int8)

    if spec.mode == "class_conditional":
        fail = clean == 0
        pass_ = clean == 1
        fp = fail & (rng.random(clean.shape) < spec.alpha)
        fn = pass_ & (rng.random(clean.shape) < spec.beta)
        noisy[fp] = 1
        noisy[fn] = 0
        mask = (fp | fn).astype(np.int8)
        return noisy, mask

    if spec.mode == "independent_flip":
        flip = rng.random(clean.shape) < spec.flip_p
        noisy = np.where(flip, 1 - clean, clean).astype(np.int8)
        mask = flip.astype(np.int8)
        return noisy, mask

    if spec.mode == "group_flip":
        if rng.random() < spec.flip_p:
            noisy = (1 - clean).astype(np.int8)
            mask = np.ones_like(clean, dtype=np.int8)
        return noisy, mask

    raise ValueError(f"Unknown noise mode: {spec.mode}")
