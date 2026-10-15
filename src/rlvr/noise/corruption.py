"""Test-level verification noise (Study A / B schedules)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NoiseSpec:
    name: str
    alpha: float = 0.0  # FPR: P(z̃=1 | z=0)
    beta: float = 0.0  # FNR: P(z̃=0 | z=1)
    mode: str = "class_conditional"  # or "independent_flip" / "group_flip"
    flip_p: float = 0.0


PLANNED_SPECS: dict[str, NoiseSpec] = {
    "clean": NoiseSpec(name="clean"),
    "fp_only": NoiseSpec(name="fp_only", alpha=0.3, beta=0.0),
    "fn_only": NoiseSpec(name="fn_only", alpha=0.0, beta=0.3),
    "fp_heavy": NoiseSpec(name="fp_heavy", alpha=0.3, beta=0.1),
    "fn_heavy": NoiseSpec(name="fn_heavy", alpha=0.1, beta=0.3),
    "ind_flip_10": NoiseSpec(name="ind_flip_10", mode="independent_flip", flip_p=0.1),
    "group_flip_10": NoiseSpec(name="group_flip_10", mode="group_flip", flip_p=0.1),
}


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
