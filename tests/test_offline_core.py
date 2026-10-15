"""Unit tests for noise, rewards, and signal metrics (no GPU)."""

from __future__ import annotations

import numpy as np

from rlvr.metrics.signal import group_signal_stats, summarize_groups
from rlvr.noise.corruption import PLANNED_SPECS, apply_noise
from rlvr.rewards.aggregation import binary_rewards, fractional_rewards, grpo_advantages


def test_fractional_and_binary():
    z = np.array([[1, 1, 1], [1, 1, 0], [0, 0, 0]], dtype=np.int8)
    assert np.allclose(fractional_rewards(z), [1.0, 2 / 3, 0.0])
    assert np.allclose(binary_rewards(z), [1.0, 0.0, 0.0])


def test_fn_hurts_binary_correct_more_than_fp_hurts_wrong():
    """Sanity check aligned with Proposition 1 extremes (monte carlo)."""
    rng = np.random.default_rng(0)
    correct = np.ones((10000, 3), dtype=np.int8)
    wrong = np.zeros((10000, 3), dtype=np.int8)
    fn_noisy, _ = apply_noise(correct, PLANNED_SPECS["fn_only"], rng)
    fp_noisy, _ = apply_noise(wrong, PLANNED_SPECS["fp_only"], rng)
    p_keep = binary_rewards(fn_noisy).mean()
    p_invent = binary_rewards(fp_noisy).mean()
    assert 0.30 < p_keep < 0.40  # ~0.7^3
    assert 0.02 < p_invent < 0.05  # ~0.3^3


def test_zero_variance_group():
    r = np.zeros(16)
    s = group_signal_stats(r, r)
    assert s.zero_variance_clean
    assert s.advantage_sign_agreement is None


def test_perfect_noise_preserves_signal():
    clean = np.array([1.0, 1.0, 0.0, 0.0])
    s = group_signal_stats(clean, clean.copy())
    assert s.advantage_sign_agreement == 1.0
    assert s.pairwise_fidelity == 1.0
    assert s.top1_preserved == 1.0


def test_grpo_advantages_unit_std():
    r = np.array([1.0, 0.0, 1.0, 0.0])
    a = grpo_advantages(r)
    assert abs(a.mean()) < 1e-9
    assert abs(a.std(ddof=0) - 1.0) < 1e-6


def test_summarize_groups_buckets():
    stats = [group_signal_stats(np.array([1.0, 0.0]), np.array([1.0, 0.0])) for _ in range(5)]
    sm = summarize_groups(stats)
    assert sm["mean_advantage_sign_agreement_bucket"] == "survives"
