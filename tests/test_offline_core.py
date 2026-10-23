"""Unit tests for noise, rewards, and signal metrics (no GPU)."""

from __future__ import annotations

import numpy as np
import pytest

from rlvr.data.mbpp import resolve_mbpp_split
from rlvr.generate.stopping import (
    append_stopping_criteria,
    inject_generate_stopping_criteria,
)
from rlvr.metrics.signal import group_signal_stats, summarize_groups
from rlvr.noise.corruption import PLANNED_SPECS, apply_noise
from rlvr.rewards.aggregation import binary_rewards, fractional_rewards, grpo_advantages
from scripts.grade_and_analyze_offline import _write_markdown as _write_analysis_report
from scripts.run_offline_diagnostic import _write_markdown_report


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
    assert abs(a.std(ddof=1) - 1.0) < 1e-6


def test_summarize_groups_buckets():
    stats = [group_signal_stats(np.array([1.0, 0.0]), np.array([1.0, 0.0])) for _ in range(5)]
    sm = summarize_groups(stats)
    assert sm["mean_advantage_sign_agreement_bucket"] == "survives"


def test_mbpp_split_validation():
    assert resolve_mbpp_split("test") == "test"
    with pytest.raises(ValueError, match="Unknown MBPP split"):
        resolve_mbpp_split("dev")


def test_generate_stopping_criteria_injection_is_temporary():
    class DummyModel:
        @staticmethod
        def generate(**kwargs):
            return kwargs

    model = DummyModel()
    criterion = object()
    with inject_generate_stopping_criteria(model, criterion):
        result = model.generate()
        assert len(result["stopping_criteria"]) == 1
        assert result["stopping_criteria"][0] is criterion

    assert model.generate() == {}


def test_stopping_criteria_append_does_not_mutate_existing_list():
    existing = [object()]
    criterion = object()

    combined = append_stopping_criteria(existing, criterion)

    assert combined == [existing[0], criterion]
    assert len(existing) == 1


def test_test_difficulty_report_is_not_presented_as_model_screening(tmp_path):
    report = {
        "model": "test-model",
        "purpose": "difficulty_stratification_only",
        "config": {
            "model_revision": "abc123",
            "group_size": 16,
            "max_new_tokens": 512,
            "stop_sequence": "[DONE]",
        },
        "difficulty_summary": {
            "n_problems": 500,
            "mean_fractional_reward": 0.4,
            "frac_problems_nonzero_variance": 0.8,
            "stratum_counts": {"easy": 100, "medium": 200, "hard": 200},
        },
    }

    for index, writer in enumerate((_write_markdown_report, _write_analysis_report)):
        path = tmp_path / f"report-{index}.md"
        writer(path, report)
        rendered = path.read_text()
        assert "Frozen test difficulty map" in rendered
        assert "not used for model selection" in rendered
        assert "Model screening" not in rendered
        assert "Proposition 1" not in rendered
