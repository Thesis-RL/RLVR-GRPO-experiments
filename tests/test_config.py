"""Tests for shared study configs and run metadata (no model/GPU)."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from rlvr.config import load_experiment_config
from rlvr.noise.corruption import resolve_noise_spec
from rlvr.rewards.aggregation import get_reward_function
from rlvr.run_logging import RunContext, write_run_manifest

ROOT = Path(__file__).resolve().parents[1]


def test_priority_bridge_expands_to_six_conditions():
    cfg = load_experiment_config(ROOT / "configs/priority_bridge.yaml")

    assert cfg.model.id == "Qwen/Qwen3-1.7B-Base"
    assert cfg.seeds == (0, 1, 2)
    assert cfg.values["generation"]["max_new_tokens"] == 512
    assert cfg.values["generation"]["stop_sequence"] == "[DONE]"
    assert [condition.name for condition in cfg.conditions] == [
        "clean__binary",
        "clean__fractional",
        "fp_only__binary",
        "fp_only__fractional",
        "fn_only__binary",
        "fn_only__fractional",
    ]


def test_study_configs_keep_their_own_matrices():
    study_a = load_experiment_config(ROOT / "configs/study_a.yaml")
    study_b = load_experiment_config(ROOT / "configs/study_b.yaml")

    assert len(study_a.conditions) == 6
    assert {condition.reward for condition in study_a.conditions} == {"binary", "fractional"}
    assert len(study_b.conditions) == 5
    assert {condition.reward for condition in study_b.conditions} == {"fractional"}


def test_inline_noise_rate_overrides_named_default():
    spec = resolve_noise_spec({"name": "fp_only", "alpha": 0.2})

    assert spec.alpha == 0.2
    assert spec.beta == 0.0


def test_reward_selection_rejects_unknown_name():
    try:
        get_reward_function("unknown")
    except ValueError as exc:
        assert "Unknown reward" in str(exc)
    else:
        raise AssertionError("unknown reward should be rejected")


def test_manifest_records_resolved_config_and_run_identity(tmp_path):
    context = RunContext(
        pipeline="online_grpo",
        experiment="priority_bridge",
        study="shared_bridge",
        model="Qwen/Qwen3-1.7B-Base",
        model_role="main_candidate",
        condition="fp_only__fractional",
        reward="fractional",
        noise="fp_only",
        seed=0,
    )
    config = {"model": {"id": "Qwen/Qwen3-1.7B-Base"}}

    write_run_manifest(
        tmp_path,
        context=context,
        config=config,
        config_path=ROOT / "configs/priority_bridge.yaml",
    )

    resolved = yaml.safe_load((tmp_path / "resolved_config.yaml").read_text())
    metadata = json.loads((tmp_path / "run_metadata.json").read_text())
    assert resolved == config
    assert metadata["run_id"] == "priority_bridge__fp_only__fractional__seed-0"
    assert metadata["context"]["reward"] == "fractional"
