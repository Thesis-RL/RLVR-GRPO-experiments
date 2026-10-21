"""CPU-only tests for the online reward path and smoke profile."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from rlvr.config import ConfigError, ExperimentCondition, load_experiment_config
from rlvr.noise.corruption import NoiseSpec
from rlvr.online.reward import MBPPNoisyReward
from rlvr.online.settings import resolve_training_settings
from rlvr.run_logging import RunContext


def _context(condition: ExperimentCondition) -> RunContext:
    return RunContext(
        pipeline="online_grpo",
        experiment="test",
        study="study_a",
        model="test-model",
        model_role="pipeline_debug",
        condition=condition.name,
        reward=condition.reward,
        noise=condition.noise.name,
        seed=0,
    )


def _call_reward(reward: MBPPNoisyReward) -> list[float]:
    return reward(
        prompts=["write add"] * 2,
        completions=["def add(a, b):\n    return a + b", "def add(a, b):\n    return 0"],
        task_id=[1, 1],
        test_list=[["assert add(1, 2) == 3"]] * 2,
        test_setup_code=["", ""],
        completion_ids=[[1, 2], [3]],
        difficulty=["easy", "easy"],
        trainer_state=SimpleNamespace(global_step=4),
    )


def test_clean_online_reward_and_rollout_log(tmp_path):
    condition = ExperimentCondition(
        name="clean__fractional",
        reward="fractional",
        noise=NoiseSpec(name="clean"),
    )
    log_path = tmp_path / "rollouts.jsonl"
    reward = MBPPNoisyReward(
        condition=condition,
        seed=0,
        group_size=2,
        context=_context(condition),
        rollout_log_path=log_path,
    )

    assert _call_reward(reward) == [1.0, 0.0]
    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["clean_outcomes"] == [1]
    assert rows[0]["corruption_mask"] == [0]
    assert rows[0]["train_advantage"] > 0
    assert rows[1]["train_advantage"] < 0
    assert rows[0]["step"] == 4
    assert rows[0]["verification_call"] == 0
    assert reward.generated_tokens == 3


def test_fp_noise_is_applied_before_fractional_aggregation(tmp_path):
    condition = ExperimentCondition(
        name="fp_all__fractional",
        reward="fractional",
        noise=NoiseSpec(name="fp_all", alpha=1.0),
    )
    reward = MBPPNoisyReward(
        condition=condition,
        seed=0,
        group_size=2,
        context=_context(condition),
        rollout_log_path=tmp_path / "rollouts.jsonl",
    )

    assert _call_reward(reward) == [1.0, 1.0]


def test_smoke_profile_is_resolved_and_main_is_blocked():
    experiment = load_experiment_config("configs/priority_bridge.yaml")

    smoke = resolve_training_settings(experiment.values, smoke=True)
    assert smoke.group_size == 8
    assert smoke.gradient_accumulation_steps == smoke.group_size
    assert smoke.max_steps == 2
    assert smoke.kl_coefficient == 0.0

    with pytest.raises(ConfigError, match="protocol_status"):
        resolve_training_settings(experiment.values, smoke=False)

    invalid = deepcopy(experiment.values)
    invalid["training"]["scale_rewards"] = "batch"
    with pytest.raises(ConfigError, match="logged advantages"):
        resolve_training_settings(invalid, smoke=True)


def test_reward_rejects_incomplete_group():
    condition = ExperimentCondition(
        name="clean__binary",
        reward="binary",
        noise=NoiseSpec(name="clean"),
    )
    reward = MBPPNoisyReward(
        condition=condition,
        seed=0,
        group_size=2,
        context=_context(condition),
    )

    with pytest.raises(ValueError, match="divisible"):
        reward(
            prompts=["prompt"],
            completions=["completion"],
            task_id=[1],
            test_list=[["assert True"]],
            test_setup_code=[""],
        )


def test_reward_logging_flags_remove_optional_payloads(tmp_path):
    condition = ExperimentCondition(
        name="clean__fractional",
        reward="fractional",
        noise=NoiseSpec(name="clean"),
    )
    log_path = tmp_path / "rollouts.jsonl"
    reward = MBPPNoisyReward(
        condition=condition,
        seed=0,
        group_size=2,
        context=_context(condition),
        rollout_log_path=log_path,
        save_clean_outcomes=False,
        save_noisy_outcomes=False,
        save_both_rewards=False,
        save_advantages=False,
    )

    _call_reward(reward)
    row = json.loads(log_path.read_text().splitlines()[0])
    assert "clean_outcomes" not in row
    assert "noisy_outcomes" not in row
    assert "clean_binary_reward" not in row
    assert "train_advantage" not in row
    assert row["train_reward"] == 1.0
    assert row["policy_step"] == 4


def test_noise_is_resampled_per_call_and_resume_continues_counter(tmp_path):
    condition = ExperimentCondition(
        name="fp_half__fractional",
        reward="fractional",
        noise=NoiseSpec(name="fp_half", alpha=0.5),
    )
    log_path = tmp_path / "rollouts.jsonl"
    reward = MBPPNoisyReward(
        condition=condition,
        seed=0,
        group_size=2,
        context=_context(condition),
        rollout_log_path=log_path,
    )

    _call_reward(reward)
    _call_reward(reward)
    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert {row["verification_call"] for row in rows} == {0, 1}
    assert rows[0]["noise_seed"] != rows[2]["noise_seed"]
    assert rows[0]["group_id"] != rows[2]["group_id"]
    assert "train_aggregation_pairwise_fidelity" in rows[0]
    assert "verifier_precision" in rows[0]

    resumed = MBPPNoisyReward(
        condition=condition,
        seed=0,
        group_size=2,
        context=_context(condition),
        rollout_log_path=log_path,
    )
    assert resumed.calls == 2
    assert resumed.generated_tokens == 6
    _call_reward(resumed)
    last = json.loads(log_path.read_text().splitlines()[-1])
    assert last["verification_call"] == 2
    assert resumed.generated_tokens == 9
