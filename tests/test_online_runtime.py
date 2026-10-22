"""Runtime-safety tests for online run lifecycle and evaluation."""

from __future__ import annotations

import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rlvr.config import ConfigError
from rlvr.data.mbpp import MBPPProblem
from rlvr.exec.mbpp_executor import ExecResult
from rlvr.online.callbacks import ScheduledCleanEvalCallback
from rlvr.online.data import load_difficulty_map
from rlvr.online.evaluation import (
    evaluate_clean_pass_at_1,
    evaluation_steps,
    fraction_steps,
    summarize_primary_metric,
)
from rlvr.online.lifecycle import prepare_run_directory
from rlvr.online.progress import RunProgressCallback
from rlvr.run_logging import RunContext, write_run_manifest


def _context() -> RunContext:
    return RunContext(
        pipeline="online_grpo",
        experiment="priority_bridge",
        study="shared_bridge",
        model="test-model",
        model_role="pipeline_debug",
        condition="clean__fractional",
        reward="fractional",
        noise="clean",
        seed=0,
        profile="smoke",
    )


def test_run_directory_rejects_accidental_reuse_and_validates_resume(tmp_path):
    out_dir = tmp_path / "run"
    config = {
        "resolved_run": {
            "profile": "smoke",
            "test_fractions": (0.9, 1.0),
            "lora_target_modules": ("q_proj", "v_proj"),
        }
    }
    context = _context()

    assert (
        prepare_run_directory(
            out_dir,
            context=context,
            resolved_config=config,
            resume=None,
        )
        is None
    )
    write_run_manifest(
        out_dir,
        context=context,
        config=config,
        config_path="configs/priority_bridge.yaml",
        environment={"git": {"commit": "abc123"}},
    )
    (out_dir / "checkpoint-2").mkdir()
    (out_dir / "checkpoint-2" / "trainer_state.json").write_text("{}")
    original_metadata = (out_dir / "run_metadata.json").read_text()

    with pytest.raises(ConfigError, match="not empty"):
        prepare_run_directory(
            out_dir,
            context=context,
            resolved_config=config,
            resume=None,
        )

    checkpoint = prepare_run_directory(
        out_dir,
        context=context,
        resolved_config=config,
        resume="latest",
        current_git_commit="abc123",
    )
    assert checkpoint == str(out_dir / "checkpoint-2")
    assert (out_dir / "run_metadata.json").read_text() == original_metadata
    assert json.loads((out_dir / "events.jsonl").read_text())["event"] == "resume_requested"

    with pytest.raises(ConfigError, match="Git commit changed"):
        prepare_run_directory(
            out_dir,
            context=context,
            resolved_config=config,
            resume="latest",
            current_git_commit="different",
        )

    outside = tmp_path / "checkpoint-9"
    outside.mkdir()
    with pytest.raises(ConfigError, match="inside the selected run"):
        prepare_run_directory(
            out_dir,
            context=context,
            resolved_config=config,
            resume=str(outside),
        )


def test_resume_rejects_changed_config(tmp_path):
    out_dir = tmp_path / "run"
    context = _context()
    original = {"resolved_run": {"profile": "smoke", "max_steps": 2}}
    prepare_run_directory(out_dir, context=context, resolved_config=original, resume=None)
    write_run_manifest(
        out_dir,
        context=context,
        config=original,
        config_path="configs/priority_bridge.yaml",
    )
    (out_dir / "checkpoint-2").mkdir()
    (out_dir / "checkpoint-2" / "trainer_state.json").write_text("{}")

    with pytest.raises(ConfigError, match="differs"):
        prepare_run_directory(
            out_dir,
            context=context,
            resolved_config={"resolved_run": {"profile": "smoke", "max_steps": 3}},
            resume="latest",
        )


def test_evaluation_is_resumable_and_restores_train_mode(tmp_path, monkeypatch):
    class DummyModel:
        training = True

        def eval(self):
            self.training = False

        def train(self):
            self.training = True

    calls = []

    def fake_sample(*args, **kwargs):  # noqa: ANN002, ANN003
        calls.append("sample")
        return ["def add(a, b):\n    return a + b"]

    monkeypatch.setattr("rlvr.online.evaluation.sample_completions", fake_sample)
    monkeypatch.setattr(
        "rlvr.online.evaluation.grade_code",
        lambda *args, **kwargs: ExecResult(outcomes=(1,)),  # noqa: ARG005
    )
    problem = MBPPProblem(
        task_id=1,
        text="add two numbers",
        code="",
        test_list=["assert add(1, 2) == 3"],
        test_setup_code="",
        challenge_test_list=[],
        split="test",
    )
    model = DummyModel()
    progress_updates = []
    kwargs = {
        "few_shot": [],
        "context": _context(),
        "out_dir": tmp_path,
        "split_name": "test",
        "checkpoint": "checkpoint-2",
        "step": 2,
        "seed": 0,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_new_tokens": 32,
        "timeout_s": 0.5,
        "mem_limit": 64 * 1024 * 1024,
        "difficulty_by_task": {1: "easy"},
        "progress_callback": lambda completed, total: progress_updates.append((completed, total)),
    }

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    expected_python = random.random()
    expected_numpy = np.random.random()
    expected_torch = torch.rand(1)
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    first = evaluate_clean_pass_at_1(model, object(), [problem], **kwargs)
    second = evaluate_clean_pass_at_1(model, object(), [problem], **kwargs)

    assert first == second
    assert first["pass_at_1"] == 1.0
    assert first["by_difficulty"]["easy"]["pass_at_1"] == 1.0
    assert calls == ["sample"]
    assert progress_updates == [(0, 1), (1, 1), (1, 1)]
    assert model.training is True
    assert random.random() == expected_python
    assert np.random.random() == expected_numpy
    assert torch.equal(torch.rand(1), expected_torch)


def test_run_progress_writes_atomic_training_and_evaluation_status(tmp_path):
    progress = RunProgressCallback(tmp_path, context=_context(), total_steps=4)
    state = SimpleNamespace(global_step=0, epoch=0.0)
    control = SimpleNamespace()
    progress.on_train_begin(None, state, control)

    progress._last_step_recorded_at -= 2.0
    state.global_step = 1
    state.epoch = 0.25
    progress.on_step_end(None, state, control)
    progress.on_log(None, state, control, logs={"loss": 0.25, "reward": 0.5})

    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == "running"
    assert status["stage"] == "training"
    assert status["current_step"] == 1
    assert status["total_steps"] == 4
    assert status["steps_remaining"] == 3
    assert status["progress_fraction"] == pytest.approx(0.25)
    assert status["seconds_per_step"] >= 2.0
    assert status["estimated_seconds_remaining"] >= 6.0
    assert status["last_metrics"]["reward"] == 0.5

    progress.begin_evaluation(
        split="test",
        checkpoint="checkpoint-1",
        step=1,
        completed=0,
        total=2,
    )
    assert progress._evaluation_started is not None
    progress._evaluation_started -= 2.0
    progress.update_evaluation(1, 2)
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["stage"] == "clean_evaluation_test"
    assert status["eta_scope"] == "training_plus_current_evaluation"
    assert status["evaluation"]["completed"] == 1
    assert status["evaluation_estimated_seconds_remaining"] >= 2.0
    assert status["estimated_seconds_remaining"] >= 8.0

    progress.mark_complete(primary_metric={"mean": 1.0}, resource_report={"gpu": {}})
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == "complete"
    assert status["epoch"] == pytest.approx(0.25)
    assert status["estimated_seconds_remaining"] == 0.0


def test_evaluation_schedules_and_callback_force_save():
    assert evaluation_steps(100, 0.1) == tuple(range(10, 101, 10))
    assert fraction_steps(100, [0.9, 1.0]) == (90, 100)
    callback = ScheduledCleanEvalCallback(
        validation_steps={10, 20},
        test_steps={20},
        evaluate=lambda *args: {},  # noqa: ARG005
        event_path="unused.jsonl",
        context=_context(),
    )
    control = SimpleNamespace(should_save=False)
    callback.on_step_end(None, SimpleNamespace(global_step=10), control)
    assert control.should_save is True


def test_primary_metric_averages_last_evaluations_by_difficulty(tmp_path):
    for step, overall, hard in ((90, 0.4, 0.2), (100, 0.6, 0.4)):
        path = tmp_path / "evaluations" / "test" / f"step-{step}"
        path.mkdir(parents=True)
        (path / "summary.json").write_text(
            json.dumps(
                {
                    "step": step,
                    "pass_at_1": overall,
                    "by_difficulty": {"hard": {"n_problems": 10, "pass_at_1": hard}},
                }
            )
        )

    summary = summarize_primary_metric(
        tmp_path,
        split_name="test",
        metric="pass_at_1",
        last_n=2,
        require_count=True,
    )
    assert summary["mean"] == pytest.approx(0.5)
    assert summary["by_difficulty"]["hard"]["mean"] == pytest.approx(0.3)


def test_load_difficulty_map_validates_coverage(tmp_path):
    path = tmp_path / "difficulty.json"
    assert load_difficulty_map(path, allow_missing=True) == {}
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_difficulty_map(path)

    path.write_text(json.dumps({"1": {"stratum": "hard"}, "2": {"stratum": "easy"}}))
    assert load_difficulty_map(path, required_task_ids=[1, 2]) == {1: "hard", 2: "easy"}
    with pytest.raises(ValueError, match="missing"):
        load_difficulty_map(path, required_task_ids=[1, 2, 3])
    with pytest.raises(ValueError, match="metadata does not exist"):
        load_difficulty_map(path, require_metadata=True)

    metadata = {
        "schema_version": 1,
        "model": "test-model",
        "model_revision": "abc123",
        "split": "test",
        "group_size": 16,
    }
    (tmp_path / "difficulty_metadata.json").write_text(json.dumps(metadata))
    assert load_difficulty_map(
        path,
        expected_model="test-model",
        expected_revision="abc123",
        expected_split="test",
        expected_group_size=16,
        require_metadata=True,
    ) == {1: "hard", 2: "easy"}
    with pytest.raises(ValueError, match="metadata mismatch"):
        load_difficulty_map(path, expected_revision="different", require_metadata=True)
