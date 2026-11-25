from __future__ import annotations

import json
import random

import numpy as np
import pytest
import torch

from rlvr.data.mbpp import MBPPProblem
from rlvr.evaluation.multisample import (
    MultiSampleTarget,
    compare_task_clustered,
    evaluate_multisample_clean,
    read_multisample_predictions,
    sample_chunk_seed,
)
from rlvr.exec.mbpp_executor import ExecResult
from scripts.evaluate_multisample import _validate_checkpoint


class _DummyModel:
    training = True

    def eval(self) -> None:
        self.training = False

    def train(self) -> None:
        self.training = True


def _problem(task_id: int) -> MBPPProblem:
    return MBPPProblem(
        task_id=task_id,
        text="return one",
        code="",
        test_list=["assert answer() == 1"],
        test_setup_code="",
        challenge_test_list=[],
        split="test",
    )


def _target() -> MultiSampleTarget:
    return MultiSampleTarget(
        target_id="base",
        kind="base",
        model="model",
        model_revision="revision",
    )


def test_multisample_evaluation_resumes_and_extends_by_complete_chunks(
    tmp_path, monkeypatch
) -> None:
    sample_calls: list[int] = []

    def fake_sample(*args, n, **kwargs):  # noqa: ANN002, ANN003
        sample_calls.append(n)
        return [f"def answer():\n    return {index % 2}" for index in range(n)]

    def fake_grade(codes, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        return [
            ExecResult(outcomes=(1 if "return 1" in code else 0,), error=None) for code in codes
        ]

    monkeypatch.setattr("rlvr.evaluation.multisample.sample_completions", fake_sample)
    monkeypatch.setattr("rlvr.evaluation.multisample.grade_codes", fake_grade)
    model = _DummyModel()
    common = {
        "few_shot": [],
        "target": _target(),
        "out_dir": tmp_path,
        "generation_seed": 0,
        "generation_chunk_size": 2,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_new_tokens": 32,
        "stop_sequence": "[DONE]",
        "timeout_s": 0.5,
        "timeout_fallback_s": 5.0,
        "mem_limit": 64 * 1024 * 1024,
        "difficulty_by_task": {1: "hard", 2: "easy"},
        "bootstrap_replicates": 100,
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

    first = evaluate_multisample_clean(
        model,
        object(),
        [_problem(1), _problem(2)],
        samples_per_task=4,
        **common,
    )
    second = evaluate_multisample_clean(
        model,
        object(),
        [_problem(1), _problem(2)],
        samples_per_task=4,
        **common,
    )
    extended = evaluate_multisample_clean(
        model,
        object(),
        [_problem(1), _problem(2)],
        samples_per_task=6,
        **common,
    )

    assert first == second
    assert first["n_generations"] == 8
    assert first["pass_at_1"]["estimate"] == pytest.approx(0.5)
    assert extended["n_generations"] == 12
    assert sample_calls == [2, 2, 2, 2, 2, 2]
    assert model.training is True
    assert random.random() == expected_python
    assert np.random.random() == expected_numpy
    assert torch.equal(torch.rand(1), expected_torch)
    manifest = json.loads((tmp_path / "evaluation_manifest.json").read_text())
    assert manifest["max_requested_samples_per_task"] == 6
    predictions = read_multisample_predictions(tmp_path / "predictions.jsonl")
    assert len(predictions) == 12


def test_multisample_reader_repairs_only_a_trailing_partial_line(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"
    row = {"task_id": 1, "sample_index": 0, "protocol_fingerprint": "abc"}
    path.write_bytes((json.dumps(row) + "\n" + '{"task_id":').encode())

    parsed = read_multisample_predictions(
        path,
        protocol_fingerprint="abc",
        repair_trailing_partial=True,
    )

    assert parsed == {(1, 0): row}
    assert path.read_text() == json.dumps(row) + "\n"


def test_paired_task_cluster_comparison_uses_task_means() -> None:
    base = [
        {
            "task_id": task,
            "sample_index": sample,
            "clean_binary_reward": 0.0,
            "clean_fractional_reward": 0.25,
        }
        for task in (1, 2)
        for sample in range(2)
    ]
    trained = [
        {
            "task_id": task,
            "sample_index": sample,
            "clean_binary_reward": float(task == 1),
            "clean_fractional_reward": 0.75,
        }
        for task in (1, 2)
        for sample in range(2)
    ]

    comparison = compare_task_clustered(
        base,
        trained,
        bootstrap_replicates=200,
        bootstrap_seed=7,
    )

    assert comparison["pass_at_1"]["base"]["estimate"] == 0.0
    assert comparison["pass_at_1"]["trained"]["estimate"] == 0.5
    assert comparison["pass_at_1"]["delta_trained_minus_base"]["estimate"] == 0.5
    assert comparison["mean_test_pass_rate"]["delta_trained_minus_base"][
        "estimate"
    ] == pytest.approx(0.5)


def test_chunk_seed_is_stable_and_changes_across_chunks() -> None:
    assert sample_chunk_seed(0, 11, 0) == sample_chunk_seed(0, 11, 0)
    assert sample_chunk_seed(0, 11, 0) != sample_chunk_seed(0, 11, 4)


def test_checkpoint_validation_accepts_equivalent_target_module_order(tmp_path) -> None:
    checkpoint = tmp_path / "clean__fractional" / "checkpoint-337"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
    (checkpoint / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "model",
                "peft_type": "LORA",
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["v_proj", "q_proj"],
            }
        )
    )

    _validate_checkpoint(
        checkpoint,
        condition="clean__fractional",
        step=337,
        model="model",
        model_revision="revision",
    )
