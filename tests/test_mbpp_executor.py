"""Regression tests for the clean MBPP verifier."""

from __future__ import annotations

import pytest

from rlvr.data.mbpp import MBPPProblem
from rlvr.exec import mbpp_executor
from rlvr.exec.mbpp_executor import ExecResult, ExecutorInfrastructureError, grade_code


def _problem(*tests: str, setup: str = "") -> MBPPProblem:
    return MBPPProblem(
        task_id=1,
        text="",
        code="",
        test_list=list(tests),
        test_setup_code=setup,
        challenge_test_list=[],
        split="test",
    )


def test_identifier_containing_pty_is_not_blocked() -> None:
    result = grade_code(
        "def empty(items):\n    return len(items) == 0",
        _problem("assert empty([])", "assert not empty([1])"),
    )
    assert result.outcomes == (1, 1)
    assert result.error is None


def test_finite_while_true_is_allowed_but_infinite_loop_times_out() -> None:
    finite = grade_code(
        "def first():\n    while True:\n        return 1",
        _problem("assert first() == 1"),
    )
    assert finite.outcomes == (1,)

    infinite = grade_code(
        "while True:\n    pass",
        _problem("assert True"),
        timeout_s=0.01,
        timeout_fallback_s=0.02,
    )
    assert infinite.outcomes == (0,)
    assert infinite.error == "timeout"


def test_candidate_is_loaded_before_setup_code() -> None:
    result = grade_code(
        "class Node:\n    def __init__(self, value):\n        self.value = value",
        _problem("assert root.value == 7", setup="root = Node(7)"),
    )
    assert result.outcomes == (1,)


@pytest.mark.parametrize(
    "code,blocked",
    [
        ("import pty", "blocked:import:pty"),
        ("import os as safe_name\nsafe_name.system('true')", "blocked:call:os.system"),
        ("from os import remove as erase\nerase('x')", "blocked:call:os.remove"),
        ("eval(\"__import__('os').system('true')\")", "blocked:call:eval"),
        ("open('x')", "blocked:call:open"),
    ],
)
def test_ast_filter_blocks_actual_dangerous_operations(code: str, blocked: str) -> None:
    result = grade_code(code, _problem("assert True"))
    assert result.outcomes == (0,)
    assert result.error == blocked


def test_partial_solution_keeps_per_test_outcomes() -> None:
    result = grade_code(
        "def identity(value):\n    return 1",
        _problem("assert identity(1) == 1", "assert identity(2) == 2"),
    )
    assert result.outcomes == (1, 0)


def test_worker_protocol_failure_raises_instead_of_becoming_zero(monkeypatch) -> None:
    monkeypatch.setattr(
        mbpp_executor,
        "_grade_one_subprocess",
        lambda *args, **kwargs: ExecResult((0,), "worker_crash"),
    )
    with pytest.raises(ExecutorInfrastructureError, match="failed twice"):
        grade_code("def ok(): return True", _problem("assert ok()"))
