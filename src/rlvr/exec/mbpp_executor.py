"""Clean MBPP executor: one memory-capped subprocess per candidate code."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rlvr.data.mbpp import MBPPProblem

_WORKER = Path(__file__).resolve().parents[3] / "scripts" / "_grade_one_problem_worker.py"
# Keep tiny: clean worker + short MBPP snippets. Prevents host OOM.
_MEM_LIMIT = 256 * 1024 * 1024
_MIN_AVAILABLE_MB = 2048
_INFRASTRUCTURE_ERRORS = frozenset({"worker_crash", "worker_bad_json"})


class ExecutorInfrastructureError(RuntimeError):
    """The verifier infrastructure failed; this is not a candidate failure."""


@dataclass(frozen=True)
class ExecResult:
    outcomes: tuple[int, ...]
    error: str | None = None


def _available_mb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        return 99999.0
    return 99999.0


def grade_code(
    code: str,
    problem: MBPPProblem,
    *,
    timeout_s: float = 0.5,
    timeout_fallback_s: float = 5.0,
    tests: Sequence[str] | None = None,
    mem_limit: int = _MEM_LIMIT,
) -> ExecResult:
    return grade_codes(
        [code],
        problem,
        timeout_s=timeout_s,
        timeout_fallback_s=timeout_fallback_s,
        tests=tests,
        mem_limit=mem_limit,
    )[0]


def grade_codes(
    codes: Sequence[str],
    problem: MBPPProblem,
    *,
    timeout_s: float = 0.5,
    timeout_fallback_s: float = 5.0,
    tests: Sequence[str] | None = None,
    mem_limit: int = _MEM_LIMIT,
) -> list[ExecResult]:
    """Grade each candidate in its own subprocess (never parallel).

    A timeout gets one fresh retry with the larger fallback budget. Worker
    protocol failures get one same-budget retry and then raise, so an
    infrastructure failure cannot silently become a zero reward.
    """
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if timeout_fallback_s < timeout_s:
        raise ValueError("timeout_fallback_s must be at least timeout_s")
    if mem_limit <= 0:
        raise ValueError("mem_limit must be positive")
    test_list = list(tests) if tests is not None else list(problem.test_list)
    out: list[ExecResult] = []
    for code in codes:
        if _available_mb() < _MIN_AVAILABLE_MB:
            raise ExecutorInfrastructureError(
                "Verifier refused to start because host memory is below the safety floor"
            )
        result = _grade_with_infrastructure_retry(
            code,
            problem.test_setup_code,
            test_list,
            timeout_s,
            mem_limit,
        )
        if result.error == "timeout" and timeout_fallback_s > timeout_s:
            result = _grade_with_infrastructure_retry(
                code,
                problem.test_setup_code,
                test_list,
                timeout_fallback_s,
                mem_limit,
            )
        out.append(result)
    return out


def _grade_with_infrastructure_retry(
    code: str,
    setup: str,
    test_list: list[str],
    timeout_s: float,
    mem_limit: int,
) -> ExecResult:
    result = _grade_one_subprocess(code, setup, test_list, timeout_s, mem_limit)
    if result.error in _INFRASTRUCTURE_ERRORS:
        result = _grade_one_subprocess(code, setup, test_list, timeout_s, mem_limit)
    if result.error in _INFRASTRUCTURE_ERRORS:
        raise ExecutorInfrastructureError(f"Verifier infrastructure failed twice: {result.error}")
    return result


def _grade_one_subprocess(
    code: str,
    setup: str,
    test_list: list[str],
    timeout_s: float,
    mem_limit: int,
) -> ExecResult:
    payload = {
        "codes": [code],
        "test_setup_code": setup,
        "test_list": test_list,
        "timeout_s": float(timeout_s),
        "mem_bytes": int(mem_limit),
    }
    wall = timeout_s * (1 + len(test_list)) + 3.0
    try:
        with tempfile.TemporaryDirectory(prefix="rlvr-exec-") as work_dir:
            # Never inherit Hugging Face, Git, cloud, or other user credentials.
            env = {
                "HOME": work_dir,
                "TMPDIR": work_dir,
                "PATH": os.defpath,
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PYTHONHASHSEED": "0",
                "PYTHONMALLOC": "malloc",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
            proc = subprocess.run(
                [sys.executable, "-I", str(_WORKER)],  # -I: isolated, no user site
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=wall,
                check=False,
                env=env,
                cwd=work_dir,
                close_fds=True,
                start_new_session=True,
            )
    except subprocess.TimeoutExpired:
        return ExecResult(outcomes=tuple(0 for _ in test_list), error="timeout")

    if proc.returncode != 0 or not proc.stdout.strip():
        # Nonzero often means killed by RLIMIT / SIGKILL from OOM in cgroup.
        err = "worker_crash"
        if proc.returncode in (-9, 137):
            err = "killed"
        return ExecResult(outcomes=tuple(0 for _ in test_list), error=err)

    try:
        rows = json.loads(proc.stdout)
        if not isinstance(rows, list) or len(rows) != 1:
            raise ValueError("worker returned an invalid row count")
        r = rows[0]
        outcomes = tuple(int(x) for x in r["outcomes"])
        if len(outcomes) != len(test_list) or any(value not in (0, 1) for value in outcomes):
            raise ValueError("worker returned invalid outcomes")
        return ExecResult(
            outcomes=outcomes,
            error=r.get("error"),
        )
    except Exception:  # noqa: BLE001
        return ExecResult(outcomes=tuple(0 for _ in test_list), error="worker_bad_json")
