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
_MEM_LIMIT = 64 * 1024 * 1024
_MIN_AVAILABLE_MB = 2048


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
    tests: Sequence[str] | None = None,
    mem_limit: int = _MEM_LIMIT,
) -> ExecResult:
    return grade_codes([code], problem, timeout_s=timeout_s, tests=tests, mem_limit=mem_limit)[0]


def grade_codes(
    codes: Sequence[str],
    problem: MBPPProblem,
    *,
    timeout_s: float = 0.5,
    tests: Sequence[str] | None = None,
    mem_limit: int = _MEM_LIMIT,
) -> list[ExecResult]:
    """Grade each candidate in its own subprocess (never parallel)."""
    test_list = list(tests) if tests is not None else list(problem.test_list)
    out: list[ExecResult] = []
    for code in codes:
        if _available_mb() < _MIN_AVAILABLE_MB:
            # Refuse to run rather than risk another host reboot.
            out.append(
                ExecResult(
                    outcomes=tuple(0 for _ in test_list),
                    error="skipped_low_host_memory",
                )
            )
            continue
        out.append(
            _grade_one_subprocess(code, problem.test_setup_code, test_list, timeout_s, mem_limit)
        )
    return out


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
        r = rows[0]
        return ExecResult(
            outcomes=tuple(int(x) for x in r["outcomes"]),
            error=r.get("error"),
        )
    except Exception:  # noqa: BLE001
        return ExecResult(outcomes=tuple(0 for _ in test_list), error="worker_bad_json")
