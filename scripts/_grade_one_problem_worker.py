#!/usr/bin/env python3
"""Isolated worker: grade codes for one MBPP problem under hard resource caps.

Designed so a single runaway candidate cannot OOM the host:
  - RLIMIT_AS (virtual address space)
  - RLIMIT_DATA / RLIMIT_RSS when available
  - RLIMIT_NPROC = 1 (no fork bombs)
  - RLIMIT_CPU
"""

from __future__ import annotations

import contextlib
import io
import json
import resource
import signal
import sys


class _Timeout(Exception):
    pass


def _alarm(signum, frame):  # noqa: ANN001, ARG001
    raise _Timeout()


def _set_limits(mem_bytes: int, cpu_s: int) -> None:
    # Address space / data segment — primary defense against huge allocations.
    for name in ("RLIMIT_AS", "RLIMIT_DATA", "RLIMIT_RSS"):
        lim = getattr(resource, name, None)
        if lim is None:
            continue
        try:
            resource.setrlimit(lim, (mem_bytes, mem_bytes))
        except (ValueError, OSError):
            try:
                soft, hard = resource.getrlimit(lim)
                resource.setrlimit(lim, (min(mem_bytes, soft), min(mem_bytes, hard)))
            except (ValueError, OSError):
                pass
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
    except (ValueError, OSError):
        pass
    # Core dumps off
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ValueError, OSError):
        pass


@contextlib.contextmanager
def _timed(timeout_s: float):
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


_DENY_SUBSTR = (
    "subprocess",
    "multiprocessing",
    "ctypes",
    "__import__",
    "os.system",
    "os.popen",
    "shutil",
    "pathlib",
    "socket",
    "http.client",
    "urllib",
    "requests",
    "pty",
    "signal",
    "resource",
    "mmap",
    "fcntl",
)


def _looks_dangerous(code: str) -> str | None:
    import re

    low = code.lower()
    for s in _DENY_SUBSTR:
        if s in low:
            return f"blocked:{s}"
    # Only flag allocation bombs, not ordinary sentinels like 10**9 in math.
    if re.search(r"\[\s*[^\]]*\*\s*\(?\s*10\s*\*\*\s*\d{6,}", code):
        return "blocked:huge_list"
    if re.search(r"range\s*\(\s*10\s*\*\*\s*\d{5,}", code):
        return "blocked:huge_range"
    if re.search(r"while\s+True\s*:", code) or re.search(r"while\s+1\s*:", code):
        return "blocked:infinite"
    return None


def grade_one(code: str, setup: str, tests: list[str], timeout_s: float):
    if not code.strip():
        return [0] * len(tests), "empty_code"
    reason = _looks_dangerous(code)
    if reason:
        return [0] * len(tests), reason

    ns: dict = {"__builtins__": __builtins__}
    try:
        with _timed(timeout_s):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                if setup.strip():
                    exec(setup, ns)  # noqa: S102
                exec(code, ns)  # noqa: S102
    except _Timeout:
        return [0] * len(tests), "timeout"
    except MemoryError:
        return [0] * len(tests), "memory"
    except Exception as exc:  # noqa: BLE001
        return [0] * len(tests), f"{type(exc).__name__}: {exc}"

    outs, err = [], None
    for test in tests:
        try:
            with _timed(timeout_s):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    exec(test, ns)  # noqa: S102
            outs.append(1)
        except _Timeout:
            outs.append(0)
            err = "timeout"
        except MemoryError:
            outs.append(0)
            err = "memory"
        except Exception as exc:  # noqa: BLE001
            outs.append(0)
            err = f"{type(exc).__name__}: {exc}"
    return outs, err


def main() -> None:
    payload = json.load(sys.stdin)
    mem = int(payload.get("mem_bytes", 64 * 1024 * 1024))
    timeout_s = float(payload["timeout_s"])
    cpu_s = max(1, int(timeout_s * (1 + len(payload["test_list"])) * len(payload["codes"])) + 2)
    _set_limits(mem, cpu_s)

    setup = payload.get("test_setup_code") or ""
    tests = list(payload["test_list"])
    results = []
    # One code at a time inside the already-capped process.
    for code in payload["codes"]:
        outcomes, err = grade_one(code, setup, tests, timeout_s)
        results.append({"outcomes": outcomes, "error": err})
    json.dump(results, sys.stdout)


if __name__ == "__main__":
    main()
