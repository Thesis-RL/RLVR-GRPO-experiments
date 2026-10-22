#!/usr/bin/env python3
"""Run one configured condition matrix sequentially with an aggregate live status."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rlvr.config import ConfigError, load_experiment_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/priority_bridge.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--conditions", nargs="+", help="Optional ordered condition subset")
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.poll_seconds <= 0:
        raise ConfigError("--poll-seconds must be positive")
    experiment = load_experiment_config(args.config)
    seed = experiment.require_seed(args.seed)
    available = [condition.name for condition in experiment.conditions]
    conditions = list(args.conditions or available)
    profile = "smoke" if args.smoke else "main"
    out_root = args.out_root or (
        ROOT / "outputs" / "online_matrix" / experiment.name / profile / f"seed-{seed}"
    )
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    status_path = out_root / "matrix_status.json"
    started_at = datetime.now(UTC)
    completed: list[str] = []
    durations: dict[str, float] = {}
    child: subprocess.Popen[Any] | None = None
    current_git_commit = _git_commit(ROOT)
    active_condition: str | None = None
    active_index: int | None = None
    active_run_dir: Path | None = None
    active_status: dict[str, Any] = {}

    _write_matrix_status(
        status_path,
        status="initializing",
        experiment=experiment.name,
        profile=profile,
        seed=seed,
        conditions=conditions,
        completed=completed,
        durations=durations,
        current_condition=None,
        current_index=None,
        current_run_dir=None,
        current={},
        started_at=started_at,
    )

    try:
        unknown = [condition for condition in conditions if condition not in available]
        if unknown:
            raise ConfigError(f"Unknown conditions: {unknown}; choose from {available}")
        if len(conditions) != len(set(conditions)):
            raise ConfigError("--conditions must not contain duplicates")

        for index, condition in enumerate(conditions):
            run_dir = out_root / condition
            existing = _read_json(run_dir / "status.json")
            active_condition = condition
            active_index = index
            active_run_dir = run_dir
            active_status = existing
            completed_duration = _completed_run_duration(
                run_dir,
                status=existing,
                resume=args.resume,
                experiment=experiment.name,
                profile=profile,
                condition=condition,
                seed=seed,
                current_git_commit=current_git_commit,
            )
            if completed_duration is not None:
                completed.append(condition)
                durations[condition] = completed_duration
                continue

            command = [
                sys.executable,
                str(ROOT / "scripts/run_online_grpo.py"),
                "--config",
                str(args.config.resolve()),
                "--condition",
                condition,
                "--seed",
                str(seed),
                "--out-dir",
                str(run_dir),
            ]
            if args.smoke:
                command.append("--smoke")
            if run_dir.exists() and any(run_dir.iterdir()):
                if args.resume and _has_checkpoint(run_dir):
                    command.extend(["--resume", "latest"])
                else:
                    raise ConfigError(
                        f"Run directory is not empty: {run_dir}. "
                        "Use --resume when it contains a valid checkpoint."
                    )

            child = subprocess.Popen(command, cwd=ROOT)
            while child.poll() is None:
                current = _read_json(run_dir / "status.json")
                active_status = current
                _write_matrix_status(
                    status_path,
                    status="running",
                    experiment=experiment.name,
                    profile=profile,
                    seed=seed,
                    conditions=conditions,
                    completed=completed,
                    durations=durations,
                    current_condition=condition,
                    current_index=index,
                    current_run_dir=run_dir,
                    current=current,
                    started_at=started_at,
                )
                time.sleep(args.poll_seconds)

            return_code = child.wait()
            current = _read_json(run_dir / "status.json")
            active_status = current
            if return_code != 0 or current.get("status") != "complete":
                _write_matrix_status(
                    status_path,
                    status="failed",
                    experiment=experiment.name,
                    profile=profile,
                    seed=seed,
                    conditions=conditions,
                    completed=completed,
                    durations=durations,
                    current_condition=condition,
                    current_index=index,
                    current_run_dir=run_dir,
                    current=current,
                    started_at=started_at,
                    error=f"Condition exited with code {return_code}",
                )
                raise SystemExit(return_code or 1)
            completed.append(condition)
            durations[condition] = float(current.get("elapsed_seconds", 0.0))

        _write_matrix_status(
            status_path,
            status="complete",
            experiment=experiment.name,
            profile=profile,
            seed=seed,
            conditions=conditions,
            completed=completed,
            durations=durations,
            current_condition=None,
            current_index=None,
            current_run_dir=None,
            current={},
            started_at=started_at,
        )
    except KeyboardInterrupt:
        if child is not None and child.poll() is None:
            child.terminate()
        _write_matrix_status(
            status_path,
            status="interrupted",
            experiment=experiment.name,
            profile=profile,
            seed=seed,
            conditions=conditions,
            completed=completed,
            durations=durations,
            current_condition=active_condition,
            current_index=active_index,
            current_run_dir=active_run_dir,
            current=active_status,
            started_at=started_at,
            error="Interrupted by user",
        )
        raise
    except Exception as exc:
        _write_matrix_status(
            status_path,
            status="failed",
            experiment=experiment.name,
            profile=profile,
            seed=seed,
            conditions=conditions,
            completed=completed,
            durations=durations,
            current_condition=active_condition,
            current_index=active_index,
            current_run_dir=active_run_dir,
            current=active_status,
            started_at=started_at,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


def _write_matrix_status(
    path: Path,
    *,
    status: str,
    experiment: str,
    profile: str,
    seed: int,
    conditions: list[str],
    completed: list[str],
    durations: dict[str, float],
    current_condition: str | None,
    current_index: int | None,
    current_run_dir: Path | None,
    current: dict[str, Any],
    started_at: datetime,
    error: str | None = None,
) -> None:
    now = datetime.now(UTC)
    pending = [condition for condition in conditions if condition not in completed]
    if current_condition in pending:
        pending.remove(current_condition)
    current_eta = _number(current.get("estimated_seconds_remaining"))
    completed_durations = [seconds for seconds in durations.values() if seconds > 0]
    current_total_estimate = None
    current_elapsed = _number(current.get("elapsed_seconds"))
    if current_elapsed is not None and current_eta is not None:
        current_total_estimate = current_elapsed + current_eta
    typical_duration = (
        sum(completed_durations) / len(completed_durations)
        if completed_durations
        else current_total_estimate
    )
    estimated_remaining = None
    eta_scope = None
    if status == "complete":
        estimated_remaining = 0.0
        eta_scope = "matrix"
    elif current_eta is not None and typical_duration is not None:
        estimated_remaining = current_eta + typical_duration * len(pending)
        current_scope = current.get("eta_scope") or "partial_current_run"
        eta_scope = (
            f"current_{current_scope}_plus_completed_run_average_for_pending"
            if completed_durations
            else f"current_{current_scope}_projection_for_all_conditions"
        )
    estimated_finished_at = (
        (now + timedelta(seconds=estimated_remaining)).isoformat()
        if estimated_remaining is not None
        else None
    )
    payload = {
        "schema_version": 1,
        "kind": "condition_matrix",
        "status": status,
        "updated_at_utc": now.isoformat(),
        "started_at_utc": started_at.isoformat(),
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "experiment": experiment,
        "profile": profile,
        "seed": seed,
        "conditions": conditions,
        "completed_conditions": completed,
        "completed_count": len(completed),
        "total_conditions": len(conditions),
        "pending_conditions": pending,
        "current_condition": current_condition,
        "current_condition_index": current_index,
        "current_run_dir": str(current_run_dir) if current_run_dir is not None else None,
        "current_run": current,
        "condition_durations_seconds": durations,
        "elapsed_seconds": (now - started_at).total_seconds(),
        "estimated_seconds_remaining": estimated_remaining,
        "eta_scope": eta_scope,
        "estimated_finished_at_utc": estimated_finished_at,
        "error": error,
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _has_checkpoint(run_dir: Path) -> bool:
    return any((path / "trainer_state.json").is_file() for path in run_dir.glob("checkpoint-*"))


def _completed_run_duration(
    run_dir: Path,
    *,
    status: dict[str, Any],
    resume: bool,
    experiment: str,
    profile: str,
    condition: str,
    seed: int,
    current_git_commit: str | None,
) -> float | None:
    if status.get("status") != "complete":
        return None
    if not resume:
        raise ConfigError(
            f"Completed run already exists: {run_dir}. Use --resume to reuse it explicitly."
        )

    expected = {
        "experiment": experiment,
        "profile": profile,
        "condition": condition,
        "seed": seed,
    }
    mismatches = {
        key: (status.get(key), value) for key, value in expected.items() if status.get(key) != value
    }
    if mismatches:
        raise ConfigError(f"Completed run status does not match the matrix: {mismatches}")

    metadata = _read_json(run_dir / "run_metadata.json")
    context = metadata.get("context")
    if not isinstance(context, dict):
        raise ConfigError(f"Completed run is missing valid metadata: {run_dir}")
    context_mismatches = {
        key: (context.get(key), value)
        for key, value in expected.items()
        if context.get(key) != value
    }
    if context_mismatches:
        raise ConfigError(f"Completed run metadata does not match the matrix: {context_mismatches}")

    environment = metadata.get("environment")
    git_state = environment.get("git") if isinstance(environment, dict) else None
    saved_commit = git_state.get("commit") if isinstance(git_state, dict) else None
    if not saved_commit or not current_git_commit:
        raise ConfigError("Completed run reuse requires both a recorded and current Git commit")
    if saved_commit != current_git_commit:
        raise ConfigError(
            f"Completed run Git commit differs from the current commit: "
            f"{saved_commit} != {current_git_commit}"
        )
    return _number(status.get("elapsed_seconds")) or 0.0


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None


if __name__ == "__main__":
    try:
        main()
    except ConfigError as exc:
        raise SystemExit(f"configuration error: {exc}") from exc
