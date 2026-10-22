"""Safety tests for condition-matrix reuse and ETA reporting."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rlvr.config import ConfigError
from scripts.run_online_matrix import _completed_run_duration, _write_matrix_status

ROOT = Path(__file__).resolve().parents[1]


def _write_metadata(run_dir, *, condition="clean__fractional", commit="abc123"):
    metadata = {
        "context": {
            "experiment": "priority_bridge",
            "profile": "main",
            "condition": condition,
            "seed": 0,
        },
        "environment": {"git": {"commit": commit, "dirty": True}},
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata))


def _complete_status(*, condition="clean__fractional"):
    return {
        "status": "complete",
        "experiment": "priority_bridge",
        "profile": "main",
        "condition": condition,
        "seed": 0,
        "elapsed_seconds": 12.5,
    }


def test_completed_matrix_run_requires_explicit_resume_and_matching_identity(tmp_path):
    run_dir = tmp_path / "clean__fractional"
    run_dir.mkdir()
    _write_metadata(run_dir)
    status = _complete_status()

    with pytest.raises(ConfigError, match="Use --resume"):
        _completed_run_duration(
            run_dir,
            status=status,
            resume=False,
            experiment="priority_bridge",
            profile="main",
            condition="clean__fractional",
            seed=0,
            current_git_commit="abc123",
        )

    assert (
        _completed_run_duration(
            run_dir,
            status=status,
            resume=True,
            experiment="priority_bridge",
            profile="main",
            condition="clean__fractional",
            seed=0,
            current_git_commit="abc123",
        )
        == 12.5
    )

    with pytest.raises(ConfigError, match="status does not match"):
        _completed_run_duration(
            run_dir,
            status=status,
            resume=True,
            experiment="priority_bridge",
            profile="main",
            condition="fp_only_030__fractional",
            seed=0,
            current_git_commit="abc123",
        )


def test_completed_matrix_run_rejects_different_commit(tmp_path):
    run_dir = tmp_path / "clean__fractional"
    run_dir.mkdir()
    _write_metadata(run_dir, commit="old")

    with pytest.raises(ConfigError, match="Git commit differs"):
        _completed_run_duration(
            run_dir,
            status=_complete_status(),
            resume=True,
            experiment="priority_bridge",
            profile="main",
            condition="clean__fractional",
            seed=0,
            current_git_commit="new",
        )

    _write_metadata(run_dir, commit=None)
    with pytest.raises(ConfigError, match="recorded and current Git commit"):
        _completed_run_duration(
            run_dir,
            status=_complete_status(),
            resume=True,
            experiment="priority_bridge",
            profile="main",
            condition="clean__fractional",
            seed=0,
            current_git_commit="new",
        )


def test_matrix_eta_reports_its_projection_scope(tmp_path):
    path = tmp_path / "matrix_status.json"
    _write_matrix_status(
        path,
        status="running",
        experiment="priority_bridge",
        profile="main",
        seed=0,
        conditions=["clean__fractional", "fp_only_030__fractional"],
        completed=[],
        durations={},
        current_condition="clean__fractional",
        current_index=0,
        current_run_dir=tmp_path / "clean__fractional",
        current={
            "elapsed_seconds": 10.0,
            "estimated_seconds_remaining": 20.0,
            "eta_scope": "training",
        },
        started_at=datetime.now(UTC),
    )

    status = json.loads(path.read_text())
    assert status["estimated_seconds_remaining"] == 50.0
    assert status["eta_scope"] == "current_training_projection_for_all_conditions"


def test_matrix_replaces_stale_status_when_preflight_fails(tmp_path):
    status_path = tmp_path / "matrix_status.json"
    status_path.write_text(json.dumps({"status": "complete", "error": None}))

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_online_matrix.py"),
            "--config",
            str(ROOT / "configs/priority_bridge.yaml"),
            "--seed",
            "0",
            "--conditions",
            "not_a_condition",
            "--out-root",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode != 0
    assert "Unknown conditions" in result.stderr
    status = json.loads(status_path.read_text())
    assert status["status"] == "failed"
    assert status["current_condition"] is None
    assert status["error"].startswith("ConfigError: Unknown conditions")


def test_matrix_records_condition_context_when_run_directory_is_invalid(tmp_path):
    run_dir = tmp_path / "clean__fractional"
    run_dir.mkdir()
    (run_dir / "unexpected.txt").write_text("stale")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_online_matrix.py"),
            "--config",
            str(ROOT / "configs/priority_bridge.yaml"),
            "--seed",
            "0",
            "--conditions",
            "clean__fractional",
            "--out-root",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode != 0
    status = json.loads((tmp_path / "matrix_status.json").read_text())
    assert status["status"] == "failed"
    assert status["current_condition"] == "clean__fractional"
    assert status["current_condition_index"] == 0
    assert status["current_run_dir"] == str(run_dir)
    assert "Run directory is not empty" in status["error"]
