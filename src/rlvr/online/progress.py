"""Atomic, queryable progress snapshots for long-running online experiments."""

from __future__ import annotations

import json
import os
import socket
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from rlvr.run_logging import RunContext


class RunProgressCallback(TrainerCallback):
    """Persist live training and evaluation progress to ``status.json``."""

    def __init__(
        self,
        out_dir: str | Path,
        *,
        context: RunContext,
        total_steps: int,
        resume_checkpoint: str | bool | None = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.path = self.out_dir / "status.json"
        self.context = context
        self.total_steps = int(total_steps)
        self.resume_checkpoint = str(resume_checkpoint) if resume_checkpoint else None
        self._attempt_started = time.perf_counter()
        self._last_step_recorded_at = self._attempt_started
        self._last_step = 0
        self._observed_step_seconds = 0.0
        self._observed_steps = 0
        self._previous = self._read_previous()
        self._elapsed_before_attempt = float(self._previous.get("elapsed_seconds", 0.0))
        self._last_step = int(self._previous.get("current_step", 0))
        previous_rate = self._previous.get("seconds_per_step")
        self._previous_seconds_per_step = (
            float(previous_rate) if isinstance(previous_rate, int | float) else None
        )
        self._started_at_utc = str(
            self._previous.get("started_at_utc") or datetime.now(UTC).isoformat()
        )
        self._last_checkpoint = self._previous.get("last_checkpoint")
        self._last_metrics: dict[str, Any] = {}
        previous_epoch = self._previous.get("epoch")
        self._last_epoch = (
            float(previous_epoch) if isinstance(previous_epoch, int | float) else None
        )
        self._evaluation: dict[str, Any] | None = None
        self._evaluation_started: float | None = None
        self._evaluation_initial_completed = 0
        self._write(status="running", stage="initializing")

    def mark_stage(self, stage: str) -> None:
        self._evaluation = None
        self._evaluation_started = None
        if stage == "training":
            self._last_step_recorded_at = time.perf_counter()
        self._write(status="running", stage=stage)

    def begin_evaluation(
        self,
        *,
        split: str,
        checkpoint: str,
        step: int,
        completed: int,
        total: int,
    ) -> None:
        self._evaluation_started = time.perf_counter()
        self._evaluation_initial_completed = completed
        self._evaluation = {
            "split": split,
            "checkpoint": checkpoint,
            "step": int(step),
            "completed": int(completed),
            "total": int(total),
        }
        self._write(status="running", stage=f"clean_evaluation_{split}")

    def update_evaluation(self, completed: int, total: int) -> None:
        if self._evaluation is None:
            return
        self._evaluation["completed"] = int(completed)
        self._evaluation["total"] = int(total)
        self._write(status="running", stage=f"clean_evaluation_{self._evaluation['split']}")

    def mark_complete(
        self,
        *,
        primary_metric: dict[str, Any] | None,
        resource_report: dict[str, Any],
    ) -> None:
        self._evaluation = None
        self._write(
            status="complete",
            stage="complete",
            extra={
                "primary_metric": primary_metric,
                "resource_report": {
                    "total_wall_seconds": resource_report.get("total_wall_seconds"),
                    "gpu": resource_report.get("gpu"),
                },
            },
        )

    def mark_failed(self, error: str) -> None:
        self._write(status="failed", stage="failed", extra={"error": error})

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ANN001, ANN201, ARG002
        self._last_step = int(state.global_step)
        self._last_step_recorded_at = time.perf_counter()
        self._write(status="running", stage="training", state=state)
        return control

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001, ANN201, ARG002
        now = time.perf_counter()
        current_step = int(state.global_step)
        step_delta = current_step - self._last_step
        if step_delta > 0:
            self._observed_step_seconds += now - self._last_step_recorded_at
            self._observed_steps += step_delta
            self._last_step = current_step
            self._last_step_recorded_at = now
        self._write(status="running", stage="training", state=state)
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001, ANN201, ARG002
        if logs:
            self._last_metrics = _json_safe(dict(logs))
        self._write(status="running", stage="training", state=state)
        return control

    def on_save(self, args, state, control, **kwargs):  # noqa: ANN001, ANN201, ARG002
        self._last_checkpoint = str(Path(args.output_dir) / f"checkpoint-{state.global_step}")
        self._write(status="running", stage="training", state=state)
        return control

    def on_train_end(self, args, state, control, **kwargs):  # noqa: ANN001, ANN201, ARG002
        self._write(status="running", stage="training_complete", state=state)
        return control

    def _write(
        self,
        *,
        status: str,
        stage: str,
        state: Any | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        now = datetime.now(UTC)
        elapsed = self._elapsed_before_attempt + (time.perf_counter() - self._attempt_started)
        current_step = int(state.global_step) if state is not None else self._last_step
        if state is not None and state.epoch is not None:
            self._last_epoch = float(state.epoch)
        total_steps = max(self.total_steps, current_step)
        steps_remaining = max(0, total_steps - current_step)
        seconds_per_step = self._seconds_per_step()
        training_eta = seconds_per_step * steps_remaining if seconds_per_step is not None else None
        evaluation_eta = None
        eta_seconds = training_eta
        eta_scope = "training" if training_eta is not None else None

        evaluation = dict(self._evaluation) if self._evaluation is not None else None
        if evaluation is not None:
            evaluation_eta = self._evaluation_eta_seconds(evaluation)
            if steps_remaining == 0:
                eta_seconds = evaluation_eta
                eta_scope = "current_evaluation" if evaluation_eta is not None else None
            elif training_eta is not None and evaluation_eta is not None:
                eta_seconds = training_eta + evaluation_eta
                eta_scope = "training_plus_current_evaluation"
            else:
                eta_seconds = None
                eta_scope = None

        if status == "complete":
            eta_seconds = 0.0
            eta_scope = "run"
        elif status == "failed":
            eta_seconds = None
            eta_scope = None

        estimated_finished_at = (
            (now + timedelta(seconds=eta_seconds)).isoformat() if eta_seconds is not None else None
        )
        payload: dict[str, Any] = {
            "schema_version": 1,
            "status": status,
            "stage": stage,
            "updated_at_utc": now.isoformat(),
            "started_at_utc": self._started_at_utc,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "run_id": self.context.run_id,
            "experiment": self.context.experiment,
            "condition": self.context.condition,
            "seed": self.context.seed,
            "profile": self.context.profile,
            "current_step": current_step,
            "total_steps": total_steps,
            "steps_remaining": steps_remaining,
            "progress_fraction": current_step / total_steps if total_steps else None,
            "epoch": self._last_epoch,
            "elapsed_seconds": elapsed,
            "seconds_per_step": seconds_per_step,
            "training_estimated_seconds_remaining": training_eta,
            "evaluation_estimated_seconds_remaining": evaluation_eta,
            "estimated_seconds_remaining": eta_seconds,
            "eta_scope": eta_scope,
            "estimated_finished_at_utc": estimated_finished_at,
            "last_checkpoint": self._last_checkpoint,
            "resume_checkpoint": self.resume_checkpoint,
            "last_metrics": self._last_metrics,
            "evaluation": evaluation,
        }
        if extra:
            payload.update(_json_safe(extra))
        _atomic_write_json(self.path, payload)

    def _seconds_per_step(self) -> float | None:
        if self._observed_steps:
            return self._observed_step_seconds / self._observed_steps
        return self._previous_seconds_per_step

    def _evaluation_eta_seconds(self, evaluation: dict[str, Any]) -> float | None:
        if self._evaluation_started is None:
            return None
        completed = int(evaluation["completed"])
        total = int(evaluation["total"])
        observed = completed - self._evaluation_initial_completed
        if observed <= 0:
            return None
        seconds_per_problem = (time.perf_counter() - self._evaluation_started) / observed
        evaluation["seconds_per_problem"] = seconds_per_problem
        evaluation["progress_fraction"] = completed / total if total else None
        return seconds_per_problem * max(0, total - completed)

    def _read_previous(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            value = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_json_safe(payload), indent=2) + "\n")
    temporary.replace(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, int | float | str | bool) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)
