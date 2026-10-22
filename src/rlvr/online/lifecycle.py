"""Safe creation and resumption of deterministic experiment directories."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from rlvr.config import ConfigError
from rlvr.run_logging import RunContext, append_jsonl


def prepare_run_directory(
    out_dir: str | Path,
    *,
    context: RunContext,
    resolved_config: dict[str, Any],
    resume: str | None,
    current_git_commit: str | None = None,
) -> str | bool | None:
    """Reject accidental reuse and validate an explicitly resumed run."""
    destination = Path(out_dir)
    if resume is None:
        if destination.exists() and any(destination.iterdir()):
            raise ConfigError(
                f"Output directory is not empty: {destination}. "
                "Use --resume for the same run or choose a new --out-dir."
            )
        destination.mkdir(parents=True, exist_ok=True)
        return None

    if not destination.exists():
        raise ConfigError(f"Cannot resume because output directory does not exist: {destination}")
    _validate_existing_run(
        destination,
        context=context,
        resolved_config=resolved_config,
        current_git_commit=current_git_commit,
    )

    checkpoint = _resolve_checkpoint(destination, resume)
    append_jsonl(
        destination / "events.jsonl",
        {
            "event": "resume_requested",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "checkpoint": str(checkpoint),
        },
        context=context,
    )
    return str(checkpoint)


def _validate_existing_run(
    destination: Path,
    *,
    context: RunContext,
    resolved_config: dict[str, Any],
    current_git_commit: str | None,
) -> None:
    metadata_path = destination / "run_metadata.json"
    config_path = destination / "resolved_config.yaml"
    if not metadata_path.exists() or not config_path.exists():
        raise ConfigError(
            f"Cannot resume {destination}: run_metadata.json or resolved_config.yaml is missing"
        )

    metadata = json.loads(metadata_path.read_text())
    existing_context = metadata.get("context")
    expected_context = {
        "pipeline": context.pipeline,
        "experiment": context.experiment,
        "study": context.study,
        "model": context.model,
        "model_role": context.model_role,
        "condition": context.condition,
        "reward": context.reward,
        "noise": context.noise,
        "seed": context.seed,
        "profile": context.profile,
        "model_revision": context.model_revision,
    }
    if existing_context != expected_context:
        raise ConfigError("Resume context does not match the existing run metadata")

    saved_commit = metadata.get("environment", {}).get("git", {}).get("commit")
    if saved_commit and current_git_commit and saved_commit != current_git_commit:
        raise ConfigError(
            f"Git commit changed since the run started: {saved_commit} != {current_git_commit}"
        )

    existing_config = yaml.safe_load(config_path.read_text())
    normalized_config = yaml.safe_load(yaml.safe_dump(resolved_config, sort_keys=False))
    if existing_config != normalized_config:
        raise ConfigError(
            "Resolved configuration differs from the existing run; resume requires an exact match"
        )


def _resolve_checkpoint(destination: Path, resume: str) -> Path:
    if resume != "latest":
        checkpoint = Path(resume).expanduser().resolve()
        if not checkpoint.is_dir():
            raise ConfigError(f"Resume checkpoint does not exist: {checkpoint}")
        destination_resolved = destination.resolve()
        if checkpoint.parent != destination_resolved or not checkpoint.name.startswith(
            "checkpoint-"
        ):
            raise ConfigError(
                "Resume checkpoint must be a checkpoint-* directory inside the selected run"
            )
        if not (checkpoint / "trainer_state.json").is_file():
            raise ConfigError(f"Resume checkpoint is incomplete: {checkpoint}")
        return checkpoint

    candidates: list[tuple[int, Path]] = []
    for path in destination.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        if not (path / "trainer_state.json").is_file():
            continue
        candidates.append((step, path))
    if not candidates:
        raise ConfigError(f"No complete checkpoint-* directories found in {destination}")
    return max(candidates, key=lambda item: item[0])[1]
