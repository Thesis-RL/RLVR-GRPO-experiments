"""Small, dependency-light helpers for reproducible run metadata."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RunContext:
    pipeline: str
    experiment: str
    study: str
    model: str
    model_role: str
    condition: str | None = None
    reward: str | None = None
    noise: str | None = None
    seed: int | None = None

    @property
    def run_id(self) -> str:
        parts: list[str] = [self.experiment]
        if self.condition:
            parts.append(self.condition)
        if self.seed is not None:
            parts.append(f"seed-{self.seed}")
        return "__".join(_slug(part) for part in parts)


def write_run_manifest(
    out_dir: str | Path,
    *,
    context: RunContext,
    config: dict[str, Any],
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> None:
    """Write the resolved config and run identity before expensive work starts."""
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    resolved_path = destination / "resolved_config.yaml"
    metadata_path = destination / "run_metadata.json"

    resolved_path.write_text(yaml.safe_dump(config, sort_keys=False))
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": context.run_id,
        "context": asdict(context),
        "source_config": str(Path(config_path).resolve()),
        "overrides": overrides or {},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any], *, context: RunContext) -> None:
    """Append one event with stable run-identifying fields."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": context.run_id,
        "experiment": context.experiment,
        "study": context.study,
        "condition": context.condition,
        "seed": context.seed,
        **row,
    }
    with destination.open("a") as stream:
        stream.write(json.dumps(payload) + "\n")


def _slug(value: str) -> str:
    normalized = "".join(char if char.isalnum() or char in "-_" else "-" for char in value)
    return normalized.strip("-")
