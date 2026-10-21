"""Small, dependency-light helpers for reproducible run metadata."""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
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
    profile: str | None = None
    model_revision: str | None = None

    @property
    def run_id(self) -> str:
        parts: list[str] = [self.experiment]
        if self.condition:
            parts.append(self.condition)
        if self.seed is not None:
            parts.append(f"seed-{self.seed}")
        if self.profile:
            parts.append(self.profile)
        return "__".join(_slug(part) for part in parts)


def write_run_manifest(
    out_dir: str | Path,
    *,
    context: RunContext,
    config: dict[str, Any],
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
    environment: dict[str, Any] | None = None,
) -> None:
    """Write the resolved config and run identity before expensive work starts."""
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    resolved_path = destination / "resolved_config.yaml"
    metadata_path = destination / "run_metadata.json"

    resolved_path.write_text(yaml.safe_dump(config, sort_keys=False))
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_id": context.run_id,
        "context": asdict(context),
        "source_config": str(Path(config_path).resolve()),
        "overrides": overrides or {},
        "environment": environment or {},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def collect_environment_metadata(repo_root: str | Path) -> dict[str, Any]:
    """Collect non-secret provenance needed to reproduce an experiment."""
    root = Path(repo_root).resolve()
    packages = {}
    for package in ("torch", "transformers", "datasets", "trl", "peft", "accelerate", "numpy"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None

    metadata: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "packages": packages,
        "git": {
            "commit": _git_output(root, "rev-parse", "HEAD"),
            "branch": _git_output(root, "branch", "--show-current"),
            "dirty": bool(_git_output(root, "status", "--porcelain")),
        },
    }
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda: dict[str, Any] = {
            "available": cuda_available,
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
        }
        if cuda_available:
            cuda["device_count"] = torch.cuda.device_count()
            cuda["devices"] = [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ]
        metadata["cuda"] = cuda
    except (ImportError, RuntimeError):
        metadata["cuda"] = {"available": False}
    return metadata


def update_run_metadata(out_dir: str | Path, updates: dict[str, Any]) -> None:
    """Merge runtime-discovered provenance without replacing the original manifest."""
    path = Path(out_dir) / "run_metadata.json"
    metadata = json.loads(path.read_text())
    metadata.update(updates)
    path.write_text(json.dumps(metadata, indent=2) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any], *, context: RunContext) -> None:
    """Append one event with stable run-identifying fields."""
    append_jsonl_rows(path, [row], context=context)


def append_jsonl_rows(
    path: str | Path,
    rows: list[dict[str, Any]],
    *,
    context: RunContext,
) -> None:
    """Append several events while opening the output file only once."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a") as stream:
        for row in rows:
            payload = {
                "run_id": context.run_id,
                "pipeline": context.pipeline,
                "experiment": context.experiment,
                "study": context.study,
                "model": context.model,
                "model_revision": context.model_revision,
                "condition": context.condition,
                "seed": context.seed,
                "profile": context.profile,
                **row,
            }
            stream.write(json.dumps(payload) + "\n")


def _slug(value: str) -> str:
    normalized = "".join(char if char.isalnum() or char in "-_" else "-" for char in value)
    return normalized.strip("-")


def slugify(value: str) -> str:
    """Public filename-safe slug used by run-directory construction."""
    return _slug(value)


def _git_output(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()
