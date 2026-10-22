#!/usr/bin/env python3
"""Print a concise live status for one or more online GRPO runs."""

from __future__ import annotations

import argparse
import json
import os
import socket
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, help="A run directory or status.json path")
    parser.add_argument("--root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--all", action="store_true", help="Show every status under --root")
    parser.add_argument("--json", action="store_true", help="Print the selected raw JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = _resolve_status_paths(args.out_dir, args.root, show_all=args.all)
    if not paths:
        raise SystemExit("No status.json files found")
    statuses = [_load_status(path) for path in paths]
    if args.json:
        value: Any = statuses if args.all else statuses[0]
        print(json.dumps(value, indent=2))
        return
    for index, status in enumerate(statuses):
        if index:
            print()
        print(_format_status(status))


def _resolve_status_paths(out_dir: Path | None, root: Path, *, show_all: bool) -> list[Path]:
    if out_dir is not None:
        if out_dir.name in {"status.json", "matrix_status.json"}:
            candidates = [out_dir]
        else:
            candidates = [out_dir / "matrix_status.json", out_dir / "status.json"]
        return [candidate.resolve() for candidate in candidates if candidate.is_file()][:1]
    candidates = []
    if root.exists():
        candidates.extend(root.resolve().rglob("status.json"))
        candidates.extend(root.resolve().rglob("matrix_status.json"))
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates if show_all else candidates[:1]


def _load_status(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Status must contain a JSON object: {path}")
    value["status_path"] = str(path)
    value["process_alive"] = _process_alive(value)
    value["updated_ago_seconds"] = _updated_ago_seconds(value.get("updated_at_utc"))
    return value


def _process_alive(status: dict[str, Any]) -> bool | None:
    if status.get("hostname") != socket.gethostname():
        return None
    try:
        pid = int(status["pid"])
        os.kill(pid, 0)
    except (KeyError, TypeError, ValueError, ProcessLookupError, PermissionError):
        return False
    return True


def _updated_ago_seconds(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        updated = datetime.fromisoformat(value)
    except ValueError:
        return None
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=UTC)
    return max(0.0, (datetime.now(UTC) - updated).total_seconds())


def _format_status(status: dict[str, Any]) -> str:
    if status.get("kind") == "condition_matrix":
        return _format_matrix_status(status)
    current = int(status.get("current_step", 0))
    total = int(status.get("total_steps", 0))
    progress = status.get("progress_fraction")
    percent = f"{float(progress) * 100:.1f}%" if progress is not None else "unknown"
    lines = [
        f"Run: {status.get('run_id', 'unknown')}",
        f"Status: {status.get('status', 'unknown')} / {status.get('stage', 'unknown')}",
        f"Condition: {status.get('condition')} | seed={status.get('seed')}",
        f"Training: step {current}/{total} ({percent}), epoch={status.get('epoch')}",
        f"Elapsed: {_duration(status.get('elapsed_seconds'))}",
        f"ETA ({status.get('eta_scope')}): {_duration(status.get('estimated_seconds_remaining'))}",
        f"Estimated finish (UTC): {status.get('estimated_finished_at_utc') or 'unknown'}",
    ]
    evaluation = status.get("evaluation")
    if isinstance(evaluation, dict):
        lines.append(
            "Evaluation: "
            f"{evaluation.get('split')} {evaluation.get('completed')}/{evaluation.get('total')}"
        )
    lines.extend(
        [
            f"Last checkpoint: {status.get('last_checkpoint') or 'none'}",
            f"Process alive: {status.get('process_alive')}",
            f"Updated: {_duration(status.get('updated_ago_seconds'))} ago",
            f"File: {status.get('status_path')}",
        ]
    )
    return "\n".join(lines)


def _format_matrix_status(status: dict[str, Any]) -> str:
    completed = int(status.get("completed_count", 0))
    total = int(status.get("total_conditions", 0))
    lines = [
        f"Matrix: {status.get('experiment', 'unknown')} | seed={status.get('seed')}",
        f"Status: {status.get('status', 'unknown')}",
        f"Conditions: {completed}/{total} complete",
        f"Current: {status.get('current_condition') or 'none'}",
        f"Pending: {', '.join(status.get('pending_conditions', [])) or 'none'}",
        f"Elapsed: {_duration(status.get('elapsed_seconds'))}",
        "Estimated remaining "
        f"({status.get('eta_scope') or 'scope unknown'}): "
        f"{_duration(status.get('estimated_seconds_remaining'))}",
        f"Estimated finish (UTC): {status.get('estimated_finished_at_utc') or 'unknown'}",
    ]
    current = status.get("current_run")
    if isinstance(current, dict) and current:
        progress = current.get("progress_fraction")
        percent = f"{float(progress) * 100:.1f}%" if progress is not None else "unknown"
        lines.append(
            f"Current run: step {current.get('current_step')}/{current.get('total_steps')} "
            f"({percent}), stage={current.get('stage')}"
        )
        evaluation = current.get("evaluation")
        if isinstance(evaluation, dict):
            lines.append(
                "Current evaluation: "
                f"{evaluation.get('split')} {evaluation.get('completed')}/{evaluation.get('total')}"
            )
    lines.extend(
        [
            f"Process alive: {status.get('process_alive')}",
            f"Updated: {_duration(status.get('updated_ago_seconds'))} ago",
            f"File: {status.get('status_path')}",
        ]
    )
    if status.get("error"):
        lines.append(f"Error: {status['error']}")
    return "\n".join(lines)


def _duration(value: Any) -> str:
    if value is None:
        return "unknown"
    seconds = max(0, int(round(float(value))))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


if __name__ == "__main__":
    main()
