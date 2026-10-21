"""Lightweight wall-time and resource telemetry for cloud pilots."""

from __future__ import annotations

import json
import platform
import resource
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch


@dataclass
class RunTelemetry:
    out_dir: Path
    started_at: float = field(default_factory=time.perf_counter)
    started_cpu: dict[str, float] = field(default_factory=lambda: _cpu_times())
    stages: dict[str, float] = field(default_factory=dict)

    def reset_gpu_peak(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        _sync_cuda()
        started = time.perf_counter()
        try:
            yield
        finally:
            _sync_cuda()
            self.stages[name] = self.stages.get(name, 0.0) + (time.perf_counter() - started)

    def write(
        self,
        *,
        status: str,
        reward_seconds: float = 0.0,
        grading_seconds: float = 0.0,
        generation_and_scoring_seconds: float = 0.0,
        generated_tokens: int = 0,
        train_metrics: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        total_seconds = time.perf_counter() - self.started_at
        estimated_generation_seconds = max(0.0, generation_and_scoring_seconds - reward_seconds)
        training_seconds = self.stages.get("training", 0.0)
        evaluation_seconds = sum(
            seconds for name, seconds in self.stages.items() if name.startswith("evaluation_")
        )
        cpu = _cpu_times()
        report: dict[str, Any] = {
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "status": status,
            "total_wall_seconds": total_seconds,
            "stage_seconds": self.stages,
            "generation_and_scoring_seconds": generation_and_scoring_seconds,
            "reward_seconds": reward_seconds,
            "grading_seconds": grading_seconds,
            "estimated_generation_seconds": estimated_generation_seconds,
            "estimated_update_and_overhead_seconds": max(
                0.0, training_seconds - generation_and_scoring_seconds - evaluation_seconds
            ),
            "training_generated_completion_tokens": generated_tokens,
            "host_peak_rss_bytes": _peak_rss_bytes(resource.RUSAGE_SELF),
            "children_peak_rss_bytes": _peak_rss_bytes(resource.RUSAGE_CHILDREN),
            "process_user_cpu_seconds": cpu["self_user"] - self.started_cpu["self_user"],
            "process_system_cpu_seconds": cpu["self_system"] - self.started_cpu["self_system"],
            "children_user_cpu_seconds": cpu["children_user"] - self.started_cpu["children_user"],
            "children_system_cpu_seconds": cpu["children_system"]
            - self.started_cpu["children_system"],
            "output_size_bytes": _directory_size(self.out_dir),
            "gpu": _gpu_usage(),
            "train_metrics": train_metrics or {},
            "error": error,
        }
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "resource_report.json").write_text(json.dumps(report, indent=2) + "\n")
        with (self.out_dir / "resource_report_attempts.jsonl").open("a") as stream:
            stream.write(json.dumps(report) + "\n")
        return report


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _peak_rss_bytes(who: int) -> int:
    value = int(resource.getrusage(who).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _cpu_times() -> dict[str, float]:
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "self_user": float(own.ru_utime),
        "self_system": float(own.ru_stime),
        "children_user": float(children.ru_utime),
        "children_system": float(children.ru_stime),
    }


def _gpu_usage() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    return {
        "available": True,
        "device": torch.cuda.current_device(),
        "name": torch.cuda.get_device_name(torch.cuda.current_device()),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total
