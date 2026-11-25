"""Resumable multi-sample clean evaluation with task-clustered uncertainty.

The unit of resampling is an MBPP task, not an individual completion.  This
keeps the confidence interval honest when several generations are collected
for the same task.  Generation is split into deterministic chunks so a run can
be resumed, or extended from four to eight samples, without invalidating the
already stored rows.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlvr.data.mbpp import MBPPProblem
from rlvr.exec.mbpp_executor import grade_codes
from rlvr.generate.sample import GenerationConfig, sample_completions
from rlvr.prompts.mbpp import build_prompt, extract_code
from rlvr.rewards.aggregation import binary_rewards, fractional_rewards


@dataclass(frozen=True)
class MultiSampleTarget:
    """Identity of one base or adapter evaluation target."""

    target_id: str
    kind: str
    model: str
    model_revision: str
    checkpoint: str | None = None
    condition: str | None = None
    step: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"base", "adapter"}:
            raise ValueError("target kind must be 'base' or 'adapter'")
        if self.kind == "base" and self.checkpoint is not None:
            raise ValueError("base target cannot have a checkpoint")
        if self.kind == "adapter" and not self.checkpoint:
            raise ValueError("adapter target requires a checkpoint")


def evaluate_multisample_clean(
    model: Any,
    tokenizer: Any,
    problems: list[MBPPProblem],
    *,
    few_shot: list[MBPPProblem],
    target: MultiSampleTarget,
    out_dir: str | Path,
    samples_per_task: int,
    generation_seed: int,
    generation_chunk_size: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    stop_sequence: str,
    timeout_s: float,
    timeout_fallback_s: float,
    mem_limit: int,
    generation_chunk_offset: int = 0,
    generation_chunk_stride: int = 1,
    difficulty_by_task: Mapping[int, str] | None = None,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20_261_116,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Evaluate one target and resume at deterministic generation-chunk granularity."""
    if samples_per_task <= 0:
        raise ValueError("samples_per_task must be positive")
    if generation_chunk_size <= 0:
        raise ValueError("generation_chunk_size must be positive")
    if generation_chunk_offset < 0 or generation_chunk_stride <= 0:
        raise ValueError("generation chunk offset must be non-negative and stride positive")
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    if not problems:
        raise ValueError("problems must not be empty")

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "predictions.jsonl"
    summary_path = output / "summary.json"
    manifest_path = output / "evaluation_manifest.json"

    protocol = _protocol_manifest(
        target=target,
        problems=problems,
        generation_seed=generation_seed,
        generation_chunk_size=generation_chunk_size,
        generation_chunk_offset=generation_chunk_offset,
        generation_chunk_stride=generation_chunk_stride,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        stop_sequence=stop_sequence,
        timeout_s=timeout_s,
        timeout_fallback_s=timeout_fallback_s,
        mem_limit=mem_limit,
    )
    protocol_fingerprint = _fingerprint(protocol)
    _validate_or_extend_manifest(
        manifest_path,
        protocol=protocol,
        protocol_fingerprint=protocol_fingerprint,
        samples_per_task=samples_per_task,
    )

    expected_task_ids = {problem.task_id for problem in problems}
    existing = read_multisample_predictions(
        predictions_path,
        protocol_fingerprint=protocol_fingerprint,
        expected_task_ids=expected_task_ids,
        repair_trailing_partial=True,
    )
    if existing:
        highest_existing = max(sample_index for _, sample_index in existing) + 1
        if highest_existing > samples_per_task:
            raise RuntimeError(
                f"Stored predictions contain {highest_existing} samples per task, "
                f"but only {samples_per_task} were requested"
            )

    total = len(problems) * samples_per_task
    completed = sum(
        (problem.task_id, sample_index) in existing
        for problem in problems
        for sample_index in range(samples_per_task)
    )
    if progress_callback is not None:
        progress_callback(completed, total)

    if completed == total and summary_path.is_file():
        saved = json.loads(summary_path.read_text())
        if (
            saved.get("protocol_fingerprint") == protocol_fingerprint
            and int(saved.get("samples_per_task", -1)) == samples_per_task
        ):
            return saved

    generation = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        stop_sequence=stop_sequence,
    )
    with _preserve_rng_state():
        was_training = bool(model.training)
        model.eval()
        try:
            for problem in problems:
                prompt = build_prompt(problem, few_shot)
                for chunk_start in range(0, samples_per_task, generation_chunk_size):
                    chunk_stop = min(
                        chunk_start + generation_chunk_size,
                        samples_per_task,
                    )
                    missing = [
                        sample_index
                        for sample_index in range(chunk_start, chunk_stop)
                        if (problem.task_id, sample_index) not in existing
                    ]
                    if not missing:
                        continue

                    chunk_seed_slot = (
                        generation_chunk_offset + chunk_start * generation_chunk_stride
                    )
                    chunk_seed = sample_chunk_seed(
                        generation_seed,
                        problem.task_id,
                        chunk_seed_slot,
                    )
                    _set_seed(chunk_seed)
                    chunk_size = chunk_stop - chunk_start
                    completions = sample_completions(
                        model,
                        tokenizer,
                        prompt,
                        n=chunk_size,
                        gen=generation,
                        batch_size=chunk_size,
                    )
                    codes = [extract_code(completion) for completion in completions]
                    results = grade_codes(
                        codes,
                        problem,
                        timeout_s=timeout_s,
                        timeout_fallback_s=timeout_fallback_s,
                        mem_limit=mem_limit,
                    )
                    if len(results) != chunk_size:
                        raise RuntimeError("grader returned an unexpected number of results")

                    for position, (completion, code, result) in enumerate(
                        zip(completions, codes, results, strict=True)
                    ):
                        sample_index = chunk_start + position
                        key = (problem.task_id, sample_index)
                        if key in existing:
                            continue
                        outcomes = np.asarray([result.outcomes], dtype=np.int8)
                        row = {
                            "schema_version": 1,
                            "event": "multisample_clean_evaluation",
                            "protocol_fingerprint": protocol_fingerprint,
                            "target_id": target.target_id,
                            "target_kind": target.kind,
                            "condition": target.condition,
                            "step": target.step,
                            "checkpoint": target.checkpoint,
                            "task_id": problem.task_id,
                            "sample_index": sample_index,
                            "generation_chunk_start": chunk_start,
                            "generation_chunk_seed_slot": chunk_seed_slot,
                            "generation_seed": chunk_seed,
                            "difficulty": (difficulty_by_task or {}).get(
                                problem.task_id,
                                "unknown",
                            ),
                            "prompt": prompt,
                            "completion": completion,
                            "code": code,
                            "exec_error": result.error,
                            "clean_outcomes": list(result.outcomes),
                            "clean_binary_reward": float(binary_rewards(outcomes)[0]),
                            "clean_fractional_reward": float(fractional_rewards(outcomes)[0]),
                        }
                        _append_jsonl(predictions_path, row)
                        existing[key] = row
                        completed += 1
                        if progress_callback is not None:
                            progress_callback(completed, total)
        finally:
            if was_training:
                model.train()

    selected = [
        existing[(problem.task_id, sample_index)]
        for problem in problems
        for sample_index in range(samples_per_task)
    ]
    summary = summarize_target_rows(
        selected,
        target=target,
        samples_per_task=samples_per_task,
        protocol_fingerprint=protocol_fingerprint,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    _write_json_atomic(summary_path, summary)
    return summary


def summarize_target_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    target: MultiSampleTarget,
    samples_per_task: int,
    protocol_fingerprint: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Summarize one target using cluster bootstrap over task-level means."""
    _validate_balanced_rows(rows, samples_per_task=samples_per_task)
    binary = _metric_summary(
        rows,
        "clean_binary_reward",
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    fractional = _metric_summary(
        rows,
        "clean_fractional_reward",
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed + 1,
    )
    by_difficulty: dict[str, Any] = {}
    for index, stratum in enumerate(("hard", "medium", "easy", "unknown"), start=1):
        selected = [row for row in rows if row.get("difficulty", "unknown") == stratum]
        if not selected:
            continue
        by_difficulty[stratum] = {
            "n_problems": len({int(row["task_id"]) for row in selected}),
            "pass_at_1": _metric_summary(
                selected,
                "clean_binary_reward",
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed + 10 * index,
            ),
            "mean_test_pass_rate": _metric_summary(
                selected,
                "clean_fractional_reward",
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed + 10 * index + 1,
            ),
        }
    return {
        "schema_version": 1,
        "kind": "multisample_target_summary",
        "protocol_fingerprint": protocol_fingerprint,
        "target": asdict(target),
        "samples_per_task": samples_per_task,
        "n_problems": len({int(row["task_id"]) for row in rows}),
        "n_generations": len(rows),
        "bootstrap": {
            "method": "percentile_task_cluster",
            "confidence_level": 0.95,
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
        },
        "pass_at_1": binary,
        "mean_test_pass_rate": fractional,
        "by_difficulty": by_difficulty,
    }


def compare_task_clustered(
    base_rows: Sequence[Mapping[str, Any]],
    trained_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20_261_117,
) -> dict[str, Any]:
    """Return paired task-clustered CIs for base, trained, and their difference."""
    base_counts = _samples_per_task(base_rows)
    trained_counts = _samples_per_task(trained_rows)
    if not base_counts or not trained_counts:
        raise ValueError("base_rows and trained_rows must not be empty")
    if set(base_counts) != set(trained_counts):
        raise ValueError("base and trained rows must cover exactly the same tasks")
    if len(set(base_counts.values())) != 1 or len(set(trained_counts.values())) != 1:
        raise ValueError("each side must have a balanced sample count across tasks")

    return {
        "n_problems": len(base_counts),
        "base_samples_per_task": next(iter(base_counts.values())),
        "trained_samples_per_task": next(iter(trained_counts.values())),
        "bootstrap": {
            "method": "paired_percentile_task_cluster",
            "confidence_level": 0.95,
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
        },
        "pass_at_1": _paired_metric_summary(
            base_rows,
            trained_rows,
            "clean_binary_reward",
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
        ),
        "mean_test_pass_rate": _paired_metric_summary(
            base_rows,
            trained_rows,
            "clean_fractional_reward",
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed + 1,
        ),
    }


def read_multisample_predictions(
    path: str | Path,
    *,
    protocol_fingerprint: str | None = None,
    expected_task_ids: set[int] | None = None,
    repair_trailing_partial: bool = False,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Read prediction rows, optionally truncating one crash-partial final line."""
    source = Path(path)
    if not source.exists():
        return {}

    rows: dict[tuple[int, int], dict[str, Any]] = {}
    last_good_offset = 0
    size = source.stat().st_size
    with source.open("rb") as handle:
        line_number = 0
        while True:
            offset = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            line_number += 1
            try:
                row = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                is_last = handle.tell() == size
                if repair_trailing_partial and is_last:
                    with source.open("r+b") as repair:
                        repair.truncate(offset)
                    break
                raise RuntimeError(f"Invalid JSONL at {source}:{line_number}") from exc
            last_good_offset = handle.tell()
            if protocol_fingerprint is not None and (
                row.get("protocol_fingerprint") != protocol_fingerprint
            ):
                raise RuntimeError(f"Protocol fingerprint mismatch at {source}:{line_number}")
            task_id = int(row["task_id"])
            sample_index = int(row["sample_index"])
            if sample_index < 0:
                raise RuntimeError(f"Negative sample index at {source}:{line_number}")
            if expected_task_ids is not None and task_id not in expected_task_ids:
                raise RuntimeError(f"Unexpected task_id {task_id} at {source}:{line_number}")
            key = (task_id, sample_index)
            if key in rows:
                raise RuntimeError(f"Duplicate task/sample key {key} at {source}:{line_number}")
            rows[key] = row
    if last_good_offset > size:
        raise AssertionError("invalid JSONL offset accounting")
    return rows


def sample_chunk_seed(base_seed: int, task_id: int, chunk_start: int) -> int:
    """Stable RNG seed for one task/chunk pair."""
    sequence = np.random.SeedSequence([base_seed, task_id, chunk_start])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _protocol_manifest(
    *,
    target: MultiSampleTarget,
    problems: Sequence[MBPPProblem],
    generation_seed: int,
    generation_chunk_size: int,
    generation_chunk_offset: int,
    generation_chunk_stride: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    stop_sequence: str,
    timeout_s: float,
    timeout_fallback_s: float,
    mem_limit: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "multisample_clean_evaluation_protocol",
        "target": asdict(target),
        "dataset": {
            "name": "mbpp",
            "split": problems[0].split,
            "task_ids": [problem.task_id for problem in problems],
        },
        "generation": {
            "seed": generation_seed,
            "chunk_size": generation_chunk_size,
            "chunk_offset": generation_chunk_offset,
            "chunk_stride": generation_chunk_stride,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "stop_sequence": stop_sequence,
            "do_sample": True,
        },
        "grading": {
            "timeout_s": timeout_s,
            "timeout_fallback_s": timeout_fallback_s,
            "mem_limit": mem_limit,
        },
    }


def _validate_or_extend_manifest(
    path: Path,
    *,
    protocol: Mapping[str, Any],
    protocol_fingerprint: str,
    samples_per_task: int,
) -> None:
    desired = {
        **protocol,
        "protocol_fingerprint": protocol_fingerprint,
        "max_requested_samples_per_task": samples_per_task,
    }
    if not path.exists():
        _write_json_atomic(path, desired)
        return
    existing = json.loads(path.read_text())
    existing_protocol = {
        key: value
        for key, value in existing.items()
        if key not in {"protocol_fingerprint", "max_requested_samples_per_task"}
    }
    if existing_protocol != dict(protocol) or existing.get("protocol_fingerprint") != (
        protocol_fingerprint
    ):
        raise RuntimeError(f"Evaluation protocol differs from the existing manifest: {path}")
    previous = int(existing.get("max_requested_samples_per_task", 0))
    if samples_per_task < previous:
        raise RuntimeError(
            f"Cannot shrink an evaluation from {previous} to {samples_per_task} samples per task"
        )
    if samples_per_task > previous:
        _write_json_atomic(path, desired)


def _metric_summary(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    values = np.asarray(list(_task_means(rows, metric).values()), dtype=np.float64)
    low, high = _bootstrap_mean_ci(
        values,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    return {
        "estimate": float(values.mean()),
        "ci95_low": low,
        "ci95_high": high,
    }


def _paired_metric_summary(
    base_rows: Sequence[Mapping[str, Any]],
    trained_rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    base_by_task = _task_means(base_rows, metric)
    trained_by_task = _task_means(trained_rows, metric)
    task_ids = sorted(base_by_task)
    base = np.asarray([base_by_task[task_id] for task_id in task_ids], dtype=np.float64)
    trained = np.asarray([trained_by_task[task_id] for task_id in task_ids], dtype=np.float64)
    delta = trained - base
    rng = np.random.default_rng(bootstrap_seed)
    base_boot = np.empty(bootstrap_replicates, dtype=np.float64)
    trained_boot = np.empty(bootstrap_replicates, dtype=np.float64)
    delta_boot = np.empty(bootstrap_replicates, dtype=np.float64)
    for index in range(bootstrap_replicates):
        selected = rng.integers(0, len(task_ids), size=len(task_ids))
        base_boot[index] = base[selected].mean()
        trained_boot[index] = trained[selected].mean()
        delta_boot[index] = delta[selected].mean()
    return {
        "base": _estimate_and_quantiles(base.mean(), base_boot),
        "trained": _estimate_and_quantiles(trained.mean(), trained_boot),
        "delta_trained_minus_base": _estimate_and_quantiles(delta.mean(), delta_boot),
    }


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[float, float]:
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty array")
    rng = np.random.default_rng(bootstrap_seed)
    estimates = np.empty(bootstrap_replicates, dtype=np.float64)
    for index in range(bootstrap_replicates):
        selected = rng.integers(0, values.size, size=values.size)
        estimates[index] = values[selected].mean()
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _estimate_and_quantiles(estimate: float, bootstrap: np.ndarray) -> dict[str, float]:
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "estimate": float(estimate),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def _task_means(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[int, float]:
    values: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        values[int(row["task_id"])].append(float(row[metric]))
    return {task_id: float(np.mean(task_values)) for task_id, task_values in values.items()}


def _samples_per_task(rows: Sequence[Mapping[str, Any]]) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    keys: set[tuple[int, int]] = set()
    for row in rows:
        key = (int(row["task_id"]), int(row["sample_index"]))
        if key in keys:
            raise ValueError(f"duplicate task/sample key: {key}")
        keys.add(key)
        counts[key[0]] += 1
    return dict(counts)


def _validate_balanced_rows(rows: Sequence[Mapping[str, Any]], *, samples_per_task: int) -> None:
    counts = _samples_per_task(rows)
    if not counts:
        raise ValueError("rows must not be empty")
    invalid = {task_id: count for task_id, count in counts.items() if count != samples_per_task}
    if invalid:
        raise ValueError(f"unbalanced samples per task: {invalid}")


def _fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def _preserve_rng_state() -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    encoded = (json.dumps(dict(row), separators=(",", ":")) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError(f"short JSONL write: {written} of {len(encoded)} bytes")
    finally:
        os.close(descriptor)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(value), indent=2) + "\n")
    temporary.replace(path)
