#!/usr/bin/env python3
"""Analyze the frozen priority-bridge protocol across several training seeds.

The primary test analysis is paired at the MBPP-task level.  It reports both a
two-way bootstrap over training seeds and tasks and a conservative Student-t
interval over the three seed-level estimates.  It also summarizes the training
signal retained in the immutable rollout logs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rlvr.config import load_experiment_config  # noqa: E402
from rlvr.evaluation.multisample import read_multisample_predictions  # noqa: E402
from rlvr.online.evaluation import fraction_steps  # noqa: E402
from rlvr.online.settings import (  # noqa: E402
    resolve_evaluation_settings,
    resolve_training_settings,
)

T_CRITICAL_95_DF2 = 4.302652729911275
DIFFICULTIES = ("hard", "medium", "easy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/priority_bridge_v2.yaml",
    )
    parser.add_argument(
        "--matrix-root",
        type=Path,
        required=True,
        help="Parent containing seed-0, seed-1, ... directories.",
    )
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze(
        config_path=args.config.resolve(),
        matrix_root=args.matrix_root.resolve(),
        base_root=args.base_root.resolve(),
        seeds=tuple(args.seeds),
        bootstrap_replicates=args.bootstrap_replicates,
    )
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir
        else (args.matrix_root / "multiseed_analysis").resolve()
    )
    write_report(out_dir, result)
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "seeds": result["protocol"]["seeds"],
                "conditions": len(result["conditions"]),
                "matched_generation_verified": True,
            },
            indent=2,
        )
    )


def analyze(
    *,
    config_path: Path,
    matrix_root: Path,
    base_root: Path,
    seeds: tuple[int, ...],
    bootstrap_replicates: int | None = None,
) -> dict[str, Any]:
    if len(seeds) != 3 or len(set(seeds)) != len(seeds):
        raise ValueError("the frozen aggregate analysis requires exactly three distinct seeds")

    experiment = load_experiment_config(config_path)
    training = resolve_training_settings(experiment.values, smoke=False)
    evaluation = resolve_evaluation_settings(experiment.values)
    steps = list(fraction_steps(training.max_steps, list(evaluation.test_fractions)))
    conditions = [condition.name for condition in experiment.conditions]
    replicates = bootstrap_replicates or evaluation.bootstrap_replicates
    if len(steps) != 2:
        raise ValueError("the frozen protocol requires exactly two test checkpoints")
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")

    base_rows = _load_rows(base_root, samples=evaluation.base_samples_per_task)
    task_ids = tuple(sorted({int(row["task_id"]) for row in base_rows}))
    base_vectors = {
        metric: _task_vector(base_rows, task_ids, metric)
        for metric in ("clean_binary_reward", "clean_fractional_reward")
    }
    difficulty_by_task = _difficulty_map(base_rows, task_ids)

    trained: dict[str, dict[str, np.ndarray]] = {}
    signal_rows: list[dict[str, Any]] = []
    for condition in conditions:
        trained[condition] = {}
        for metric in ("clean_binary_reward", "clean_fractional_reward"):
            seed_vectors = []
            for seed in seeds:
                rows = _combined_condition_rows(
                    matrix_root=matrix_root,
                    base_rows=base_rows,
                    condition=condition,
                    seed=seed,
                    steps=steps,
                    samples=evaluation.test_samples_per_task,
                )
                seed_vectors.append(_task_vector(rows, task_ids, metric))
            trained[condition][metric] = np.stack(seed_vectors)
        for seed in seeds:
            signal_rows.append(
                _summarize_rollouts(matrix_root / f"seed-{seed}" / condition, condition, seed)
            )

    metric_names = {
        "pass_at_1": "clean_binary_reward",
        "mean_test_pass_rate": "clean_fractional_reward",
    }
    condition_results: dict[str, Any] = {}
    for condition_index, condition in enumerate(conditions):
        condition_metrics: dict[str, Any] = {}
        for metric_index, (metric_name, field) in enumerate(metric_names.items()):
            condition_metrics[metric_name] = _condition_result(
                trained=trained[condition][field],
                base=base_vectors[field],
                seeds=seeds,
                replicates=replicates,
                bootstrap_seed=evaluation.bootstrap_seed
                + 10_000 * condition_index
                + 1_000 * metric_index,
            )
        condition_metrics["by_difficulty"] = {}
        for difficulty_index, difficulty in enumerate(DIFFICULTIES):
            indices = np.asarray(
                [
                    index
                    for index, task_id in enumerate(task_ids)
                    if difficulty_by_task[task_id] == difficulty
                ],
                dtype=np.int64,
            )
            if not len(indices):
                continue
            slices: dict[str, Any] = {"n_problems": int(len(indices))}
            for metric_index, (metric_name, field) in enumerate(metric_names.items()):
                slices[metric_name] = _condition_result(
                    trained=trained[condition][field][:, indices],
                    base=base_vectors[field][indices],
                    seeds=seeds,
                    replicates=replicates,
                    bootstrap_seed=evaluation.bootstrap_seed
                    + 100_000
                    + 10_000 * condition_index
                    + 2_000 * difficulty_index
                    + 1_000 * metric_index,
                )
            condition_metrics["by_difficulty"][difficulty] = slices
        condition_results[condition] = condition_metrics

    contrasts = _planned_contrasts(
        trained=trained,
        metrics=metric_names,
        seeds=seeds,
        replicates=replicates,
        bootstrap_seed=evaluation.bootstrap_seed + 1_000_000,
    )
    signal = _aggregate_signal(signal_rows, seeds)
    return {
        "schema_version": 1,
        "kind": "priority_bridge_multiseed_analysis",
        "experiment": experiment.name,
        "protocol": {
            "config": str(config_path),
            "matrix_root": str(matrix_root),
            "base_root": str(base_root),
            "seeds": list(seeds),
            "checkpoints": steps,
            "n_tasks": len(task_ids),
            "base_samples_per_task": evaluation.base_samples_per_task,
            "trained_samples_per_task": len(steps) * evaluation.test_samples_per_task,
            "generation_seed": evaluation.seed,
            "matched_generation_verified": True,
            "bootstrap": {
                "method": "paired two-way percentile bootstrap over seeds and tasks",
                "replicates": replicates,
                "seed": evaluation.bootstrap_seed,
            },
            "seed_interval": {
                "method": "Student-t over seed-level estimates",
                "confidence_level": 0.95,
                "degrees_of_freedom": len(seeds) - 1,
                "warning": "Only three training seeds; seed-level intervals are necessarily wide.",
            },
        },
        "conditions": condition_results,
        "planned_contrasts": contrasts,
        "training_signal": signal,
        "interpretation_policy": {
            "primary_metric": "pass_at_1",
            "test_was_not_used_for_hyperparameter_selection": True,
            "contrast_intervals_are_unadjusted": True,
            "population_claim_requires_seed_level_interval_to_exclude_zero": True,
        },
    }


def _load_rows(path: Path, *, samples: int) -> list[dict[str, Any]]:
    summary_path = path / "summary.json"
    predictions_path = path / "predictions.jsonl"
    if not summary_path.is_file() or not predictions_path.is_file():
        raise ValueError(f"incomplete evaluation target: {path}")
    summary = json.loads(summary_path.read_text())
    if int(summary.get("samples_per_task", -1)) < samples:
        raise ValueError(f"target {path} has fewer than {samples} samples per task")
    rows = read_multisample_predictions(predictions_path)
    selected = [row for (_, sample), row in rows.items() if sample < samples]
    expected = int(summary["n_problems"]) * samples
    if len(selected) != expected:
        raise ValueError(f"target {path} has {len(selected)} of {expected} required rows")
    return selected


def _combined_condition_rows(
    *,
    matrix_root: Path,
    base_rows: list[dict[str, Any]],
    condition: str,
    seed: int,
    steps: list[int],
    samples: int,
) -> list[dict[str, Any]]:
    seed_root = matrix_root / f"seed-{seed}"
    status_path = seed_root / "matrix_status.json"
    if not status_path.is_file():
        raise ValueError(f"missing matrix status: {status_path}")
    status = json.loads(status_path.read_text())
    if status.get("status") != "complete" or int(status.get("seed", -1)) != seed:
        raise ValueError(f"seed {seed} matrix is not complete or has the wrong identity")

    combined: list[dict[str, Any]] = []
    for checkpoint_index, step in enumerate(steps):
        rows = _load_rows(
            seed_root / condition / "evaluations" / "test" / f"step-{step}",
            samples=samples,
        )
        task_ids = {int(row["task_id"]) for row in rows}
        base_slice = _base_slice(
            base_rows,
            checkpoint_index=checkpoint_index,
            samples=samples,
            task_ids=task_ids,
        )
        _validate_matched_generation(base_slice, rows, condition=condition, seed=seed, step=step)
        for row in rows:
            adjusted = dict(row)
            adjusted["sample_index"] = checkpoint_index * samples + int(row["sample_index"])
            combined.append(adjusted)
    return combined


def _base_slice(
    rows: list[dict[str, Any]],
    *,
    checkpoint_index: int,
    samples: int,
    task_ids: set[int],
) -> list[dict[str, Any]]:
    start = checkpoint_index * samples
    stop = start + samples
    selected = []
    for row in rows:
        if int(row["task_id"]) not in task_ids:
            continue
        sample_index = int(row["sample_index"])
        if start <= sample_index < stop:
            adjusted = dict(row)
            adjusted["sample_index"] = sample_index - start
            selected.append(adjusted)
    return selected


def _validate_matched_generation(
    base_rows: list[dict[str, Any]],
    trained_rows: list[dict[str, Any]],
    *,
    condition: str,
    seed: int,
    step: int,
) -> None:
    def signature(rows: list[dict[str, Any]]) -> dict[int, tuple[tuple[int, int], ...]]:
        by_task: dict[int, set[tuple[int, int]]] = defaultdict(set)
        for row in rows:
            by_task[int(row["task_id"])].add(
                (int(row["generation_chunk_seed_slot"]), int(row["generation_seed"]))
            )
        return {task_id: tuple(sorted(values)) for task_id, values in by_task.items()}

    if signature(base_rows) != signature(trained_rows):
        raise ValueError(
            f"generation seeds do not match for seed {seed}, {condition}, checkpoint {step}"
        )


def _task_vector(rows: list[dict[str, Any]], task_ids: tuple[int, ...], metric: str) -> np.ndarray:
    by_task: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_task[int(row["task_id"])].append(float(row[metric]))
    missing = [task_id for task_id in task_ids if task_id not in by_task]
    if missing:
        raise ValueError(f"missing task rows for metric {metric}: {missing[:5]}")
    sample_counts = {len(by_task[task_id]) for task_id in task_ids}
    if len(sample_counts) != 1:
        raise ValueError(f"unbalanced samples per task for metric {metric}: {sample_counts}")
    return np.asarray([np.mean(by_task[task_id]) for task_id in task_ids], dtype=np.float64)


def _difficulty_map(rows: list[dict[str, Any]], task_ids: tuple[int, ...]) -> dict[int, str]:
    result: dict[int, str] = {}
    for row in rows:
        task_id = int(row["task_id"])
        difficulty = str(row.get("difficulty", "unknown"))
        if task_id in result and result[task_id] != difficulty:
            raise ValueError(f"inconsistent difficulty for task {task_id}")
        result[task_id] = difficulty
    if set(result) != set(task_ids):
        raise ValueError("difficulty mapping does not cover all tasks")
    return result


def _condition_result(
    *,
    trained: np.ndarray,
    base: np.ndarray,
    seeds: tuple[int, ...],
    replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if trained.shape != (len(seeds), len(base)):
        raise ValueError("trained matrix and base vector are not aligned")
    delta = trained - base[None, :]
    return {
        "base": _task_summary(base, replicates=replicates, seed=bootstrap_seed),
        "trained": _seed_task_summary(
            trained,
            seeds=seeds,
            replicates=replicates,
            seed=bootstrap_seed + 1,
        ),
        "delta_trained_minus_base": _seed_task_summary(
            delta,
            seeds=seeds,
            replicates=replicates,
            seed=bootstrap_seed + 2,
        ),
    }


def _task_summary(values: np.ndarray, *, replicates: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 500):
        count = min(500, replicates - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        estimates[start : start + count] = values[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "estimate": float(values.mean()),
        "task_bootstrap_ci95_low": float(low),
        "task_bootstrap_ci95_high": float(high),
    }


def _seed_task_summary(
    values: np.ndarray,
    *,
    seeds: tuple[int, ...],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if values.shape[0] != len(seeds):
        raise ValueError("seed labels do not match the value matrix")
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    seed_count, task_count = values.shape
    for start in range(0, replicates, 200):
        count = min(200, replicates - start)
        seed_indices = rng.integers(0, seed_count, size=(count, seed_count))
        task_indices = rng.integers(0, task_count, size=(count, task_count))
        sampled = values[seed_indices[:, :, None], task_indices[:, None, :]]
        estimates[start : start + count] = sampled.mean(axis=(1, 2))
    low, high = np.quantile(estimates, [0.025, 0.975])
    by_seed_values = values.mean(axis=1)
    seed_low, seed_high = _seed_t_interval(by_seed_values)
    return {
        "estimate": float(values.mean()),
        "two_way_bootstrap_ci95_low": float(low),
        "two_way_bootstrap_ci95_high": float(high),
        "by_seed": {
            str(seed_value): float(estimate)
            for seed_value, estimate in zip(seeds, by_seed_values, strict=True)
        },
        "seed_std": float(np.std(by_seed_values, ddof=1)),
        "seed_t_ci95_low": seed_low,
        "seed_t_ci95_high": seed_high,
    }


def _seed_t_interval(values: np.ndarray) -> tuple[float, float]:
    if len(values) != 3:
        raise ValueError("the frozen Student-t interval is defined for exactly three seeds")
    mean = float(np.mean(values))
    half_width = T_CRITICAL_95_DF2 * float(np.std(values, ddof=1)) / math.sqrt(len(values))
    return mean - half_width, mean + half_width


def _planned_contrasts(
    *,
    trained: dict[str, dict[str, np.ndarray]],
    metrics: dict[str, str],
    seeds: tuple[int, ...],
    replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    definitions: dict[str, dict[str, float]] = {
        "clean_fractional_minus_binary": {
            "clean__fractional": 1.0,
            "clean__binary": -1.0,
        },
        "fp_fractional_minus_binary": {
            "fp_only__fractional": 1.0,
            "fp_only__binary": -1.0,
        },
        "fn_fractional_minus_binary": {
            "fn_only__fractional": 1.0,
            "fn_only__binary": -1.0,
        },
        "binary_fp_minus_clean": {"fp_only__binary": 1.0, "clean__binary": -1.0},
        "binary_fn_minus_clean": {"fn_only__binary": 1.0, "clean__binary": -1.0},
        "binary_fn_minus_fp": {"fn_only__binary": 1.0, "fp_only__binary": -1.0},
        "fractional_fp_minus_clean": {
            "fp_only__fractional": 1.0,
            "clean__fractional": -1.0,
        },
        "fractional_fn_minus_clean": {
            "fn_only__fractional": 1.0,
            "clean__fractional": -1.0,
        },
        "fractional_fn_minus_fp": {
            "fn_only__fractional": 1.0,
            "fp_only__fractional": -1.0,
        },
        "fp_aggregation_interaction": {
            "fp_only__fractional": 1.0,
            "clean__fractional": -1.0,
            "fp_only__binary": -1.0,
            "clean__binary": 1.0,
        },
        "fn_aggregation_interaction": {
            "fn_only__fractional": 1.0,
            "clean__fractional": -1.0,
            "fn_only__binary": -1.0,
            "clean__binary": 1.0,
        },
        "fn_vs_fp_aggregation_interaction": {
            "fn_only__fractional": 1.0,
            "fp_only__fractional": -1.0,
            "fn_only__binary": -1.0,
            "fp_only__binary": 1.0,
        },
    }
    results: dict[str, Any] = {}
    for contrast_index, (name, weights) in enumerate(definitions.items()):
        result = {"weights": weights, "metrics": {}}
        for metric_index, (metric_name, field) in enumerate(metrics.items()):
            effect = sum(
                weight * trained[condition][field] for condition, weight in weights.items()
            )
            result["metrics"][metric_name] = _seed_task_summary(
                effect,
                seeds=seeds,
                replicates=replicates,
                seed=bootstrap_seed + 10_000 * contrast_index + 1_000 * metric_index,
            )
        results[name] = result
    return results


def _summarize_rollouts(run_dir: Path, condition: str, seed: int) -> dict[str, Any]:
    path = run_dir / "rollouts.jsonl"
    if not path.is_file():
        raise ValueError(f"missing rollout log: {path}")
    reward_type = condition.split("__", maxsplit=1)[1]
    clean_field = f"clean_{reward_type}_reward"
    noisy_field = f"noisy_{reward_type}_reward"
    row_values: dict[str, list[float]] = defaultdict(list)
    group_values: dict[str, dict[str, float | None]] = {}
    row_count = 0
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row["seed"]) != seed or row["condition"] != condition:
                raise ValueError(f"rollout identity mismatch in {path}")
            row_count += 1
            for output_name, field in (
                ("mean_clean_reward", clean_field),
                ("mean_noisy_reward", noisy_field),
                ("mean_train_reward", "train_reward"),
                ("mean_abs_train_advantage", "train_advantage"),
                ("mean_changed_label_rate", "changed_label_rate"),
                ("mean_verifier_precision", "verifier_precision"),
                ("mean_verifier_recall", "verifier_recall"),
                ("mean_realized_alpha", "realized_alpha"),
                ("mean_realized_beta", "realized_beta"),
            ):
                value = row.get(field)
                if value is not None and math.isfinite(float(value)):
                    numeric = abs(float(value)) if field == "train_advantage" else float(value)
                    row_values[output_name].append(numeric)
            row_values["nonzero_train_advantage"].append(
                float(abs(float(row["train_advantage"])) > 1e-12)
            )
            group_id = str(row["group_id"])
            values = {
                "zero_variance_group_fraction": float(bool(row["zero_variance_group"])),
                "clean_zero_variance_group_fraction": float(
                    bool(row["train_aggregation_zero_variance_clean"])
                ),
                "mean_advantage_sign_agreement": _optional_float(
                    row["train_aggregation_advantage_sign_agreement"]
                ),
                "mean_advantage_cosine": _optional_float(row["train_aggregation_advantage_cosine"]),
                "mean_pairwise_fidelity": _optional_float(
                    row["train_aggregation_pairwise_fidelity"]
                ),
                "mean_spearman": _optional_float(row["train_aggregation_spearman"]),
                "mean_top1_preserved": _optional_float(row["train_aggregation_top1_preserved"]),
            }
            if group_id in group_values and group_values[group_id] != values:
                raise ValueError(f"inconsistent group diagnostics for {group_id}")
            group_values[group_id] = values

    if not row_count or not group_values:
        raise ValueError(f"empty rollout log: {path}")
    result: dict[str, Any] = {
        "condition": condition,
        "seed": seed,
        "row_count": row_count,
        "group_count": len(group_values),
    }
    result.update({name: float(np.mean(values)) for name, values in row_values.items()})
    for name in next(iter(group_values.values())):
        values = [
            float(group[name])
            for group in group_values.values()
            if group[name] is not None and math.isfinite(float(group[name]))
        ]
        if not values:
            raise ValueError(f"no finite {name} diagnostics in {path}")
        result[name] = float(np.mean(values))
    return result


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _aggregate_signal(rows: list[dict[str, Any]], seeds: tuple[int, ...]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)
    result: dict[str, Any] = {"per_seed": rows, "conditions": {}}
    excluded = {"condition", "seed", "row_count", "group_count"}
    for condition, condition_rows in by_condition.items():
        if {int(row["seed"]) for row in condition_rows} != set(seeds):
            raise ValueError(f"incomplete signal rows for {condition}")
        metrics: dict[str, Any] = {}
        for name in condition_rows[0]:
            if name in excluded:
                continue
            values = np.asarray([float(row[name]) for row in condition_rows], dtype=np.float64)
            low, high = _seed_t_interval(values)
            metrics[name] = {
                "mean": float(values.mean()),
                "seed_std": float(values.std(ddof=1)),
                "seed_t_ci95_low": low,
                "seed_t_ci95_high": high,
                "by_seed": {str(row["seed"]): float(row[name]) for row in condition_rows},
            }
        result["conditions"][condition] = metrics
    return result


def write_report(out_dir: Path, result: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(out_dir / "analysis.json", result)
    (out_dir / "analysis.md").write_text(_markdown(result))
    _write_csv(out_dir / "condition_results.csv", _condition_csv_rows(result))
    _write_csv(out_dir / "planned_contrasts.csv", _contrast_csv_rows(result))
    _write_csv(out_dir / "training_signal.csv", result["training_signal"]["per_seed"])


def _condition_csv_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for condition, values in result["conditions"].items():
        for metric in ("pass_at_1", "mean_test_pass_rate"):
            for series in ("base", "trained", "delta_trained_minus_base"):
                summary = values[metric][series]
                rows.append(
                    {
                        "condition": condition,
                        "metric": metric,
                        "series": series,
                        **summary,
                    }
                )
    return rows


def _contrast_csv_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, values in result["planned_contrasts"].items():
        for metric, summary in values["metrics"].items():
            rows.append({"contrast": name, "metric": metric, **summary})
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flattened = []
    for row in rows:
        flattened.append(
            {
                key: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
                for key, value in row.items()
            }
        )
    fieldnames = list(dict.fromkeys(key for row in flattened for key in row))
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)
    temporary.replace(path)


def _markdown(result: dict[str, Any]) -> str:
    protocol = result["protocol"]
    lines = [
        "# Priority bridge: three-seed analysis",
        "",
        f"Seeds: {', '.join(map(str, protocol['seeds']))}; tasks: {protocol['n_tasks']}; "
        f"checkpoints: {', '.join(map(str, protocol['checkpoints']))}.",
        "Generation seed slots are matched to the shared eight-sample base.",
        "",
        "## Primary Pass@1",
        "",
        "| Condition | Trained | Delta vs base | Two-way 95% CI | Seed-t 95% CI | Per seed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition, values in result["conditions"].items():
        trained = values["pass_at_1"]["trained"]
        delta = values["pass_at_1"]["delta_trained_minus_base"]
        per_seed = ", ".join(
            f"{seed}: {_pct(value, signed=True)}" for seed, value in delta["by_seed"].items()
        )
        lines.append(
            f"| `{condition}` | {_pct(trained['estimate'])} | {_pct(delta['estimate'], signed=True)} | "
            f"[{_pct(delta['two_way_bootstrap_ci95_low'], signed=True)}, "
            f"{_pct(delta['two_way_bootstrap_ci95_high'], signed=True)}] | "
            f"[{_pct(delta['seed_t_ci95_low'], signed=True)}, "
            f"{_pct(delta['seed_t_ci95_high'], signed=True)}] | {per_seed} |"
        )

    lines.extend(
        [
            "",
            "## Planned Pass@1 contrasts",
            "",
            "| Contrast | Estimate | Two-way 95% CI | Seed-t 95% CI |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, values in result["planned_contrasts"].items():
        summary = values["metrics"]["pass_at_1"]
        lines.append(
            f"| `{name}` | {_pct(summary['estimate'], signed=True)} | "
            f"[{_pct(summary['two_way_bootstrap_ci95_low'], signed=True)}, "
            f"{_pct(summary['two_way_bootstrap_ci95_high'], signed=True)}] | "
            f"[{_pct(summary['seed_t_ci95_low'], signed=True)}, "
            f"{_pct(summary['seed_t_ci95_high'], signed=True)}] |"
        )
    lines.extend(
        [
            "",
            "Intervals for planned contrasts are unadjusted. With only three training seeds, "
            "the seed-t interval is the conservative basis for population-level claims; the "
            "two-way interval additionally represents task-composition uncertainty.",
            "",
            "## Training-signal diagnostics",
            "",
            "| Condition | Nonzero advantage | Zero-variance groups | Sign agreement | Pairwise fidelity |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for condition, metrics in result["training_signal"]["conditions"].items():
        lines.append(
            f"| `{condition}` | {_pct(metrics['nonzero_train_advantage']['mean'])} | "
            f"{_pct(metrics['zero_variance_group_fraction']['mean'])} | "
            f"{_pct(metrics['mean_advantage_sign_agreement']['mean'])} | "
            f"{_pct(metrics['mean_pairwise_fidelity']['mean'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def _pct(value: float, *, signed: bool = False) -> str:
    prefix = "+" if signed and value >= 0 else ""
    return f"{prefix}{100 * value:.2f}%"


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


if __name__ == "__main__":
    main()
