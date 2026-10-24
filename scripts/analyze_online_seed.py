#!/usr/bin/env python3
"""Validate and summarize one completed online GRPO seed matrix.

The script is intentionally CPU-only.  It consumes the immutable run artifacts,
checks the frozen main-run contract, and writes small, reviewable tables, plots,
and manifests.  Raw rollouts and model checkpoints remain in the matrix folder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

EXPECTED_CONDITIONS = (
    "clean__binary",
    "clean__fractional",
    "fp_only__binary",
    "fp_only__fractional",
    "fn_only__binary",
    "fn_only__fractional",
)
EXPECTED_CHECKPOINTS = (37, 75, 112, 150, 187, 224, 262, 299, 337, 374)
EXPECTED_TEST_STEPS = (337, 374)
EXPECTED_MODEL_REVISION = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
GROUP_SIZE = 16
TOTAL_STEPS = 374
VALIDATION_PROBLEMS = 90
TEST_PROBLEMS = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze_matrix(args.matrix_dir.resolve())
    write_report(args.report_dir.resolve(), report)
    print(json.dumps({"report_dir": str(args.report_dir.resolve()), **report["summary"]}, indent=2))


def analyze_matrix(matrix_dir: Path) -> dict[str, Any]:
    matrix_status = _read_json(matrix_dir / "matrix_status.json")
    errors: list[str] = []
    warnings: list[str] = []

    if matrix_status.get("status") != "complete":
        errors.append(f"matrix status is {matrix_status.get('status')!r}, expected 'complete'")
    if matrix_status.get("error") is not None:
        errors.append(f"matrix error is not null: {matrix_status['error']!r}")
    if tuple(matrix_status.get("conditions", ())) != EXPECTED_CONDITIONS:
        errors.append("matrix condition order does not match the frozen priority_bridge order")
    if tuple(matrix_status.get("completed_conditions", ())) != EXPECTED_CONDITIONS:
        errors.append("completed condition list is incomplete or out of order")
    if matrix_status.get("seed") != 0:
        errors.append(f"matrix seed is {matrix_status.get('seed')!r}, expected 0")

    rows: list[dict[str, Any]] = []
    validations: dict[str, Any] = {}
    manifest: dict[str, Any] = {}
    for condition in EXPECTED_CONDITIONS:
        run_dir = matrix_dir / condition
        row, validation, condition_manifest = _analyze_condition(run_dir, condition)
        rows.append(row)
        validations[condition] = validation
        manifest[condition] = condition_manifest
        errors.extend(f"{condition}: {message}" for message in validation["errors"])
        warnings.extend(f"{condition}: {message}" for message in validation["warnings"])

    baseline_by_reward = {
        row["reward_type"]: row["primary_pass_at_1"]
        for row in rows
        if row["noise"] == "clean"
    }
    for row in rows:
        row["delta_vs_clean"] = row["primary_pass_at_1"] - baseline_by_reward[
            row["reward_type"]
        ]

    reward_deltas = []
    for noise in ("clean", "fp_only", "fn_only"):
        by_reward = {row["reward_type"]: row for row in rows if row["noise"] == noise}
        reward_deltas.append(
            {
                "noise": noise,
                "fractional_minus_binary": (
                    by_reward["fractional"]["primary_pass_at_1"]
                    - by_reward["binary"]["primary_pass_at_1"]
                ),
            }
        )

    max_abs_noise_delta = max(
        abs(row["delta_vs_clean"]) for row in rows if row["noise"] != "clean"
    )
    fractional_deltas = [item["fractional_minus_binary"] for item in reward_deltas]
    by_condition = {row["condition"]: row for row in rows}
    zero_variance_reductions = {
        noise: (
            by_condition[f"{noise}__binary"]["zero_variance_group_fraction"]
            - by_condition[f"{noise}__fractional"]["zero_variance_group_fraction"]
        )
        for noise in ("clean", "fp_only", "fn_only")
    }
    fp_reward_shifts = {
        reward: (
            by_condition[f"fp_only__{reward}"]["mean_noisy_reward"]
            - by_condition[f"fp_only__{reward}"]["mean_clean_reward"]
        )
        for reward in ("binary", "fractional")
    }
    fn_reward_shifts = {
        reward: (
            by_condition[f"fn_only__{reward}"]["mean_noisy_reward"]
            - by_condition[f"fn_only__{reward}"]["mean_clean_reward"]
        )
        for reward in ("binary", "fractional")
    }
    conclusions = [
        (
            "Across this single seed, the largest absolute clean Pass@1 change from 30% "
            f"one-sided label noise was {max_abs_noise_delta * 100:.1f} percentage points."
        ),
        (
            "Fractional reward exceeded binary reward by "
            f"{min(fractional_deltas) * 100:.1f} to {max(fractional_deltas) * 100:.1f} "
            "percentage points across the three noise regimes."
        ),
        (
            "The corruption moved training rewards in the intended direction: FP-only raised "
            f"the noisy-minus-clean mean by {fp_reward_shifts['binary']:.3f} (binary) and "
            f"{fp_reward_shifts['fractional']:.3f} (fractional), while FN-only changed it by "
            f"{fn_reward_shifts['binary']:.3f} and {fn_reward_shifts['fractional']:.3f}."
        ),
        (
            "Fractional reward reduced zero-variance groups relative to binary by "
            f"{zero_variance_reductions['clean'] * 100:.1f} points for clean, "
            f"{zero_variance_reductions['fp_only'] * 100:.1f} for FP-only, and "
            f"{zero_variance_reductions['fn_only'] * 100:.1f} for FN-only; the largest "
            "signal-preservation difference appeared under FN noise."
        ),
        (
            "These are descriptive results from seed 0 only. The differences are small "
            "relative to checkpoint-to-checkpoint variation and are not evidence of a "
            "population-level effect."
        ),
    ]

    return {
        "schema_version": 1,
        "scope": "priority_bridge seed 0 only",
        "matrix_dir": str(matrix_dir),
        "matrix_status": matrix_status,
        "summary": {
            "valid": not errors,
            "condition_count": len(rows),
            "completed_condition_count": sum(item["complete"] for item in validations.values()),
            "error_count": len(errors),
            "warning_count": len(warnings),
            "single_seed": True,
        },
        "errors": errors,
        "warnings": warnings,
        "conditions": rows,
        "reward_deltas": reward_deltas,
        "conclusions": conclusions,
        "validation": validations,
        "artifact_manifest": manifest,
    }


def _analyze_condition(
    run_dir: Path, condition: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    required = (
        "events.jsonl",
        "primary_metric_summary.json",
        "resolved_config.yaml",
        "resource_report.json",
        "rollouts.jsonl",
        "run_metadata.json",
        "status.json",
        "trainer_state.json",
    )
    for name in required:
        path = run_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty required artifact: {name}")

    status = _read_json(run_dir / "status.json")
    metadata = _read_json(run_dir / "run_metadata.json")
    primary = _read_json(run_dir / "primary_metric_summary.json")
    resources = _read_json(run_dir / "resource_report.json")
    trainer_state = _read_json(run_dir / "trainer_state.json")
    context = metadata.get("context", {})

    expected_noise, expected_reward = condition.split("__", maxsplit=1)
    checks = {
        "status_complete": status.get("status") == "complete" and status.get("stage") == "complete",
        "no_run_error": status.get("error") is None and resources.get("error") is None,
        "steps_complete": status.get("current_step") == TOTAL_STEPS
        and status.get("total_steps") == TOTAL_STEPS,
        "trainer_state_complete": trainer_state.get("global_step") == TOTAL_STEPS,
        "context_matches": (
            context.get("experiment") == "priority_bridge"
            and context.get("profile") == "main"
            and context.get("seed") == 0
            and context.get("condition") == condition
            and context.get("noise") == expected_noise
            and context.get("reward") == expected_reward
        ),
        "revision_matches": context.get("model_revision") == EXPECTED_MODEL_REVISION,
        "primary_complete": (
            primary.get("split") == "test"
            and primary.get("metric") == "pass_at_1"
            and primary.get("selected_steps") == list(EXPECTED_TEST_STEPS)
            and len(primary.get("values", ())) == 2
        ),
    }
    for name, passed in checks.items():
        if not passed:
            errors.append(f"failed validation check: {name}")

    checkpoint_steps = sorted(
        int(path.name.removeprefix("checkpoint-"))
        for path in run_dir.glob("checkpoint-*")
        if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
    )
    if checkpoint_steps != list(EXPECTED_CHECKPOINTS):
        errors.append(f"checkpoint steps differ: {checkpoint_steps}")

    checkpoint_manifest: dict[str, Any] = {}
    for step in checkpoint_steps:
        checkpoint = run_dir / f"checkpoint-{step}"
        checkpoint_state = checkpoint / "trainer_state.json"
        adapter = checkpoint / "adapter_model.safetensors"
        if not checkpoint_state.is_file() or not adapter.is_file():
            errors.append(f"checkpoint-{step} is missing trainer state or adapter weights")
            continue
        state = _read_json(checkpoint_state)
        if state.get("global_step") != step:
            errors.append(f"checkpoint-{step} trainer global_step is {state.get('global_step')}")
        checkpoint_manifest[str(step)] = {
            "trainer_state_size": checkpoint_state.stat().st_size,
            "trainer_state_sha256": _sha256(checkpoint_state),
            "adapter_size": adapter.stat().st_size,
            "adapter_sha256": _sha256(adapter),
        }

    early_hash = checkpoint_manifest.get(str(EXPECTED_CHECKPOINTS[0]), {}).get("adapter_sha256")
    final_hash = checkpoint_manifest.get(str(EXPECTED_CHECKPOINTS[-1]), {}).get("adapter_sha256")
    adapter_changed = bool(early_hash and final_hash and early_hash != final_hash)
    if not adapter_changed:
        errors.append("adapter weights did not change between first and final checkpoint")

    rollout_summary = _summarize_rollouts(run_dir / "rollouts.jsonl", expected_reward)
    if rollout_summary["row_count"] != TOTAL_STEPS * GROUP_SIZE:
        errors.append(
            f"rollout row count is {rollout_summary['row_count']}, "
            f"expected {TOTAL_STEPS * GROUP_SIZE}"
        )
    if rollout_summary["group_count"] != TOTAL_STEPS:
        errors.append(
            f"rollout group count is {rollout_summary['group_count']}, expected {TOTAL_STEPS}"
        )
    if rollout_summary["invalid_group_sizes"]:
        errors.append(f"invalid group sizes: {rollout_summary['invalid_group_sizes'][:5]}")
    if rollout_summary["nonfinite_value_count"]:
        errors.append(
            f"rollouts contain {rollout_summary['nonfinite_value_count']} non-finite values"
        )

    metrics = _summarize_metric_events(run_dir / "events.jsonl")
    if metrics["optimizer_step_count"] != TOTAL_STEPS:
        errors.append(
            f"optimizer metric steps are {metrics['optimizer_step_count']}, expected {TOTAL_STEPS}"
        )
    if metrics["nonzero_grad_step_count"] == 0:
        errors.append("no optimizer step recorded a non-zero gradient norm")

    evaluation = _validate_evaluations(run_dir)
    errors.extend(evaluation["errors"])

    if resources.get("status") != "complete":
        errors.append(f"resource report status is {resources.get('status')!r}")
    gpu = resources.get("gpu", {})
    if not gpu.get("available"):
        errors.append("resource report does not record an available GPU")
    if int(resources.get("training_generated_completion_tokens", 0)) <= 0:
        errors.append("resource report has no generated training tokens")

    row = {
        "condition": condition,
        "noise": expected_noise,
        "reward_type": expected_reward,
        "primary_pass_at_1": float(primary["mean"]),
        "step_337_pass_at_1": float(primary["values"][0]),
        "step_374_pass_at_1": float(primary["values"][1]),
        "hard_pass_at_1": float(primary["by_difficulty"]["hard"]["mean"]),
        "medium_pass_at_1": float(primary["by_difficulty"]["medium"]["mean"]),
        "easy_pass_at_1": float(primary["by_difficulty"]["easy"]["mean"]),
        "mean_clean_reward": rollout_summary["mean_selected_clean_reward"],
        "mean_noisy_reward": rollout_summary["mean_selected_noisy_reward"],
        "mean_abs_advantage": rollout_summary["mean_abs_train_advantage"],
        "nonzero_advantage_fraction": rollout_summary["nonzero_train_advantage_fraction"],
        "zero_variance_group_fraction": rollout_summary["zero_variance_group_fraction"],
        "changed_label_rate": rollout_summary["mean_changed_label_rate"],
        "mean_pairwise_fidelity": rollout_summary["mean_pairwise_fidelity"],
        "mean_advantage_sign_agreement": rollout_summary["mean_advantage_sign_agreement"],
        "nonzero_grad_step_fraction": metrics["nonzero_grad_step_count"] / TOTAL_STEPS,
        "train_loss": float(resources["train_metrics"]["train_loss"]),
        "train_runtime_hours": float(resources["train_metrics"]["train_runtime"]) / 3600.0,
        "generated_completion_tokens": int(resources["training_generated_completion_tokens"]),
        "peak_gpu_allocated_gib": float(gpu["peak_allocated_bytes"]) / 2**30,
        "peak_gpu_reserved_gib": float(gpu["peak_reserved_bytes"]) / 2**30,
        "peak_host_rss_gib": float(resources["host_peak_rss_bytes"]) / 2**30,
        "artifact_size_mib": float(resources["output_size_bytes"]) / 2**20,
        "adapter_changed": adapter_changed,
    }
    validation = {
        "complete": not errors,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "checkpoint_steps": checkpoint_steps,
        "adapter_changed": adapter_changed,
        "rollouts": rollout_summary,
        "metrics": metrics,
        "evaluations": evaluation,
    }
    manifest = {
        "root_artifacts": {
            name: _file_manifest(run_dir / name) for name in required if (run_dir / name).is_file()
        },
        "checkpoints": checkpoint_manifest,
    }
    return row, validation, manifest


def _summarize_rollouts(path: Path, reward_type: str) -> dict[str, Any]:
    values: dict[str, list[float]] = defaultdict(list)
    groups: dict[str, dict[str, Any]] = {}
    group_sizes: dict[str, int] = defaultdict(int)
    nonfinite = 0
    clean_key = f"clean_{reward_type}_reward"
    noisy_key = f"noisy_{reward_type}_reward"
    keys = (clean_key, noisy_key, "train_reward", "train_advantage")
    row_count = 0
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row_count += 1
            row = json.loads(line)
            group_id = str(row["group_id"])
            group_sizes[group_id] += 1
            groups.setdefault(group_id, row)
            for key in keys:
                value = float(row[key])
                if not math.isfinite(value):
                    nonfinite += 1
                values[key].append(value)
            if row.get("condition") is None:
                raise RuntimeError(f"missing condition in {path}:{line_number}")

    group_rows = list(groups.values())
    invalid_group_sizes = sorted(
        (group_id, size) for group_id, size in group_sizes.items() if size != GROUP_SIZE
    )
    abs_advantages = [abs(value) for value in values["train_advantage"]]
    return {
        "row_count": row_count,
        "group_count": len(groups),
        "invalid_group_sizes": invalid_group_sizes,
        "nonfinite_value_count": nonfinite,
        "mean_selected_clean_reward": _mean(values[clean_key]),
        "mean_selected_noisy_reward": _mean(values[noisy_key]),
        "mean_train_reward": _mean(values["train_reward"]),
        "mean_abs_train_advantage": _mean(abs_advantages),
        "nonzero_train_advantage_fraction": _mean(
            [float(value > 1e-12) for value in abs_advantages]
        ),
        "zero_variance_group_count": sum(bool(row["zero_variance_group"]) for row in group_rows),
        "zero_variance_group_fraction": _mean(
            [float(bool(row["zero_variance_group"])) for row in group_rows]
        ),
        "mean_changed_label_rate": _mean(
            [float(row["changed_label_rate"]) for row in group_rows]
        ),
        "mean_pairwise_fidelity": _finite_mean(
            row.get("train_aggregation_pairwise_fidelity") for row in group_rows
        ),
        "mean_advantage_sign_agreement": _finite_mean(
            row.get("train_aggregation_advantage_sign_agreement") for row in group_rows
        ),
    }


def _summarize_metric_events(path: Path) -> dict[str, Any]:
    metric_steps: set[int] = set()
    grad_norms: list[float] = []
    checkpoint_steps: list[int] = []
    run_complete_count = 0
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            event = row.get("event")
            if event == "metrics" and "grad_norm" in row.get("metrics", {}):
                step = int(row["step"])
                metric_steps.add(step)
                grad_norms.append(float(row["metrics"]["grad_norm"]))
            elif event == "checkpoint":
                checkpoint_steps.append(int(row["step"]))
            elif event == "run_complete":
                run_complete_count += 1
    return {
        "optimizer_step_count": len(metric_steps),
        "first_optimizer_step": min(metric_steps) if metric_steps else None,
        "last_optimizer_step": max(metric_steps) if metric_steps else None,
        "nonzero_grad_step_count": sum(value > 0.0 for value in grad_norms),
        "max_grad_norm": max(grad_norms, default=0.0),
        "checkpoint_event_steps": checkpoint_steps,
        "run_complete_event_count": run_complete_count,
    }


def _validate_evaluations(run_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    details: dict[str, Any] = {}
    expected = {
        "validation": {step: VALIDATION_PROBLEMS for step in EXPECTED_CHECKPOINTS},
        "test": {step: TEST_PROBLEMS for step in EXPECTED_TEST_STEPS},
    }
    for split, steps in expected.items():
        split_details: dict[str, Any] = {}
        actual_steps = sorted(
            int(path.name.removeprefix("step-"))
            for path in (run_dir / "evaluations" / split).glob("step-*")
            if path.is_dir() and path.name.removeprefix("step-").isdigit()
        )
        if actual_steps != sorted(steps):
            errors.append(f"{split} evaluation steps differ: {actual_steps}")
        for step, expected_count in steps.items():
            root = run_dir / "evaluations" / split / f"step-{step}"
            summary_path = root / "summary.json"
            predictions_path = root / "predictions.jsonl"
            if not summary_path.is_file() or not predictions_path.is_file():
                errors.append(f"missing {split} evaluation artifacts at step {step}")
                continue
            summary = _read_json(summary_path)
            prediction_count, unique_tasks = _jsonl_task_counts(predictions_path)
            if summary.get("n_problems") != expected_count:
                errors.append(
                    f"{split} step {step} summary count is {summary.get('n_problems')}, "
                    f"expected {expected_count}"
                )
            if prediction_count != expected_count or unique_tasks != expected_count:
                errors.append(
                    f"{split} step {step} predictions are {prediction_count} rows / "
                    f"{unique_tasks} tasks, expected {expected_count}"
                )
            split_details[str(step)] = {
                "prediction_count": prediction_count,
                "unique_task_count": unique_tasks,
                "pass_at_1": float(summary["pass_at_1"]),
                "mean_test_pass_rate": float(summary["mean_test_pass_rate"]),
            }
        details[split] = split_details
    return {"errors": errors, "details": details}


def write_report(report_dir: Path, report: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    (report_dir / "artifact_manifest.json").write_text(
        json.dumps(report["artifact_manifest"], indent=2) + "\n"
    )
    _write_csv(report_dir / "condition_summary.csv", report["conditions"])
    (report_dir / "condition_summary.md").write_text(_markdown_table(report["conditions"]))
    (report_dir / "report.md").write_text(_markdown_report(report))
    (report_dir / "pass_at_1.svg").write_text(_pass_at_1_svg(report["conditions"]))
    (report_dir / "training_diagnostics.svg").write_text(
        _training_diagnostics_svg(report["conditions"])
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = (
        "condition",
        "primary_pass_at_1",
        "delta_vs_clean",
        "mean_clean_reward",
        "mean_noisy_reward",
        "mean_abs_advantage",
        "zero_variance_group_fraction",
        "changed_label_rate",
    )
    labels = (
        "Condition",
        "Clean Pass@1",
        "Δ vs clean",
        "Mean clean reward",
        "Mean noisy reward",
        "Mean abs. advantage",
        "Zero-var groups",
        "Changed labels",
    )
    lines = ["| " + " | ".join(labels) + " |", "|" + "---|" * len(labels)]
    for row in rows:
        cells = [row["condition"]]
        cells.extend(f"{float(row[key]):.4f}" for key in headers[1:])
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _markdown_report(report: dict[str, Any]) -> str:
    matrix = report["matrix_status"]
    lines = [
        "# Priority bridge — seed 0 report",
        "",
        "> **Scope:** one seed only. All comparisons are descriptive and must not be "
        "interpreted as uncertainty-calibrated effects.",
        "",
        "## Completion and integrity",
        "",
        f"- Matrix status: **{matrix['status']}** ({matrix['completed_count']}/"
        f"{matrix['total_conditions']} conditions).",
        f"- Validation: **{'passed' if report['summary']['valid'] else 'failed'}**; "
        f"{report['summary']['error_count']} errors and {report['summary']['warning_count']} warnings.",
        f"- Each condition contains {TOTAL_STEPS} optimizer steps, {len(EXPECTED_CHECKPOINTS)} "
        f"scheduled checkpoints, {TOTAL_STEPS * GROUP_SIZE} rollout rows, ten 90-problem "
        "validation evaluations, and two 500-problem test evaluations.",
        "- Adapter hashes differ between checkpoint 37 and checkpoint 374 in every run, "
        "and every run records non-zero gradient steps.",
        "",
        "## Results",
        "",
        _markdown_table(report["conditions"]).rstrip(),
        "",
        "The primary metric is the mean clean test Pass@1 over checkpoints 337 and 374.",
        "",
        "## Descriptive conclusions",
        "",
    ]
    lines.extend(f"- {item}" for item in report["conclusions"])
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `analysis.json`: complete machine-readable analysis and validation details.",
            "- `artifact_manifest.json`: SHA-256 hashes and sizes for source artifacts and checkpoints.",
            "- `condition_summary.csv` / `.md`: compact result tables.",
            "- `pass_at_1.svg`: primary-metric comparison.",
            "- `training_diagnostics.svg`: advantage signal and zero-variance groups.",
            "",
        ]
    )
    if report["errors"]:
        lines.extend(["## Validation errors", ""] + [f"- {item}" for item in report["errors"]])
        lines.append("")
    return "\n".join(lines)


def _pass_at_1_svg(rows: list[dict[str, Any]]) -> str:
    labels = [row["condition"].replace("__", "\n") for row in rows]
    values = [float(row["primary_pass_at_1"]) for row in rows]
    colors = ["#2563eb" if row["reward_type"] == "binary" else "#f97316" for row in rows]
    return _bar_svg(
        title="Clean test Pass@1 — priority_bridge seed 0",
        subtitle="Mean of checkpoints 337 and 374; one seed, descriptive only",
        labels=labels,
        series=[("Pass@1", values, colors)],
        y_max=max(0.55, max(values) * 1.15),
        value_format=lambda value: f"{value:.3f}",
    )


def _training_diagnostics_svg(rows: list[dict[str, Any]]) -> str:
    labels = [row["condition"].replace("__", "\n") for row in rows]
    advantages = [float(row["mean_abs_advantage"]) for row in rows]
    zero_variance = [float(row["zero_variance_group_fraction"]) for row in rows]
    return _bar_svg(
        title="Training signal diagnostics — priority_bridge seed 0",
        subtitle="Mean absolute selected advantage and fraction of zero-variance groups",
        labels=labels,
        series=[
            ("Mean |advantage|", advantages, ["#7c3aed"] * len(rows)),
            ("Zero-var fraction", zero_variance, ["#14b8a6"] * len(rows)),
        ],
        y_max=max(1.0, max(advantages + zero_variance) * 1.15),
        value_format=lambda value: f"{value:.2f}",
    )


def _bar_svg(
    *,
    title: str,
    subtitle: str,
    labels: list[str],
    series: list[tuple[str, list[float], list[str]]],
    y_max: float,
    value_format,
) -> str:
    width, height = 1000, 560
    left, right, top, bottom = 90, 30, 95, 115
    plot_w, plot_h = width - left - right, height - top - bottom
    group_w = plot_w / len(labels)
    bar_w = min(48.0, group_w * 0.7 / len(series))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="36" font-family="sans-serif" font-size="24" font-weight="700" fill="#111827">{_xml(title)}</text>',
        f'<text x="{left}" y="62" font-family="sans-serif" font-size="14" fill="#4b5563">{_xml(subtitle)}</text>',
    ]
    for tick in range(6):
        value = y_max * tick / 5
        y = top + plot_h - (value / y_max) * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(
            f'<text x="{left-10}" y="{y+5:.1f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#6b7280">{value:.2f}</text>'
        )
    for index, label in enumerate(labels):
        center = left + group_w * (index + 0.5)
        offset_start = -(len(series) * bar_w + (len(series) - 1) * 6) / 2
        for series_index, (_, values, colors) in enumerate(series):
            value = values[index]
            x = center + offset_start + series_index * (bar_w + 6)
            bar_h = max(0.0, value / y_max * plot_h)
            y = top + plot_h - bar_h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="3" fill="{colors[index]}"/>'
            )
            parts.append(
                f'<text x="{x + bar_w/2:.1f}" y="{max(top+12, y-6):.1f}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#374151">{_xml(value_format(value))}</text>'
            )
        first, second = label.split("\n", maxsplit=1)
        parts.append(
            f'<text x="{center:.1f}" y="{top+plot_h+26}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#111827">{_xml(first)}</text>'
        )
        parts.append(
            f'<text x="{center:.1f}" y="{top+plot_h+43}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#4b5563">{_xml(second)}</text>'
        )
    legend_x = left
    legend_y = height - 30
    for name, _, colors in series:
        parts.append(f'<rect x="{legend_x}" y="{legend_y-11}" width="14" height="14" rx="2" fill="{colors[0]}"/>')
        parts.append(
            f'<text x="{legend_x+22}" y="{legend_y}" font-family="sans-serif" font-size="13" fill="#374151">{_xml(name)}</text>'
        )
        legend_x += 190
    parts.append("</svg>\n")
    return "\n".join(parts)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _jsonl_task_counts(path: Path) -> tuple[int, int]:
    count = 0
    tasks: set[int] = set()
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            count += 1
            tasks.add(int(json.loads(line)["task_id"]))
    return count, len(tasks)


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _finite_mean(values) -> float:
    selected = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            selected.append(number)
    return _mean(selected)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_manifest(path: Path) -> dict[str, Any]:
    return {"size": path.stat().st_size, "sha256": _sha256(path)}


def _xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    main()
