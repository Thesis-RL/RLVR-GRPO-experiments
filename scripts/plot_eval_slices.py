#!/usr/bin/env python3
"""Build evaluation tables and dependency-free SVG figures for a seed matrix."""

from __future__ import annotations

import argparse
import csv
import html
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

CONDITIONS = (
    "clean__binary",
    "clean__fractional",
    "fp_only__binary",
    "fp_only__fractional",
    "fn_only__binary",
    "fn_only__fractional",
)
VALIDATION_STEPS = (37, 75, 112, 150, 187, 224, 262, 299, 337, 374)
TEST_STEPS = (337, 374)
DIFFICULTIES = ("hard", "medium", "easy")
DIFFICULTY_COUNTS = {"hard": 220, "medium": 130, "easy": 150}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument(
        "--baseline-grades",
        type=Path,
        required=True,
        help="Frozen base-model test clean_grades.jsonl with 16 completions per task.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix_dir = args.matrix_dir.resolve()
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    data = load_evaluations(matrix_dir)
    baseline = load_pretraining_baseline(args.baseline_grades.resolve())
    write_long_csv(report_dir / "evaluation_metrics.csv", data, baseline)
    write_baseline_csv(report_dir / "pretraining_baseline.csv", baseline)
    summaries = summarize_conditions(data)
    write_summary_csv(report_dir / "evaluation_metric_summary.csv", summaries)
    (report_dir / "eval_pass_at_1_by_condition.svg").write_text(
        small_multiples_svg(data), encoding="utf-8"
    )
    (report_dir / "test_pass_at_1_by_difficulty.svg").write_text(
        difficulty_heatmap_svg(data, baseline), encoding="utf-8"
    )
    (report_dir / "pass_at_1_vs_partial_credit.svg").write_text(
        strict_vs_partial_svg(data, baseline), encoding="utf-8"
    )
    (report_dir / "evaluation_slices.md").write_text(
        markdown_report(summaries, baseline), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "matrix_dir": str(matrix_dir),
                "report_dir": str(report_dir),
                "conditions": len(data),
                "overall_evaluation_points": sum(
                    len(item["validation"]) + len(item["test"]) for item in data.values()
                ),
                "pretraining_test_completions": baseline["all"]["n_completions"],
            },
            indent=2,
        )
    )


def load_evaluations(matrix_dir: Path) -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        run_dir = matrix_dir / condition
        if not run_dir.is_dir():
            raise FileNotFoundError(f"missing condition directory: {run_dir}")
        condition_data: dict[str, Any] = {
            "noise": condition.split("__", maxsplit=1)[0],
            "reward_type": condition.split("__", maxsplit=1)[1],
        }
        for split, expected_steps, expected_n in (
            ("validation", VALIDATION_STEPS, 90),
            ("test", TEST_STEPS, 500),
        ):
            points: list[dict[str, Any]] = []
            for step in expected_steps:
                path = run_dir / "evaluations" / split / f"step-{step}" / "summary.json"
                summary = _read_json(path)
                if int(summary["step"]) != step:
                    raise ValueError(f"step mismatch in {path}")
                if int(summary["n_problems"]) != expected_n:
                    raise ValueError(f"problem-count mismatch in {path}")
                points.append(summary)
            condition_data[split] = points
        data[condition] = condition_data
    return data


def load_pretraining_baseline(path: Path) -> dict[str, dict[str, Any]]:
    """Aggregate the frozen base-model test map into approximate step-0 metrics."""
    rewards: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"binary": [], "fractional": []}
    )
    task_counts: dict[str, int] = defaultdict(int)
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            stratum = str(row["stratum"])
            if stratum not in DIFFICULTIES:
                raise ValueError(f"unknown stratum {stratum!r} at {path}:{line_number}")
            binary = [float(value) for value in row["binary_rewards"]]
            fractional = [float(value) for value in row["fractional_rewards"]]
            if len(binary) != 16 or len(fractional) != 16:
                raise ValueError(f"expected 16 completions at {path}:{line_number}")
            for key in (stratum, "all"):
                rewards[key]["binary"].extend(binary)
                rewards[key]["fractional"].extend(fractional)
                task_counts[key] += 1

    expected_counts = {"all": sum(DIFFICULTY_COUNTS.values()), **DIFFICULTY_COUNTS}
    if dict(task_counts) != expected_counts:
        raise ValueError(
            f"baseline task counts mismatch: got {dict(task_counts)}, expected {expected_counts}"
        )

    return {
        difficulty: {
            "n_problems": task_counts[difficulty],
            "n_completions": len(rewards[difficulty]["binary"]),
            "pass_at_1": statistics.fmean(rewards[difficulty]["binary"]),
            "mean_test_pass_rate": statistics.fmean(rewards[difficulty]["fractional"]),
        }
        for difficulty in ("all", *DIFFICULTIES)
    }


def write_long_csv(
    path: Path,
    data: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
) -> None:
    fieldnames = (
        "condition",
        "noise",
        "reward_type",
        "split",
        "step",
        "difficulty",
        "n_problems",
        "pass_at_1",
        "mean_test_pass_rate",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for difficulty in ("all", *DIFFICULTIES):
            metrics = baseline[difficulty]
            writer.writerow(
                {
                    "condition": "pretraining_base_estimate",
                    "noise": "clean",
                    "reward_type": "base_model",
                    "split": "test_baseline_estimate",
                    "step": 0,
                    "difficulty": difficulty,
                    "n_problems": metrics["n_problems"],
                    "pass_at_1": metrics["pass_at_1"],
                    "mean_test_pass_rate": metrics["mean_test_pass_rate"],
                }
            )
        for condition in CONDITIONS:
            item = data[condition]
            for split in ("validation", "test"):
                for point in item[split]:
                    base = {
                        "condition": condition,
                        "noise": item["noise"],
                        "reward_type": item["reward_type"],
                        "split": split,
                        "step": point["step"],
                    }
                    writer.writerow(
                        {
                            **base,
                            "difficulty": "all",
                            "n_problems": point["n_problems"],
                            "pass_at_1": point["pass_at_1"],
                            "mean_test_pass_rate": point["mean_test_pass_rate"],
                        }
                    )
                    for difficulty in DIFFICULTIES:
                        if difficulty not in point.get("by_difficulty", {}):
                            continue
                        metrics = point["by_difficulty"][difficulty]
                        writer.writerow(
                            {
                                **base,
                                "difficulty": difficulty,
                                "n_problems": metrics["n_problems"],
                                "pass_at_1": metrics["pass_at_1"],
                                "mean_test_pass_rate": metrics["mean_test_pass_rate"],
                            }
                        )


def write_baseline_csv(path: Path, baseline: dict[str, dict[str, Any]]) -> None:
    fieldnames = (
        "difficulty",
        "n_problems",
        "n_completions",
        "pass_at_1_estimate",
        "mean_test_pass_rate",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for difficulty in ("all", *DIFFICULTIES):
            metrics = baseline[difficulty]
            writer.writerow(
                {
                    "difficulty": difficulty,
                    "n_problems": metrics["n_problems"],
                    "n_completions": metrics["n_completions"],
                    "pass_at_1_estimate": metrics["pass_at_1"],
                    "mean_test_pass_rate": metrics["mean_test_pass_rate"],
                }
            )


def summarize_conditions(data: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        item = data[condition]
        validation = item["validation"]
        test = item["test"]
        val_by_step = {int(point["step"]): point for point in validation}
        test_by_step = {int(point["step"]): point for point in test}
        pass_values = [float(point["pass_at_1"]) for point in validation]
        test_pass = [float(point["pass_at_1"]) for point in test]
        partial = [float(point["mean_test_pass_rate"]) for point in test]
        summary: dict[str, Any] = {
            "condition": condition,
            "noise": item["noise"],
            "reward_type": item["reward_type"],
            "validation_step_37": val_by_step[37]["pass_at_1"],
            "validation_step_374": val_by_step[374]["pass_at_1"],
            "validation_min": min(pass_values),
            "validation_max": max(pass_values),
            "validation_change": val_by_step[374]["pass_at_1"] - val_by_step[37]["pass_at_1"],
            "test_step_337": test_by_step[337]["pass_at_1"],
            "test_step_374": test_by_step[374]["pass_at_1"],
            "test_change": test_by_step[374]["pass_at_1"] - test_by_step[337]["pass_at_1"],
            "primary_test_pass_at_1": statistics.fmean(test_pass),
            "mean_test_pass_rate": statistics.fmean(partial),
            "partial_credit_gap": statistics.fmean(partial) - statistics.fmean(test_pass),
            "validation_test_gap_step_337": val_by_step[337]["pass_at_1"]
            - test_by_step[337]["pass_at_1"],
            "validation_test_gap_step_374": val_by_step[374]["pass_at_1"]
            - test_by_step[374]["pass_at_1"],
        }
        for difficulty in DIFFICULTIES:
            summary[f"test_{difficulty}_pass_at_1"] = statistics.fmean(
                float(point["by_difficulty"][difficulty]["pass_at_1"]) for point in test
            )
        summaries.append(summary)
    return summaries


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(summaries[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summaries)


def small_multiples_svg(data: dict[str, dict[str, Any]]) -> str:
    width, height = 1280, 980
    margin_x, margin_top = 75, 125
    gap_x, gap_y = 48, 58
    panel_w = (width - 2 * margin_x - gap_x) / 2
    panel_h = (height - margin_top - 65 - 2 * gap_y) / 3
    y_min, y_max = 0.44, 0.60
    parts = _svg_start(width, height)
    parts += [
        _text(65, 38, "Clean Pass@1 across checkpoints", 26, weight=700),
        _text(
            65,
            67,
            "Validation: 90 tasks at 10 checkpoints. Test: 500 tasks at steps 337 and 374 only.",
            15,
            color="#475569",
        ),
        _legend(65, 96, (("Validation", "#2563eb"), ("Test", "#f97316"))),
    ]
    for index, condition in enumerate(CONDITIONS):
        col, row = index % 2, index // 2
        x0 = margin_x + col * (panel_w + gap_x)
        y0 = margin_top + row * (panel_h + gap_y)
        parts.extend(
            _line_panel(
                x0,
                y0,
                panel_w,
                panel_h,
                condition,
                data[condition]["validation"],
                data[condition]["test"],
                y_min,
                y_max,
            )
        )
    parts.append("</svg>\n")
    return "\n".join(parts)


def _line_panel(
    x0: float,
    y0: float,
    width: float,
    height: float,
    title: str,
    validation: list[dict[str, Any]],
    test: list[dict[str, Any]],
    y_min: float,
    y_max: float,
) -> list[str]:
    left, right, top, bottom = 48, 16, 29, 37
    plot_x, plot_y = x0 + left, y0 + top
    plot_w, plot_h = width - left - right, height - top - bottom

    def sx(step: int) -> float:
        return (
            plot_x
            + (step - VALIDATION_STEPS[0]) / (VALIDATION_STEPS[-1] - VALIDATION_STEPS[0]) * plot_w
        )

    def sy(value: float) -> float:
        return plot_y + (y_max - value) / (y_max - y_min) * plot_h

    parts = [
        f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{width:.1f}" height="{height:.1f}" rx="8" fill="#ffffff" stroke="#cbd5e1"/>',
        _text(x0 + 14, y0 + 20, title.replace("__", " / "), 15, weight=700),
    ]
    for value in (0.45, 0.50, 0.55, 0.60):
        y = sy(value)
        parts.append(
            f'<line x1="{plot_x:.1f}" y1="{y:.1f}" x2="{plot_x + plot_w:.1f}" y2="{y:.1f}" stroke="#e2e8f0"/>'
        )
        parts.append(_text(plot_x - 7, y + 4, f"{value:.2f}", 10, anchor="end", color="#64748b"))
    for step in (37, 112, 187, 262, 337, 374):
        x = sx(step)
        parts.append(
            f'<line x1="{x:.1f}" y1="{plot_y:.1f}" x2="{x:.1f}" y2="{plot_y + plot_h:.1f}" stroke="#f1f5f9"/>'
        )
        parts.append(_text(x, plot_y + plot_h + 17, str(step), 9, anchor="middle", color="#64748b"))
    val_points = [(sx(int(point["step"])), sy(float(point["pass_at_1"]))) for point in validation]
    test_points = [(sx(int(point["step"])), sy(float(point["pass_at_1"]))) for point in test]
    parts.append(_polyline(val_points, "#2563eb", 2.2))
    parts.append(_polyline(test_points, "#f97316", 2.8))
    for x, y in val_points:
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="#2563eb"/>')
    for x, y in test_points:
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.8" fill="#f97316" stroke="#ffffff" stroke-width="1.5"/>'
        )
    for point, (x, y) in zip(test, test_points, strict=True):
        parts.append(
            _text(
                x, y - 9, f"{float(point['pass_at_1']):.3f}", 10, anchor="middle", color="#9a3412"
            )
        )
    parts.append(
        _text(
            plot_x + plot_w / 2,
            y0 + height - 5,
            "optimizer step",
            10,
            anchor="middle",
            color="#64748b",
        )
    )
    return parts


def difficulty_heatmap_svg(
    data: dict[str, dict[str, Any]], baseline: dict[str, dict[str, Any]]
) -> str:
    width, height = 980, 690
    left, top, cell_w, cell_h = 285, 145, 205, 60
    values: dict[str, dict[str, float]] = {
        "pretraining estimate": {
            difficulty: float(baseline[difficulty]["pass_at_1"]) for difficulty in DIFFICULTIES
        },
        **{
            condition: {
                difficulty: statistics.fmean(
                    float(point["by_difficulty"][difficulty]["pass_at_1"])
                    for point in data[condition]["test"]
                )
                for difficulty in DIFFICULTIES
            }
            for condition in CONDITIONS
        },
    }
    rows = ("pretraining estimate", *CONDITIONS)
    parts = _svg_start(width, height)
    parts += [
        _text(55, 38, "Test Pass@1 by frozen difficulty stratum", 26, weight=700),
        _text(
            55,
            67,
            "Base estimate plus trained mean at checkpoints 337 and 374; one seed.",
            15,
            color="#475569",
        ),
    ]
    for col, difficulty in enumerate(DIFFICULTIES):
        x = left + col * cell_w
        parts.append(
            _text(x + cell_w / 2, top - 26, difficulty.title(), 15, anchor="middle", weight=700)
        )
        parts.append(
            _text(
                x + cell_w / 2,
                top - 8,
                f"n={DIFFICULTY_COUNTS[difficulty]}",
                11,
                anchor="middle",
                color="#64748b",
            )
        )
    for row, condition in enumerate(rows):
        y = top + row * cell_h
        label = "pretraining estimate*" if condition == "pretraining estimate" else condition
        parts.append(_text(left - 15, y + cell_h / 2 + 5, label, 13, anchor="end", weight=600))
        for col, difficulty in enumerate(DIFFICULTIES):
            value = values[condition][difficulty]
            x = left + col * cell_w
            color = _heat_color(value)
            text_color = "#ffffff" if value >= 0.55 else "#0f172a"
            parts.append(
                f'<rect x="{x + 3:.1f}" y="{y + 3:.1f}" width="{cell_w - 6:.1f}" height="{cell_h - 6:.1f}" rx="5" fill="{color}"/>'
            )
            parts.append(
                _text(
                    x + cell_w / 2,
                    y + cell_h / 2 + 6,
                    f"{value:.3f}",
                    17,
                    anchor="middle",
                    weight=700,
                    color=text_color,
                )
            )
        if condition == "pretraining estimate":
            separator_y = y + cell_h + 4
            parts.append(
                f'<line x1="{left - 235}" y1="{separator_y:.1f}" '
                f'x2="{left + 3 * cell_w}" y2="{separator_y:.1f}" '
                'stroke="#94a3b8" stroke-width="1.5"/>'
            )
    legend_y = top + len(rows) * cell_h + 35
    parts.append(_text(left, legend_y, "Pass@1", 11, color="#475569"))
    for index in range(101):
        value = index / 100
        parts.append(
            f'<rect x="{left + 70 + index * 4:.1f}" y="{legend_y - 13:.1f}" width="4.2" height="14" fill="{_heat_color(value)}"/>'
        )
    parts.append(_text(left + 65, legend_y + 18, "0", 10, color="#64748b"))
    parts.append(_text(left + 480, legend_y + 18, "1", 10, anchor="end", color="#64748b"))
    parts.append(
        _text(
            55,
            height - 18,
            "* Base: 16 samples/task at T=1.0; trained evaluation: 1 sample/task at T=0.7.",
            11,
            color="#64748b",
        )
    )
    parts.append("</svg>\n")
    return "\n".join(parts)


def strict_vs_partial_svg(
    data: dict[str, dict[str, Any]], baseline: dict[str, dict[str, Any]]
) -> str:
    width, height = 1320, 650
    left, right, top, bottom = 85, 30, 115, 125
    plot_w, plot_h = width - left - right, height - top - bottom
    groups = ("pretraining estimate", *CONDITIONS)
    group_w = plot_w / len(groups)
    bar_w = 44
    y_max = 0.60
    parts = _svg_start(width, height)
    parts += [
        _text(65, 38, "Strict task success vs partial test success", 26, weight=700),
        _text(
            65,
            67,
            "Base estimate at T=1.0; trained mean over steps 337/374 at T=0.7.",
            15,
            color="#475569",
        ),
        _legend(65, 98, (("Pass@1", "#0f766e"), ("Mean test pass rate", "#f59e0b"))),
    ]
    for tick in range(7):
        value = tick / 10
        y = top + plot_h - value / y_max * plot_h
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e2e8f0"/>'
        )
        parts.append(_text(left - 10, y + 4, f"{value:.1f}", 11, anchor="end", color="#64748b"))
    for index, condition in enumerate(groups):
        center = left + group_w * (index + 0.5)
        if condition == "pretraining estimate":
            strict = float(baseline["all"]["pass_at_1"])
            partial = float(baseline["all"]["mean_test_pass_rate"])
            noise, reward = "pretraining*", "base model"
        else:
            test = data[condition]["test"]
            strict = statistics.fmean(float(point["pass_at_1"]) for point in test)
            partial = statistics.fmean(float(point["mean_test_pass_rate"]) for point in test)
            noise, reward = condition.split("__", maxsplit=1)
        for offset, value, color in (
            (-bar_w - 3, strict, "#0f766e"),
            (3, partial, "#f59e0b"),
        ):
            x = center + offset
            bar_h = value / y_max * plot_h
            y = top + plot_h - bar_h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" rx="4" fill="{color}"/>'
            )
            parts.append(
                _text(x + bar_w / 2, y - 7, f"{value:.3f}", 10, anchor="middle", color="#334155")
            )
        parts.append(_text(center, top + plot_h + 28, noise, 11, anchor="middle", weight=600))
        parts.append(_text(center, top + plot_h + 46, reward, 11, anchor="middle", color="#64748b"))
    parts.append(
        _text(24, top + plot_h / 2, "score", 12, anchor="middle", color="#475569", rotate=-90)
    )
    parts.append(
        _text(
            65,
            height - 26,
            "* Approximate context only: sampling temperature and completions per task differ.",
            11,
            color="#64748b",
        )
    )
    parts.append("</svg>\n")
    return "\n".join(parts)


def markdown_report(summaries: list[dict[str, Any]], baseline: dict[str, dict[str, Any]]) -> str:
    trained_overall = [float(row["primary_test_pass_at_1"]) for row in summaries]
    trained_by_difficulty = {
        difficulty: [float(row[f"test_{difficulty}_pass_at_1"]) for row in summaries]
        for difficulty in DIFFICULTIES
    }
    gains = {
        difficulty: (
            min(trained_by_difficulty[difficulty]) - float(baseline[difficulty]["pass_at_1"]),
            max(trained_by_difficulty[difficulty]) - float(baseline[difficulty]["pass_at_1"]),
        )
        for difficulty in DIFFICULTIES
    }
    overall_gain = (
        min(trained_overall) - float(baseline["all"]["pass_at_1"]),
        max(trained_overall) - float(baseline["all"]["pass_at_1"]),
    )
    lines = [
        "# Evaluation slices — priority_bridge seed 0",
        "",
        "> Scope: seed 0 only. The curves are descriptive; they do not quantify variation across training seeds.",
        "> The pretraining row is an approximate context baseline, not an exact matched step-0 evaluation.",
        "",
        "## What the metrics mean",
        "",
        "- **Clean Pass@1** is the fraction of evaluation problems for which the single generated program passes every clean unit test. It is strict task-level success.",
        "- **Mean test pass rate** is the mean fraction of unit tests passed by the generated programs. It gives partial credit when a program passes some, but not all, tests.",
        "- Validation contains 90 fixed problems, so one problem moves Pass@1 by 1/90 = 1.11 percentage points.",
        "- Test contains 500 fixed problems, so one problem moves Pass@1 by 1/500 = 0.20 percentage points.",
        "- The primary reported metric is the mean clean test Pass@1 over checkpoints 337 and 374. The two checkpoints use the same 500 tasks and are not independent samples.",
        "",
        "## Before-training estimate",
        "",
        "An exact step-0 validation or test evaluation under the trained-run protocol was not saved. The closest available baseline is the frozen test difficulty map from the same base model revision. It contains 16 sampled completions for each of 500 test tasks, with temperature 1.0. Trained checkpoints were evaluated with one completion per task at temperature 0.7, so the comparison below is approximate rather than controlled.",
        "",
        "| Difficulty | Tasks | Completions | Base Pass@1 estimate | Base mean test pass rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for difficulty in ("all", *DIFFICULTIES):
        metrics = baseline[difficulty]
        lines.append(
            f"| {difficulty} | {metrics['n_problems']} | {metrics['n_completions']} | "
            f"{metrics['pass_at_1']:.3f} | {metrics['mean_test_pass_rate']:.3f} |"
        )
    lines += [
        "",
        "The strata are frozen performance buckets defined by the base model's mean fractional reward: hard ≤ 1/3, medium > 1/3 and < 2/3, easy ≥ 2/3. They describe empirical base-model difficulty, not semantic task types.",
        "",
        "## Available checkpoints",
        "",
        "- Validation: steps 37, 75, 112, 150, 187, 224, 262, 299, 337, and 374.",
        "- Test: steps 337 and 374 only. A full test learning curve cannot be reconstructed from the saved artifacts.",
        "- Step 0 is shown only for test and only as the approximate multi-sample baseline described above; there is no matching validation step 0.",
        "- Validation predictions are marked with difficulty `unknown`; therefore easy/medium/hard slicing is available for test only. Assigning validation strata now would require a new base-model grading run.",
        "",
        "## Condition summary",
        "",
        "| Condition | Val 37 | Val 374 | Val range | Test 337 | Test 374 | Test change | Primary test mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['condition']} | {row['validation_step_37']:.3f} | "
            f"{row['validation_step_374']:.3f} | {row['validation_min']:.3f}–"
            f"{row['validation_max']:.3f} | {row['test_step_337']:.3f} | "
            f"{row['test_step_374']:.3f} | {row['test_change']:+.3f} | "
            f"{row['primary_test_pass_at_1']:.3f} |"
        )
    lines += [
        "",
        "## Test Pass@1 by difficulty",
        "",
        "Each trained value is the mean over test checkpoints 337 and 374.",
        "",
        "| Condition | Hard (n=220) | Medium (n=130) | Easy (n=150) |",
        "|---|---:|---:|---:|",
        f"| pretraining estimate* | {baseline['hard']['pass_at_1']:.3f} | {baseline['medium']['pass_at_1']:.3f} | {baseline['easy']['pass_at_1']:.3f} |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['condition']} | {row['test_hard_pass_at_1']:.3f} | "
            f"{row['test_medium_pass_at_1']:.3f} | {row['test_easy_pass_at_1']:.3f} |"
        )
    lines += [
        "",
        "## Figures",
        "",
        "![Validation and test Pass@1 by condition](eval_pass_at_1_by_condition.svg)",
        "",
        "![Test Pass@1 by difficulty](test_pass_at_1_by_difficulty.svg)",
        "",
        "![Strict Pass@1 and partial test success](pass_at_1_vs_partial_credit.svg)",
        "",
        "## Reading the curves",
        "",
        "- Validation curves fluctuate in increments of roughly 0.011 because validation has only 90 tasks. Most visible changes correspond to one to four problems.",
        "- Test changes from step 337 to 374 range from -0.014 to +0.008. This is between seven fewer and four more fully solved tasks out of 500.",
        "- Validation Pass@1 is higher than test Pass@1 at both common checkpoints for every condition. The gap ranges from about 0.037 to 0.103, so validation and test should not be treated as interchangeable scales.",
        f"- Overall base Pass@1 is approximately {baseline['all']['pass_at_1']:.3f}; trained condition means range from {min(trained_overall):.3f} to {max(trained_overall):.3f}. The apparent gain is {overall_gain[0]:+.3f} to {overall_gain[1]:+.3f}, subject to the protocol mismatch.",
        f"- Hard: base {baseline['hard']['pass_at_1']:.3f}; trained {min(trained_by_difficulty['hard']):.3f}–{max(trained_by_difficulty['hard']):.3f}; apparent gain from {gains['hard'][0]:+.3f} to {gains['hard'][1]:+.3f}.",
        f"- Medium: base {baseline['medium']['pass_at_1']:.3f}; trained {min(trained_by_difficulty['medium']):.3f}–{max(trained_by_difficulty['medium']):.3f}; apparent gain from {gains['medium'][0]:+.3f} to {gains['medium'][1]:+.3f}.",
        f"- Easy: base {baseline['easy']['pass_at_1']:.3f}; trained {min(trained_by_difficulty['easy']):.3f}–{max(trained_by_difficulty['easy']):.3f}; apparent gain from {gains['easy'][0]:+.3f} to {gains['easy'][1]:+.3f}.",
        "- The largest apparent absolute gain is on medium tasks; hard remains the main bottleneck despite improvement, while easy tasks are already near saturation.",
        "- Mean test pass rate exceeds strict Pass@1 by about 0.06–0.07 in every condition. This is the mass of partially correct programs that pass some tests but fail at least one.",
        "- No condition shows a large, monotone evaluation trend. The saved seed-0 differences are small relative to checkpoint-to-checkpoint validation fluctuation.",
        "",
        "## Machine-readable tables",
        "",
        "- `evaluation_metrics.csv`: every saved split/checkpoint/difficulty point.",
        "- `evaluation_metric_summary.csv`: one compact row per condition.",
        "- `pretraining_baseline.csv`: the approximate base-model test baseline by difficulty.",
        "",
    ]
    return "\n".join(lines)


def _svg_start(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
    ]


def _text(
    x: float,
    y: float,
    value: str,
    size: int,
    *,
    anchor: str = "start",
    weight: int = 400,
    color: str = "#0f172a",
    rotate: int | None = None,
) -> str:
    transform = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate is not None else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}"{transform}>'
        f"{html.escape(value)}</text>"
    )


def _polyline(points: list[tuple[float, float]], color: str, width: float) -> str:
    values = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{values}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>'


def _legend(x: float, y: float, items: tuple[tuple[str, str], ...]) -> str:
    parts = []
    cursor = x
    for label, color in items:
        parts.append(
            f'<line x1="{cursor}" y1="{y}" x2="{cursor + 28}" y2="{y}" stroke="{color}" stroke-width="4"/>'
        )
        parts.append(_text(cursor + 36, y + 5, label, 13, color="#334155"))
        cursor += 180
    return "".join(parts)


def _heat_color(value: float) -> str:
    value = min(1.0, max(0.0, value))
    low = (239, 246, 255)
    high = (30, 64, 175)
    rgb = tuple(round(a + (b - a) * value) for a, b in zip(low, high, strict=True))
    return "#" + "".join(f"{component:02x}" for component in rgb)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
