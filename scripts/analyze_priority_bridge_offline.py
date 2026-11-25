#!/usr/bin/env python3
"""Analyze the frozen v2 archive without loading a model or running verifier code."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from itertools import combinations, groupby
from pathlib import Path

from analysis_svg import _legend, _polyline, _svg_start, _text

SEEDS = (0, 1, 2)
NOISES = ("clean", "fp_only", "fn_only")
REWARDS = ("binary", "fractional")
CONDITIONS = tuple(f"{noise}__{reward}" for noise in NOISES for reward in REWARDS)
DIFFICULTIES = ("hard", "medium", "easy")
PHASES = ("early", "middle", "late")
VALIDATION_STEPS = (0, 112, 224, 337, 449, 561, 673, 785, 898, 1010, 1122)
TEST_STEPS = (1010, 1122)
CONTRAST = {
    "fn_only__fractional": 1,
    "fp_only__fractional": -1,
    "fn_only__binary": -1,
    "fp_only__binary": 1,
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        yield from map(json.loads, stream)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def outcomes(value):
    require(len(value) == 3 and all(bit in (0, 1) for bit in value), "invalid test outcomes")
    return sum(value)


def evaluation_rows(path, samples):
    tasks = defaultdict(dict)
    for row in read_rows(path):
        task = row["task_id"]
        slot = row["sample_index"] if samples > 1 else 0
        require(slot not in tasks[task], f"duplicate task/sample in {path}")
        passed = outcomes(row["clean_outcomes"])
        require(
            row["clean_binary_reward"] == int(passed == 3)
            and math.isclose(row["clean_fractional_reward"], passed / 3, abs_tol=1e-10),
            f"reward mismatch in {path}",
        )
        signature = (
            (row["generation_seed"], row["generation_chunk_seed_slot"])
            if samples > 1
            else row["evaluation_seed"]
        )
        tasks[task][slot] = (passed, signature, row["difficulty"])
    for slots in tasks.values():
        require(set(slots) == set(range(samples)), f"incomplete sample slots in {path}")
    return tasks


def task_scores(rows, metric):
    def score(passed):
        return int(passed == 3) if metric == "pass_at_1" else passed / 3

    return {
        task: statistics.fmean(score(row[0]) for row in slots.values())
        for task, slots in rows.items()
    }


def statistics_for_seeds(values, *, resamples=0):
    per_seed = [statistics.fmean(row) for row in values]
    mean = statistics.fmean(per_seed)
    half = 4.302652729911275 * statistics.stdev(per_seed) / math.sqrt(len(per_seed))
    result = {"mean": mean, "by_seed": per_seed, "seed_t_ci95": [mean - half, mean + half]}
    if resamples:
        # Draw one shared task sample across resampled seed blocks. Inputs are
        # already paired differences at the seed and task level.
        rng = random.Random(20261125)
        task_count = len(values[0])
        boot = []
        for _ in range(resamples):
            seed_rows = rng.choices(values, k=len(values))
            across_seeds = [sum(samples) / len(values) for samples in zip(*seed_rows, strict=True)]
            boot.append(statistics.fmean(rng.choices(across_seeds, k=task_count)))
        result["two_way_ci95"] = [
            statistics.quantiles(boot, n=40, method="inclusive")[i] for i in (0, 38)
        ]
    return result


def evaluation(root):
    base = evaluation_rows(root / "common-base/predictions.jsonl", 8)
    difficulty = {task: slots[0][2] for task, slots in base.items()}
    require(
        Counter(difficulty.values()) == {"hard": 220, "medium": 130, "easy": 150},
        "unexpected frozen test strata",
    )
    test, validation = {}, []
    validation_reference = None
    for seed in SEEDS:
        seed_dir = root / f"main/seed-{seed}"
        status = read_json(seed_dir / "matrix_status.json")
        require(
            status["status"] == "complete" and set(status["conditions"]) == set(CONDITIONS),
            f"incomplete matrix for seed {seed}",
        )
        for condition in CONDITIONS:
            eval_dir = seed_dir / condition / "evaluations"
            for index, step in enumerate(TEST_STEPS):
                path = eval_dir / f"test/step-{step}/predictions.jsonl"
                rows = evaluation_rows(path, 4)
                require(set(rows) == set(base), f"test task mismatch in {path}")
                for task, slots in rows.items():
                    for slot, (_, signature, stratum) in slots.items():
                        require(
                            (signature, stratum) == base[task][slot + 4 * index][1:],
                            f"test sampling or stratum mismatch in {path}",
                        )
                test[(seed, condition, step)] = rows
            for step in VALIDATION_STEPS:
                path = eval_dir / f"validation/step-{step}/predictions.jsonl"
                rows = evaluation_rows(path, 1)
                require(len(rows) == 90, f"expected 90 validation tasks in {path}")
                signature = {task: slots[0][1] for task, slots in rows.items()}
                if validation_reference is None:
                    validation_reference = signature
                require(
                    signature == validation_reference, f"validation tasks/seeds differ in {path}"
                )
                validation.append(
                    {
                        "seed": seed,
                        "condition": condition,
                        "step": step,
                        **{
                            metric: statistics.fmean(task_scores(rows, metric).values())
                            for metric in ("pass_at_1", "mean_test_pass_rate")
                        },
                    }
                )

    results = {}
    for metric in ("pass_at_1", "mean_test_pass_rate"):
        results[metric] = {}
        base_scores = task_scores(base, metric)
        scored = {key: task_scores(rows, metric) for key, rows in test.items()}
        for stratum in ("all", *DIFFICULTIES):
            tasks = sorted(
                task for task, label in difficulty.items() if stratum == "all" or label == stratum
            )
            block = {
                "n_tasks": len(tasks),
                "baseline": statistics.fmean(base_scores[task] for task in tasks),
            }
            for epoch_point, steps in (
                ("primary", TEST_STEPS),
                *((str(s), (s,)) for s in TEST_STEPS),
            ):
                slots = [
                    slot
                    for step in steps
                    for slot in range(4 * TEST_STEPS.index(step), 4 * (TEST_STEPS.index(step) + 1))
                ]
                baseline_slice = task_scores(
                    {task: {slot: base[task][slot] for slot in slots} for task in tasks}, metric
                )
                trained = {
                    condition: [
                        [
                            statistics.fmean(scored[seed, condition, step][task] for step in steps)
                            for task in tasks
                        ]
                        for seed in SEEDS
                    ]
                    for condition in CONDITIONS
                }
                by_condition = {}
                for condition, values in trained.items():
                    deltas = [
                        [
                            value - baseline_slice[task]
                            for value, task in zip(row, tasks, strict=True)
                        ]
                        for row in values
                    ]
                    by_condition[condition] = {
                        "trained": statistics.fmean(statistics.fmean(row) for row in values),
                        "delta": statistics_for_seeds(deltas),
                    }
                contrasts = {
                    "main": CONTRAST,
                    "fn_minus_fp_binary": {"fn_only__binary": 1, "fp_only__binary": -1},
                    "fn_minus_fp_fractional": {"fn_only__fractional": 1, "fp_only__fractional": -1},
                }
                block[epoch_point] = {
                    "matched_baseline": statistics.fmean(baseline_slice.values()),
                    "conditions": by_condition,
                    "contrasts": {},
                }
                for name, terms in contrasts.items():
                    differences = [
                        [
                            sum(
                                weight * trained[condition][seed_index][task_index]
                                for condition, weight in terms.items()
                            )
                            for task_index in range(len(tasks))
                        ]
                        for seed_index in range(len(SEEDS))
                    ]
                    block[epoch_point]["contrasts"][name] = statistics_for_seeds(
                        differences,
                        resamples=10_000
                        if name == "main" and epoch_point == "primary" and metric == "pass_at_1"
                        else 0,
                    )
            results[metric][stratum] = block
    return {"validation": validation, "test": results}


def reward_vectors(counts, aggregation):
    return [float(count == 3) if aggregation == "binary" else count / 3 for count in counts]


def advantages(rewards):
    mean = statistics.fmean(rewards)
    scale = statistics.stdev(rewards) + 1e-4
    return [(value - mean) / scale for value in rewards]


def verify_group(group, clean_counts, noisy_counts, aggregation):
    clean = reward_vectors(clean_counts, aggregation)
    noisy = reward_vectors(noisy_counts, aggregation)
    clean_adv, noisy_adv = advantages(clean), advantages(noisy)
    for row, clean_r, noisy_r, clean_a, noisy_a in zip(
        group, clean, noisy, clean_adv, noisy_adv, strict=True
    ):
        fields = {
            f"clean_{aggregation}_reward": clean_r,
            f"noisy_{aggregation}_reward": noisy_r,
            f"clean_{aggregation}_advantage": clean_a,
            f"noisy_{aggregation}_advantage": noisy_a,
        }
        if row["condition"].endswith(f"__{aggregation}"):
            fields.update(train_reward=noisy_r, train_advantage=noisy_a)
        require(
            all(
                math.isclose(row[field], value, rel_tol=0, abs_tol=1e-9)
                for field, value in fields.items()
            ),
            f"reward/advantage mismatch in {row['group_id']}",
        )
    return clean, noisy, clean_adv, noisy_adv


def comparison_counts(clean_counts, noisy_counts, aggregation):
    """Count preserved, tied and reversed pairs with and without common support."""
    histogram = Counter(zip(clean_counts, noisy_counts, strict=True))
    result = Counter()
    for ((i, a), left), ((j, b), right) in combinations(histogram.items(), 2):
        clean_gap = (int(i == 3) - int(j == 3)) if aggregation == "binary" else i - j
        noisy_gap = (int(a == 3) - int(b == 3)) if aggregation == "binary" else a - b
        comparison = clean_gap * noisy_gap
        if clean_gap:
            state = "preserved" if comparison > 0 else "tied" if comparison == 0 else "reversed"
            result["distinct_clean_pairs"] += left * right
            result[state] += left * right
            if (i == 3) != (j == 3):
                result["common_pairs"] += left * right
                result[f"common_{state}"] += left * right
        elif noisy_gap:
            result["split_clean_ties"] += left * right
    return result


def group_metrics(counts, noisy_counts, clean, noisy, clean_adv, noisy_adv, aggregation):
    clean_var = min(clean) != max(clean)
    noisy_var = min(noisy) != max(noisy)
    support = min(counts) < 3 == max(counts)
    paired_signs = [
        (a, b)
        for a, b in zip(clean_adv, noisy_adv, strict=True)
        if abs(a) > 1e-12 and abs(b) > 1e-12
    ]
    metrics = Counter(
        {
            "groups": 1,
            "clean_variable": int(clean_var),
            "both_variable": int(clean_var and noisy_var),
            "erased": int(clean_var and not noisy_var),
            "created": int(not clean_var and noisy_var),
            "both_constant": int(not clean_var and not noisy_var),
            "common_groups": int(support),
            "common_erased": int(support and not noisy_var),
            "sign_groups_defined": int(bool(paired_signs)),
            "sign_answers_defined": len(paired_signs),
            "sign_answers_agree": sum(a * b > 0 for a, b in paired_signs),
            "clean_advantage_nonzero": sum(abs(a) > 1e-12 for a in clean_adv),
            "noisy_advantage_nonzero": sum(abs(a) > 1e-12 for a in noisy_adv),
            "clean_reward_sum": sum(clean),
            "noisy_reward_sum": sum(noisy),
        }
    )
    metrics.update(comparison_counts(counts, noisy_counts, aggregation))
    return metrics


def rollout_analysis(root):
    signal = defaultdict(Counter)
    outcomes_by_slice = defaultdict(Counter)
    fn_composition = defaultdict(Counter)
    audit_groups = {}
    for seed in SEEDS:
        for condition in CONDITIONS:
            noise, train_aggregation = condition.split("__")
            path = root / f"main/seed-{seed}/{condition}/rollouts.jsonl"
            seen = set()
            for group_id, rows in groupby(read_rows(path), key=lambda row: row["group_id"]):
                group = list(rows)
                first = group[0]
                stratum = first["difficulty"]
                step = first["policy_step"]
                require(
                    group_id not in seen
                    and len(group) == 16
                    and stratum in DIFFICULTIES
                    and 0 <= step < 1122,
                    f"invalid group {group_id} in {path}",
                )
                seen.add(group_id)
                require(
                    all(
                        (r["condition"], r["seed"], r["difficulty"], r["policy_step"], r["task_id"])
                        == (condition, seed, stratum, step, first["task_id"])
                        for r in group
                    )
                    and {r["completion_index"] for r in group} == set(range(16)),
                    f"inconsistent group {group_id}",
                )
                if step in (112, 561, 1010):
                    audit_groups[(seed, condition, step)] = (group_id, first["task_id"], stratum)
                phase = PHASES[step // 374]
                keys = (
                    ("overall", noise, "all", "all", "all", "all"),
                    ("seed", noise, "all", "all", str(seed), "all"),
                    ("difficulty_phase", noise, stratum, phase, "all", "all"),
                    ("difficulty", noise, stratum, "all", "all", "all"),
                    ("phase", noise, "all", phase, "all", "all"),
                    ("trajectory", noise, stratum, phase, str(seed), train_aggregation),
                )
                counts = [outcomes(row["clean_outcomes"]) for row in group]
                noisy_counts = [outcomes(row["noisy_outcomes"]) for row in group]
                transitions = Counter(
                    f"{a}->{b}" for a, b in zip(counts, noisy_counts, strict=True)
                )
                for row in group:
                    for a, b, mask in zip(
                        row["clean_outcomes"],
                        row["noisy_outcomes"],
                        row["corruption_mask"],
                        strict=True,
                    ):
                        require(mask == (a != b), f"mask mismatch in {group_id}")
                        require(
                            (noise == "fn_only" and b <= a)
                            or (noise == "fp_only" and b >= a)
                            or (noise == "clean" and b == a),
                            f"wrong noise direction in {group_id}",
                        )
                        transitions["eligible_fp"] += int(a == 0)
                        transitions["eligible_fn"] += int(a == 1)
                        transitions["flips_fp"] += int(a == 0 and b == 1)
                        transitions["flips_fn"] += int(a == 1 and b == 0)
                for key in keys:
                    outcomes_by_slice[key].update(transitions)
                if noise == "fn_only" and 0 < counts.count(3) < 16:
                    correct = counts.count(3)
                    for key in ("all", stratum):
                        fn_composition[key]["mixed_groups"] += 1
                        fn_composition[key]["correct_completions"] += correct
                        fn_composition[key]["erased"] += max(noisy_counts) < 3
                        fn_composition[key]["expected_erased"] += (1 - 0.7**3) ** correct
                for agg in REWARDS:
                    vectors = verify_group(group, counts, noisy_counts, agg)
                    metrics = group_metrics(counts, noisy_counts, *vectors, agg)
                    if agg == train_aggregation:
                        require(
                            metrics["both_constant"] + metrics["erased"]
                            == int(first["zero_variance_group"]),
                            f"wrong zero-variance flag in {group_id}",
                        )
                    for key in keys:
                        signal[(*key, agg)].update(metrics)
            require(len(seen) == 1122, f"expected 1122 groups in {path}")
    columns = ("slice", "noise", "difficulty", "phase", "seed", "training_aggregation")
    return (
        {
            "outcome_counts": [
                dict(zip(columns, key, strict=True), **counts)
                for key, counts in sorted(outcomes_by_slice.items())
            ],
            "signal_counts": [
                dict(zip((*columns, "aggregation"), key, strict=True), **counts)
                for key, counts in sorted(signal.items())
            ],
            "fn_binary_group_composition": fn_composition,
        },
        audit_groups,
    )


def gradient_analysis(root, audit_groups):
    audit_root = root / "main/gradient_audit_v1_1"
    records = []
    for (seed, condition, step), (group_id, task, stratum) in sorted(audit_groups.items()):
        path = audit_root / f"targets/seed-{seed}/{condition}/step-{step}.json"
        row = read_json(path)
        require(
            (
                row["group_id"],
                row["task_id"],
                row["difficulty"],
                row["step"],
                row["seed"],
                row["condition"],
                row["completion_count"],
            )
            == (group_id, task, stratum, step, seed, condition, 16),
            f"audit target does not match raw group: {path}",
        )
        require(
            row["phase"] == PHASES[step // 374] and set(row["aggregations"]) == set(REWARDS),
            f"invalid audit phase or aggregation in {path}",
        )
        records.append(
            {
                field: row[field]
                for field in (
                    "seed",
                    "condition",
                    "phase",
                    "step",
                    "task_id",
                    "group_id",
                    "noise",
                    "difficulty",
                    "train_aggregation",
                    "aggregations",
                    "token_reconstruction",
                )
            }
        )
    require(len(records) == 54, "expected all 54 LoRA audit groups")
    summaries = []
    for noise in NOISES:
        for stratum, phase in (
            ("all", "all"),
            *((d, "all") for d in DIFFICULTIES),
            *(("all", p) for p in PHASES),
            *((d, p) for d in DIFFICULTIES for p in PHASES),
        ):
            selected = [
                r
                for r in records
                if r["noise"] == noise
                and (stratum == "all" or r["difficulty"] == stratum)
                and (phase == "all" or r["phase"] == phase)
            ]
            row = {"noise": noise, "difficulty": stratum, "phase": phase, "groups": len(selected)}
            for metric in ("advantage", "gradient"):
                values = []
                actual = {}
                for agg in REWARDS:
                    stats = [
                        r["aggregations"][agg][metric]
                        for r in selected
                        if r["train_aggregation"] == agg
                    ]
                    cosine = [r["cosine"] for r in stats if r["cosine"] is not None]
                    actual[agg] = {
                        "groups": len(stats),
                        "defined": len(cosine),
                        "mean_cosine": statistics.fmean(cosine) if cosine else None,
                        "clean_zero": sum(s["clean_zero"] for s in stats),
                        "noisy_zero": sum(s["noisy_zero"] for s in stats),
                        "erased": sum(not s["clean_zero"] and s["noisy_zero"] for s in stats),
                        "negative": sum(v < 0 for v in cosine),
                    }
                for r in selected:
                    b, f = (r["aggregations"][agg][metric]["cosine"] for agg in REWARDS)
                    if b is not None and f is not None:
                        values.append(f - b)
                row[metric] = {
                    "actual": actual,
                    "defined_pairs": len(values),
                    "mean_paired_delta": statistics.fmean(values) if values else None,
                    "fractional_higher": sum(value > 1e-12 for value in values),
                }
            summaries.append(row)
    return {"groups": records, "summaries": summaries}


COLORS = dict(zip(REWARDS, ("#1d4ed8", "#a34111"), strict=True))
NOISE_LABELS = dict(zip(NOISES, ("Clean", "FP-only 30%", "FN-only 30%"), strict=True))
DIFFICULTY_LABELS = ("Hard", "Medium", "Easy", "Early", "Middle", "Late")


def svg_canvas(width, height, title, subtitle):
    svg = _svg_start(width, height)
    svg[1] = '<rect width="100%" height="100%" fill="#ffffff"/>'
    svg.extend((_text(26, 33, title, 23, weight=700), _text(26, 58, subtitle, 13)))
    return svg


def plot_axes(svg, box, *, title, ticks, step_ticks, x_scale, y_scale, format_y):
    left, top, width, height = box
    svg.append(_text(left, top - 13, title, 14, weight=700))
    for tick in ticks:
        y = y_scale(tick)
        svg.append(
            f'<line x1="{left}" x2="{left + width}" y1="{y}" y2="{y}" '
            'stroke="#cbd5e1" stroke-width="1"/>'
        )
        svg.append(_text(left - 7, y + 4, format_y(tick), 11, anchor="end"))
    for step in step_ticks:
        svg.append(_text(x_scale(step), top + height + 17, str(step), 11, anchor="middle"))


def validation_svg(result):
    svg = svg_canvas(
        1060,
        585,
        "Validation learning curves",
        "90 tasks · 1 decode per task · bold = mean of three seeds · thin = individual seeds",
    )
    curves = defaultdict(dict)
    for row in result["evaluation"]["validation"]:
        curves[row["condition"], row["seed"]][row["step"]] = row
    for r, (metric, label) in enumerate(
        (
            ("pass_at_1", "Pass@1"),
            ("mean_test_pass_rate", "Unit-test pass rate"),
        )
    ):
        values = [point[metric] * 100 for curve in curves.values() for point in curve.values()]
        lo, hi = 2 * math.floor(min(values) / 2), 2 * math.ceil(max(values) / 2)
        ticks = range(lo, hi + 1, 2)
        for c, noise in enumerate(NOISES):
            left, top, width, height = 72 + c * 330, 115 + r * 218, 289, 157

            def x(step, left=left, width=width):
                return left + step / 1122 * width

            def y(value, top=top, height=height, lo=lo, hi=hi):
                return top + height * (hi - value) / (hi - lo)

            plot_axes(
                svg,
                (left, top, width, height),
                title=f"{NOISE_LABELS[noise]} · {label}",
                ticks=ticks,
                step_ticks=(0, 374, 748, 1122),
                x_scale=x,
                y_scale=y,
                format_y=lambda value: f"{value}%",
            )
            for agg in REWARDS:
                series = [curves[f"{noise}__{agg}", seed] for seed in SEEDS]
                color = COLORS[agg]
                for seed_values in series:
                    points = [(x(s), y(100 * seed_values[s][metric])) for s in VALIDATION_STEPS]
                    svg.append(_polyline(points, color, 1).replace("/>", ' opacity="0.22"/>'))
                mean = [
                    (x(s), y(100 * statistics.fmean(v[s][metric] for v in series)))
                    for s in VALIDATION_STEPS
                ]
                svg.append(_polyline(mean, color, 2.8))
    svg.extend(
        (
            _legend(74, 545, tuple((agg.title(), color) for agg, color in COLORS.items())),
            _text(1015, 549, "Optimizer step", 12, anchor="end"),
        )
    )
    return "\n".join([*svg, "</svg>"])


def test_checkpoints_svg(result):
    svg = svg_canvas(
        1060,
        353,
        "Clean test Pass@1 at the two saved checkpoints",
        "Gain over the baseline with matched decoding slots · dots = per-seed gains · bars = means",
    )
    test = result["evaluation"]["test"]["pass_at_1"]["all"]
    values = [
        100 * value
        for step in TEST_STEPS
        for r in test[str(step)]["conditions"].values()
        for value in r["delta"]["by_seed"]
    ]
    lo, hi = min(0, math.floor(min(values))), math.ceil(max(values))
    for c, noise in enumerate(NOISES):
        left, top, width, height = 72 + c * 330, 114, 289, 144

        def x(step, left=left):
            return left + (70 if step == 1010 else 219)

        def y(value, top=top, height=height):
            return top + height * (hi - value) / (hi - lo)

        plot_axes(
            svg,
            (left, top, width, height),
            title=NOISE_LABELS[noise],
            ticks=range(lo, hi + 1),
            step_ticks=TEST_STEPS,
            x_scale=x,
            y_scale=y,
            format_y=lambda value: f"{value:+d} pp" if value else "0 pp",
        )
        for index, agg in enumerate(REWARDS):
            for step in TEST_STEPS:
                px = x(step) + (-22 if index == 0 else 22)
                score = test[str(step)]["conditions"][f"{noise}__{agg}"]["delta"]
                py = y(100 * score["mean"])
                svg.append(
                    f'<line x1="{px - 16}" x2="{px + 16}" y1="{py}" y2="{py}" '
                    f'stroke="{COLORS[agg]}" stroke-width="3"/>'
                )
                for offset, value in zip((-8, 0, 8), score["by_seed"], strict=True):
                    svg.append(
                        f'<circle cx="{px + offset}" cy="{y(100 * value)}" r="3.6" '
                        f'fill="{COLORS[agg]}" stroke="white" stroke-width="1"/>'
                    )
    svg.extend(
        (
            _legend(74, 317, tuple((agg.title(), color) for agg, color in COLORS.items())),
            _text(1015, 321, "Optimizer step", 12, anchor="end"),
        )
    )
    return "\n".join([*svg, "</svg>"])


def fn_signal_svg(result):
    data = result["rollouts"]["signal_counts"]
    svg = svg_canvas(
        960,
        332,
        "FN noise frequently erases the binary reward signal",
        "Same completions and masks · denominator: groups with both a 3/3 and a less-than-3/3 answer",
    )
    for i, (mode, slices) in enumerate((("difficulty", DIFFICULTIES), ("phase", PHASES))):
        left, top, width, height = 66 + i * 450, 116, 394, 127
        marks = {label: left + (index + 0.5) * width / 3 for index, label in enumerate(slices)}

        def y(value, top=top, height=height):
            return top + (40 - value) / 40 * height

        plot_axes(
            svg,
            (left, top, width, height),
            title="Training task difficulty" if i == 0 else "Training phase",
            ticks=range(0, 41, 10),
            step_ticks=(),
            x_scale=marks.__getitem__,
            y_scale=y,
            format_y=lambda value: f"{value}%",
        )
        for j, label in enumerate(slices):
            center = marks[label]
            svg.append(
                _text(
                    center,
                    269,
                    DIFFICULTY_LABELS[3 * i + j],
                    12,
                    anchor="middle",
                    weight=600,
                    color="#0f172a",
                )
            )
            for k, agg in enumerate(REWARDS):
                row = next(
                    r
                    for r in data
                    if r["slice"] == mode
                    and r["noise"] == "fn_only"
                    and r[mode] == label
                    and r["aggregation"] == agg
                )
                probability = 100 * row["common_erased"] / row["common_groups"]
                bx = center + (-37 if k == 0 else 5)
                svg.append(
                    f'<rect x="{bx}" y="{y(probability)}" width="32" '
                    f'height="{y(0) - y(probability)}" fill="{COLORS[agg]}"/>'
                )
                shown = f"{probability:.2f}%" if 0 < probability < 0.1 else f"{probability:.1f}%"
                svg.append(_text(bx + 16, y(probability) - 5, shown, 11, anchor="middle"))
    svg.append(_legend(70, 310, tuple((agg.title(), color) for agg, color in COLORS.items())))
    return "\n".join([*svg, "</svg>"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    eval_result = evaluation(args.archive)
    rollout_result, audit_groups = rollout_analysis(args.archive)
    result = {
        "experiment": "priority_bridge_v2",
        "seeds": SEEDS,
        "phase_policy_step_ranges": dict(
            zip(PHASES, ((0, 373), (374, 747), (748, 1121)), strict=True)
        ),
        "evaluation": eval_result,
        "rollouts": rollout_result,
        "gradient_audit": gradient_analysis(args.archive, audit_groups),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "analysis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    for name, figure in (
        ("validation_curves.svg", validation_svg),
        ("test_checkpoints.svg", test_checkpoints_svg),
        ("fn_signal_erasure.svg", fn_signal_svg),
    ):
        (args.out_dir / name).write_text(figure(result) + "\n", encoding="utf-8")
    print(
        f"Verified {len(eval_result['validation'])} validation points, 36 test targets and "
        f"{len(audit_groups)} gradient groups. Wrote {args.out_dir / 'analysis.json'}"
    )


if __name__ == "__main__":
    main()
