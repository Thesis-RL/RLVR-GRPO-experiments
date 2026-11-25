#!/usr/bin/env python3
"""Extract matched-J FN−FP contrasts from published v2 analysis artifacts.

Does not read raw rollouts. Inputs are the frozen offline ``analysis.json`` and
the Stage 6 multiseed reference JSON under the analysis package. Outputs a
machine-readable summary and a short markdown note for the FP vs FN question in
Proposition 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    for parent in (resolved, *resolved.parents):
        if (parent / ".git").exists() or (parent / "AGENTS.md").exists():
            try:
                return str(resolved.relative_to(parent))
            except ValueError:
                break
    return str(path)

SIGNAL_METRICS = (
    "mean_pairwise_fidelity",
    "mean_advantage_sign_agreement",
    "mean_advantage_cosine",
    "mean_top1_preserved",
    "mean_spearman",
    "zero_variance_group_fraction",
    "nonzero_train_advantage",
    "mean_changed_label_rate",
)

ENDPOINT_CONTRASTS = (
    "binary_fn_minus_fp",
    "fractional_fn_minus_fp",
    "fn_vs_fp_aggregation_interaction",
)

PROP1_OFFLINE_MARGIN = 0.05


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float, *, signed: bool = False, digits: int = 2) -> str:
    scaled = 100.0 * value
    prefix = "+" if signed and scaled >= 0 else ""
    return f"{prefix}{scaled:.{digits}f}"


def endpoint_block(stage6: dict) -> dict:
    out: dict[str, dict] = {}
    for name in ENDPOINT_CONTRASTS:
        metrics = {}
        for metric_name, summary in stage6["planned_contrasts"][name]["metrics"].items():
            metrics[metric_name] = {
                "estimate": summary["estimate"],
                "two_way_bootstrap_ci95": [
                    summary["two_way_bootstrap_ci95_low"],
                    summary["two_way_bootstrap_ci95_high"],
                ],
                "seed_t_ci95": [summary["seed_t_ci95_low"], summary["seed_t_ci95_high"]],
                "by_seed": summary.get("by_seed"),
            }
        out[name] = {
            "weights": stage6["planned_contrasts"][name]["weights"],
            "metrics": metrics,
        }
    gains = {}
    for condition in (
        "fp_only__binary",
        "fn_only__binary",
        "fp_only__fractional",
        "fn_only__fractional",
        "clean__binary",
        "clean__fractional",
    ):
        delta = stage6["conditions"][condition]["pass_at_1"]["delta_trained_minus_base"]
        gains[condition] = {
            "estimate": delta["estimate"],
            "seed_t_ci95": [delta["seed_t_ci95_low"], delta["seed_t_ci95_high"]],
            "by_seed": delta.get("by_seed"),
        }
    return {"planned_contrasts": out, "pass_at_1_gains_vs_base": gains}


def signal_block(stage6: dict) -> dict:
    conditions = stage6["training_signal"]["conditions"]
    per_seed_rows = stage6["training_signal"]["per_seed"]
    by_condition_seed: dict[str, dict[int, dict]] = {}
    for row in per_seed_rows:
        by_condition_seed.setdefault(row["condition"], {})[int(row["seed"])] = row

    levels: dict[str, dict] = {}
    for condition in (
        "fp_only__binary",
        "fn_only__binary",
        "fp_only__fractional",
        "fn_only__fractional",
    ):
        levels[condition] = {
            metric: {
                "mean": conditions[condition][metric]["mean"],
                "seed_t_ci95": [
                    conditions[condition][metric]["seed_t_ci95_low"],
                    conditions[condition][metric]["seed_t_ci95_high"],
                ],
            }
            for metric in SIGNAL_METRICS
        }

    deltas: dict[str, dict] = {}
    for aggregation in ("binary", "fractional"):
        fn = f"fn_only__{aggregation}"
        fp = f"fp_only__{aggregation}"
        metric_deltas = {}
        for metric in SIGNAL_METRICS:
            mean_delta = conditions[fn][metric]["mean"] - conditions[fp][metric]["mean"]
            by_seed = {}
            for seed in (0, 1, 2):
                by_seed[str(seed)] = (
                    by_condition_seed[fn][seed][metric] - by_condition_seed[fp][seed][metric]
                )
            metric_deltas[metric] = {"mean": mean_delta, "by_seed": by_seed}
        deltas[aggregation] = metric_deltas

    prop1 = {}
    for aggregation in ("binary", "fractional"):
        checks = {}
        for metric in (
            "mean_advantage_sign_agreement",
            "mean_pairwise_fidelity",
            "mean_top1_preserved",
        ):
            delta = deltas[aggregation][metric]["mean"]
            checks[metric] = {
                "fn_minus_fp": delta,
                "abs_delta": abs(delta),
                "within_0_05": abs(delta) <= PROP1_OFFLINE_MARGIN,
                "fn_worse_by_at_least_0_05": delta <= -PROP1_OFFLINE_MARGIN,
            }
        prop1[aggregation] = checks

    return {
        "condition_means": levels,
        "fn_minus_fp": deltas,
        "fn_minus_fp_threshold_checks": {
            "margin": PROP1_OFFLINE_MARGIN,
            "rule": (
                "Under fractional, |FN−FP| ≤ 0.05 on advantage agreement and "
                "pairwise fidelity counts as near-equal. Under binary, FN worse than FP "
                "by ≥ 0.05 on at least one of those two counts as a clear asymmetry."
            ),
            "by_aggregation": prop1,
        },
    }


def overall_signal_rows(analysis: dict) -> list[dict]:
    rows = []
    for row in analysis["rollouts"]["signal_counts"]:
        if row.get("slice") != "overall" or row["noise"] == "clean":
            continue
        preserved = row["preserved"] / row["distinct_clean_pairs"]
        erased = row["erased"] / row["clean_variable"] if row["clean_variable"] else None
        sign = (
            row["sign_answers_agree"] / row["sign_answers_defined"]
            if row["sign_answers_defined"]
            else None
        )
        rows.append(
            {
                "noise": row["noise"],
                "aggregation": row["aggregation"],
                "groups": row["groups"],
                "clean_variable_groups": row["clean_variable"],
                "erased_among_clean_variable": row["erased"],
                "erased_rate": erased,
                "created_among_clean_constant": row["created"],
                "pairwise_preserved_rate": preserved,
                "sign_agreement_rate": sign,
                "distinct_clean_pairs": row["distinct_clean_pairs"],
                "preserved_pairs": row["preserved"],
            }
        )
    return rows


def difficulty_erasure(analysis: dict) -> list[dict]:
    out = []
    for row in analysis["rollouts"]["signal_counts"]:
        if not (
            row.get("slice") == "difficulty"
            and row.get("seed") == "all"
            and row.get("phase") == "all"
            and row.get("training_aggregation") == "all"
            and row["noise"] in {"fn_only", "fp_only"}
        ):
            continue
        erased_rate = (
            row["erased"] / row["clean_variable"] if row["clean_variable"] else None
        )
        out.append(
            {
                "difficulty": row["difficulty"],
                "noise": row["noise"],
                "aggregation": row["aggregation"],
                "erased": row["erased"],
                "clean_variable": row["clean_variable"],
                "erased_rate": erased_rate,
                "pairwise_preserved_rate": row["preserved"] / row["distinct_clean_pairs"],
            }
        )
    return out


def gradient_block(analysis: dict) -> dict:
    summaries = []
    for row in analysis["gradient_audit"]["summaries"]:
        if (
            row["difficulty"] == "all"
            and row["phase"] == "all"
            and row["noise"] in {"fn_only", "fp_only"}
        ):
            summaries.append(row)

    groups = analysis["gradient_audit"]["groups"]
    actual = {}
    for noise in ("fn_only", "fp_only"):
        for train_aggregation in ("binary", "fractional"):
            advantages = []
            gradients = []
            for group in groups:
                if (
                    group["noise"] != noise
                    or group["train_aggregation"] != train_aggregation
                ):
                    continue
                block = group["aggregations"][train_aggregation]
                if block["advantage"]["cosine"] is not None:
                    advantages.append(block["advantage"]["cosine"])
                if block["gradient"]["cosine"] is not None:
                    gradients.append(block["gradient"]["cosine"])
            key = f"{noise}__{train_aggregation}"
            actual[key] = {
                "advantage_cosine": {
                    "n_defined": len(advantages),
                    "mean": statistics.fmean(advantages) if advantages else None,
                    "values": advantages,
                },
                "gradient_cosine": {
                    "n_defined": len(gradients),
                    "mean": statistics.fmean(gradients) if gradients else None,
                    "values": gradients,
                },
            }

    deltas = {}
    for train_aggregation in ("binary", "fractional"):
        for kind in ("advantage_cosine", "gradient_cosine"):
            fn = actual[f"fn_only__{train_aggregation}"][kind]
            fp = actual[f"fp_only__{train_aggregation}"][kind]
            if fn["mean"] is None or fp["mean"] is None:
                continue
            deltas[f"{train_aggregation}__{kind}"] = {
                "fn_mean": fn["mean"],
                "fp_mean": fp["mean"],
                "fn_minus_fp": fn["mean"] - fp["mean"],
                "n_fn": fn["n_defined"],
                "n_fp": fp["n_defined"],
            }
    return {"summaries_all_difficulty_phase": summaries, "actual_train_aggregation": actual, "fn_minus_fp": deltas}


def build_payload(*, analysis_path: Path, stage6_path: Path) -> dict:
    analysis = read_json(analysis_path)
    stage6 = read_json(stage6_path)
    endpoint = endpoint_block(stage6)
    training_signal = signal_block(stage6)
    rollout_groups = {
        "overall": overall_signal_rows(analysis),
        "by_difficulty": difficulty_erasure(analysis),
    }
    gradients = gradient_block(analysis)
    binary_pass = endpoint["planned_contrasts"]["binary_fn_minus_fp"]["metrics"][
        "pass_at_1"
    ]
    frac_pass = endpoint["planned_contrasts"]["fractional_fn_minus_fp"]["metrics"][
        "pass_at_1"
    ]
    return {
        "kind": "priority_bridge_v2_fp_vs_fn_contrasts",
        "schema_version": 1,
        "question": (
            "At matched J=0.7, how do FN-only and FP-only differ under binary vs "
            "fractional reward aggregation?"
        ),
        "sources": {
            "analysis_json": {
                "path": repo_relative(analysis_path),
                "sha256": file_sha256(analysis_path),
            },
            "stage6_multiseed_reference": {
                "path": repo_relative(stage6_path),
                "sha256": file_sha256(stage6_path),
            },
            "extractor": "scripts/extract_fp_vs_fn_contrasts.py",
            "upstream_scripts": {
                "stage6": "scripts/analyze_priority_bridge_multiseed.py",
                "offline": "scripts/analyze_priority_bridge_offline.py",
            },
            "notes": (
                "Endpoint Pass@1 contrasts and per-condition training-signal means come "
                "from Stage 6 (analyze_priority_bridge_multiseed.py). Within-group "
                "erasure / pairwise preservation and LoRA audit cosines come from the "
                "offline reanalysis (analyze_priority_bridge_offline.py → analysis.json). "
                "This script only re-aggregates those published artifacts; it does not "
                "re-read raw rollouts."
            ),
        },
        "endpoint": endpoint,
        "training_signal": training_signal,
        "rollout_groups": rollout_groups,
        "gradient_audit": gradients,
        "takeaways": {
            "endpoint_pass_at_1": (
                f"FN−FP is {pct(binary_pass['estimate'], signed=True)} pp under binary "
                f"and {pct(frac_pass['estimate'], signed=True)} pp under fractional; both "
                "seed-level intervals span zero."
            ),
            "training_signal_binary": (
                "Under binary reward, FN is much worse than FP on fidelity, advantage "
                "agreement/cosine, top-1 preservation, and contrast erasure."
            ),
            "training_signal_fractional": (
                "Under fractional reward, FN≈FP on fidelity and advantage metrics "
                "(|Δ|≲0.01), but top-1 preservation remains much lower under FN."
            ),
            "gradients": (
                "On audited groups with the actual training aggregation, clean→noisy "
                "cosine is substantially lower for FN than FP under binary, and nearly "
                "matched under fractional."
            ),
        },
    }


def render_markdown(payload: dict) -> str:
    endpoint = payload["endpoint"]
    signal = payload["training_signal"]
    lines = [
        "# FP vs FN at matched J · Priority Bridge v2",
        "",
        "## Sources",
        "",
        "| Input | Contents | SHA-256 |",
        "| --- | --- | --- |",
        (
            f"| `{payload['sources']['stage6_multiseed_reference']['path']}` | "
            "Pass@1 planned contrasts; training-signal condition means | "
            f"`{payload['sources']['stage6_multiseed_reference']['sha256'][:12]}…` |"
        ),
        (
            f"| `{payload['sources']['analysis_json']['path']}` | "
            "Group erasure / pairwise preservation; LoRA audit | "
            f"`{payload['sources']['analysis_json']['sha256'][:12]}…` |"
        ),
        "",
        (
            f"`{payload['sources']['extractor']}` ← "
            f"`{payload['sources']['upstream_scripts']['stage6']}`, "
            f"`{payload['sources']['upstream_scripts']['offline']}`"
        ),
        "",
        "```bash",
        "python3 -B scripts/extract_fp_vs_fn_contrasts.py \\",
        "  --analysis-json analyses/priority_bridge_v2_2026-11-25/analysis.json \\",
        "  --stage6-json analyses/priority_bridge_v2_2026-11-25/provenance/stage6_multiseed_reference.json \\",
        "  --out-dir analyses/priority_bridge_v2_2026-11-25",
        "```",
        "",
        "## 1. Endpoint · clean test Pass@1 (FN − FP)",
        "",
        "| Contrast | Estimate (pp) | Seed-t 95% CI (pp) | By seed (pp) |",
        "| --- | ---: | --- | --- |",
    ]
    for name in ENDPOINT_CONTRASTS:
        block = endpoint["planned_contrasts"][name]["metrics"]["pass_at_1"]
        by_seed = ", ".join(
            f"{seed}: {pct(value, signed=True)}"
            for seed, value in (block["by_seed"] or {}).items()
        )
        low, high = block["seed_t_ci95"]
        lines.append(
            f"| `{name}` | {pct(block['estimate'], signed=True)} | "
            f"[{pct(low, signed=True)}, {pct(high, signed=True)}] | {by_seed} |"
        )
    lines.extend(
        [
            "",
            "Gains over the matched untrained baseline:",
            "",
            "| Condition | Gain (pp) | Seed-t 95% CI (pp) |",
            "| --- | ---: | --- |",
        ]
    )
    for condition, block in endpoint["pass_at_1_gains_vs_base"].items():
        low, high = block["seed_t_ci95"]
        lines.append(
            f"| `{condition}` | {pct(block['estimate'], signed=True)} | "
            f"[{pct(low, signed=True)}, {pct(high, signed=True)}] |"
        )

    lines.extend(
        [
            "",
            "## 2. Training-signal means and FN − FP",
            "",
            "Condition means over training seeds 0, 1, 2.",
            "",
            "| Metric | FP/binary | FN/binary | FN−FP | FP/frac | FN/frac | FN−FP |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for metric in SIGNAL_METRICS:
        fp_b = signal["condition_means"]["fp_only__binary"][metric]["mean"]
        fn_b = signal["condition_means"]["fn_only__binary"][metric]["mean"]
        fp_f = signal["condition_means"]["fp_only__fractional"][metric]["mean"]
        fn_f = signal["condition_means"]["fn_only__fractional"][metric]["mean"]
        lines.append(
            f"| `{metric}` | {pct(fp_b)}% | {pct(fn_b)}% | "
            f"{pct(fn_b - fp_b, signed=True)} pp | {pct(fp_f)}% | {pct(fn_f)}% | "
            f"{pct(fn_f - fp_f, signed=True)} pp |"
        )

    lines.extend(
        [
            "",
            "### FN−FP threshold checks",
            "",
            "Thresholds: abs(FN−FP) ≤ 0.05 for near-equality; FN worse than FP by ≥ 0.05.",
            "",
            "| Aggregation | Metric | FN−FP | abs(Δ) ≤ 0.05 | FN worse ≥ 0.05 |",
            "| --- | --- | ---: | :---: | :---: |",
        ]
    )
    for aggregation, checks in signal["fn_minus_fp_threshold_checks"][
        "by_aggregation"
    ].items():
        for metric, row in checks.items():
            lines.append(
                f"| {aggregation} | `{metric}` | {pct(row['fn_minus_fp'], signed=True)} pp | "
                f"{'yes' if row['within_0_05'] else 'no'} | "
                f"{'yes' if row['fn_worse_by_at_least_0_05'] else 'no'} |"
            )

    lines.extend(
        [
            "",
            "## 3. Within-group rollout contrast (`analysis.json`)",
            "",
            "| Noise | Aggregation | Erased rate | Pairwise preserved | Created contrast |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in payload["rollout_groups"]["overall"]:
        lines.append(
            f"| {row['noise']} | {row['aggregation']} | "
            f"{pct(row['erased_rate'])}% ({row['erased_among_clean_variable']}/"
            f"{row['clean_variable_groups']}) | {pct(row['pairwise_preserved_rate'])}% | "
            f"{row['created_among_clean_constant']} |"
        )

    lines.extend(
        [
            "",
            "Hard tasks under binary (erasure):",
            "",
        ]
    )
    for row in payload["rollout_groups"]["by_difficulty"]:
        if row["difficulty"] == "hard" and row["aggregation"] == "binary":
            lines.append(
                f"- `{row['noise']}`: {pct(row['erased_rate'])}% "
                f"({row['erased']}/{row['clean_variable']})"
            )

    lines.extend(
        [
            "",
            "## 4. LoRA audit · actual training aggregation",
            "",
            "| Train aggregation · metric | FN mean | FP mean | FN−FP | n FN / n FP |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, row in payload["gradient_audit"]["fn_minus_fp"].items():
        lines.append(
            f"| `{key}` | {row['fn_mean']:.3f} | {row['fp_mean']:.3f} | "
            f"{row['fn_minus_fp']:+.3f} | {row['n_fn']} / {row['n_fp']} |"
        )

    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Endpoint: {payload['takeaways']['endpoint_pass_at_1']}",
            f"- Signal, binary: {payload['takeaways']['training_signal_binary']}",
            f"- Signal, fractional: {payload['takeaways']['training_signal_fractional']}",
            f"- Gradients: {payload['takeaways']['gradients']}",
            "",
            "[fp_vs_fn.json](fp_vs_fn.json)",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-json", type=Path, required=True)
    parser.add_argument("--stage6-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    payload = build_payload(analysis_path=args.analysis_json, stage6_path=args.stage6_json)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "fp_vs_fn.json"
    md_path = args.out_dir / "FP_VS_FN.md"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
