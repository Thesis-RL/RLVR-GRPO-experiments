#!/usr/bin/env python3
"""Memory-safe clean-grade + noise analysis over saved rollouts (no model, no GPU).

Streams one problem at a time. Each candidate runs in a forked child with a
256MB address-space cap so buggy generated code cannot OOM the laptop.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rlvr.config import load_config, resolve_model_config  # noqa: E402
from rlvr.data.mbpp import load_mbpp  # noqa: E402
from rlvr.exec.mbpp_executor import grade_codes  # noqa: E402
from rlvr.metrics.signal import (  # noqa: E402
    difficulty_stratum,
    group_signal_stats,
    stats_to_dict,
    summarize_groups,
)
from rlvr.noise.corruption import apply_noise, resolve_noise_specs  # noqa: E402
from rlvr.rewards.aggregation import (  # noqa: E402
    binary_rewards,
    fractional_rewards,
    resolve_reward_functions,
)
from rlvr.run_logging import RunContext, write_run_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/offline_diagnostic.yaml")
    p.add_argument("--model", type=str, default=None, help="Override the model selected by config")
    p.add_argument("--role", type=str, default=None, help="Override the configured model role")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--mem-mb", type=int, default=64, help="Per-code child RAM cap (MiB)")
    p.add_argument("--timeout-s", type=float, default=0.5)
    p.add_argument(
        "--grade-only",
        action="store_true",
        help="Only write clean_grades.jsonl (skip noise analysis)",
    )
    p.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only build report from existing clean_grades.jsonl",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model_cfg = resolve_model_config(cfg, model_id=args.model, role=args.role)
    model_id = model_cfg.id
    model_role = model_cfg.role
    safe_name = model_id.replace("/", "__")
    out_dir = args.out_dir or (ROOT / "outputs" / "offline_diagnostic" / safe_name)
    rollouts_path = out_dir / "rollouts.jsonl"
    if not rollouts_path.exists():
        raise SystemExit(f"Missing rollouts: {rollouts_path}")

    clean_path = out_dir / "clean_grades.jsonl"
    mem_bytes = args.mem_mb * 1024 * 1024
    timeout_s = float(args.timeout_s)
    noise_specs = resolve_noise_specs(cfg["noise_specs"])
    reward_functions = resolve_reward_functions(cfg["aggregations"])

    write_run_manifest(
        out_dir,
        context=RunContext(
            pipeline="offline_grade_analyze",
            experiment="offline_diagnostic",
            study="shared",
            model=model_id,
            model_role=model_role,
        ),
        config=cfg,
        config_path=args.config,
        overrides={
            key: value
            for key, value in {
                "model": args.model,
                "role": args.role,
                "out_dir": str(args.out_dir) if args.out_dir else None,
                "mem_mb": args.mem_mb,
                "timeout_s": args.timeout_s,
                "grade_only": args.grade_only,
                "analyze_only": args.analyze_only,
            }.items()
            if value is not None and value is not False
        },
    )

    if not args.analyze_only:
        problems = {p.task_id: p for p in load_mbpp("train")}
        # Resume: keep already graded task_ids; drop a truncated last line if any
        done: set[int] = set()
        if clean_path.exists() and clean_path.stat().st_size > 0:
            valid_lines: list[str] = []
            with clean_path.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        done.add(json.loads(line)["task_id"])
                        valid_lines.append(line if line.endswith("\n") else line + "\n")
                    except json.JSONDecodeError:
                        continue
            clean_path.write_text("".join(valid_lines))
            mode = "a"
        else:
            mode = "w"

        n_total = sum(1 for _ in rollouts_path.open())
        print(
            f"Grading {n_total} problems ({len(done)} already done), "
            f"mem_cap={args.mem_mb}MB, timeout={timeout_s}s"
        )

        with rollouts_path.open() as fin, clean_path.open(mode) as fout:
            for line in tqdm(fin, total=n_total, desc="clean-grade"):
                row = json.loads(line)
                task_id = row["task_id"]
                if task_id in done:
                    continue
                problem = problems[task_id]
                graded = grade_codes(
                    row["codes"],
                    problem,
                    timeout_s=timeout_s,
                    mem_limit=mem_bytes,
                )
                outcomes = [list(r.outcomes) for r in graded]
                errors = [r.error for r in graded]
                outcomes_arr = np.asarray(outcomes, dtype=np.int8)
                frac = fractional_rewards(outcomes_arr)
                bin_ = binary_rewards(outcomes_arr)
                mean_frac = float(frac.mean())
                out = {
                    "task_id": task_id,
                    "model": model_id,
                    "outcomes": outcomes,
                    "errors": errors,
                    "fractional_rewards": frac.tolist(),
                    "binary_rewards": bin_.tolist(),
                    "mean_fractional": mean_frac,
                    "stratum": difficulty_stratum(mean_frac),
                }
                fout.write(json.dumps(out) + "\n")
                fout.flush()

        if args.grade_only:
            n_done = sum(1 for _ in clean_path.open() if _.strip())
            print(json.dumps({"status": "grade_only_done", "n_graded": n_done}))
            return

    # Build difficulty + noise summaries streaming from clean_grades
    difficulty: dict[int, dict] = {}
    clean_by_id: dict[int, np.ndarray] = {}
    with clean_path.open() as f:
        for line in f:
            row = json.loads(line)
            outcomes = np.asarray(row["outcomes"], dtype=np.int8)
            clean_by_id[row["task_id"]] = outcomes
            frac = fractional_rewards(outcomes)
            difficulty[row["task_id"]] = {
                "mean_fractional": float(frac.mean()),
                "mean_binary": float(binary_rewards(outcomes).mean()),
                "stratum": row["stratum"],
                "nonzero_variance_frac": bool(frac.std(ddof=0) > 1e-12),
            }
    (out_dir / "difficulty.json").write_text(json.dumps(difficulty, indent=2))

    noise_draws = int(cfg["noise_draws"])
    noise_base_seed = int(cfg["noise_base_seed"])
    summaries: dict[str, dict] = {}
    signal_path = out_dir / "signal_by_group.jsonl"

    with signal_path.open("w") as sigf:
        for spec in noise_specs:
            spec_name = spec.name
            for agg_name, agg_fn in reward_functions.items():
                key = f"{spec_name}__{agg_name}"
                per_draw = []
                for draw in range(noise_draws):
                    stats_list = []
                    for task_id, clean in clean_by_id.items():
                        rng = np.random.default_rng(noise_base_seed + 1_000_003 * draw + task_id)
                        noisy, mask = apply_noise(clean, spec, rng)
                        stats = group_signal_stats(agg_fn(clean), agg_fn(noisy))
                        stats_list.append(stats)
                        sigf.write(
                            json.dumps(
                                {
                                    "task_id": task_id,
                                    "model": model_id,
                                    "noise": spec_name,
                                    "aggregation": agg_name,
                                    "noise_draw": draw,
                                    "corruption_rate": float(mask.mean()),
                                    "stratum": difficulty[task_id]["stratum"],
                                    **stats_to_dict(stats),
                                }
                            )
                            + "\n"
                        )
                    per_draw.append(summarize_groups(stats_list))
                summaries[key] = _mean_summaries(per_draw)

    mean_frac = float(np.mean([d["mean_fractional"] for d in difficulty.values()]))
    frac_nonzero = float(np.mean([d["nonzero_variance_frac"] for d in difficulty.values()]))
    screening = {
        "model": model_id,
        "role": model_role,
        "n_problems": len(difficulty),
        "mean_fractional_reward": mean_frac,
        "frac_problems_nonzero_variance": frac_nonzero,
        "eligible_as_main": (0.05 <= mean_frac <= 0.60) and (frac_nonzero >= 0.15),
        "stratum_counts": {
            s: sum(1 for d in difficulty.values() if d["stratum"] == s)
            for s in ("easy", "medium", "hard")
        },
    }

    prop1 = {}
    for agg in ("fractional", "binary"):
        fp = summaries.get(f"fp_only__{agg}", {})
        fn = summaries.get(f"fn_only__{agg}", {})
        prop1[agg] = {
            "delta_advantage_sign_agreement": _sub(
                fn.get("mean_advantage_sign_agreement"), fp.get("mean_advantage_sign_agreement")
            ),
            "delta_pairwise_fidelity": _sub(
                fn.get("mean_pairwise_fidelity"), fp.get("mean_pairwise_fidelity")
            ),
            "fp": {
                "agreement": fp.get("mean_advantage_sign_agreement"),
                "pairwise": fp.get("mean_pairwise_fidelity"),
                "agreement_bucket": fp.get("mean_advantage_sign_agreement_bucket"),
                "pairwise_bucket": fp.get("mean_pairwise_fidelity_bucket"),
            },
            "fn": {
                "agreement": fn.get("mean_advantage_sign_agreement"),
                "pairwise": fn.get("mean_pairwise_fidelity"),
                "agreement_bucket": fn.get("mean_advantage_sign_agreement_bucket"),
                "pairwise_bucket": fn.get("mean_pairwise_fidelity_bucket"),
            },
        }

    report = {
        "model": model_id,
        "role": model_role,
        "config": {
            "group_size": cfg["group_size"],
            "noise_draws": noise_draws,
            "noise_specs": [spec.name for spec in noise_specs],
            "aggregations": list(reward_functions),
            "n_problems": len(difficulty),
            "mem_mb": args.mem_mb,
            "timeout_s": timeout_s,
        },
        "screening": screening,
        "summaries": summaries,
        "proposition1_offline": prop1,
        "pre_registration": str(ROOT / "notes/pre_registration.md"),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    _write_markdown(out_dir / "report.md", report)
    print(json.dumps({"out_dir": str(out_dir), "screening": screening, "prop1": prop1}, indent=2))


def _sub(a, b):
    if a is None or b is None:
        return None
    return float(a) - float(b)


def _mean_summaries(items: list[dict]) -> dict:
    if not items:
        return {}
    out: dict = {}
    for k in items[0]:
        vals = [it[k] for it in items]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            out[k] = float(np.mean(vals))
        elif all(isinstance(v, str) for v in vals):
            out[k] = max(set(vals), key=vals.count)
        else:
            out[k] = vals[0]
    return out


def _write_markdown(path: Path, report: dict) -> None:
    s = report["screening"]
    lines = [
        f"# Offline diagnostic — `{report['model']}`",
        "",
        "## Model screening (pre-registered)",
        "",
        f"- Mean fractional reward: **{s['mean_fractional_reward']:.4f}** (need ∈ [0.05, 0.60])",
        f"- Frac problems with nonzero clean variance: **{s['frac_problems_nonzero_variance']:.3f}** (need ≥ 0.15)",
        f"- Eligible as Phase-1 main: **{s['eligible_as_main']}**",
        f"- Strata: {s['stratum_counts']}",
        "",
        "## Proposition 1 offline (FN − FP)",
        "",
        "| Aggregation | Δ agreement (FN−FP) | Δ pairwise (FN−FP) |",
        "|---|---:|---:|",
    ]
    for agg, block in report["proposition1_offline"].items():
        da = block["delta_advantage_sign_agreement"]
        dp = block["delta_pairwise_fidelity"]
        lines.append(
            f"| {agg} | {da if da is None else f'{da:.3f}'} | {dp if dp is None else f'{dp:.3f}'} |"
        )
    lines += ["", "## Signal summaries", ""]
    lines.append("| Condition | Agr. | Cos | Pairwise | Top-1 | Agr. bucket | Pair bucket |")
    lines.append("|---|---:|---:|---:|---:|---|---|")
    for key, sm in sorted(report["summaries"].items()):
        lines.append(
            "| {k} | {a} | {c} | {p} | {t} | {ab} | {pb} |".format(
                k=key,
                a=_fmt(sm.get("mean_advantage_sign_agreement")),
                c=_fmt(sm.get("mean_advantage_cosine")),
                p=_fmt(sm.get("mean_pairwise_fidelity")),
                t=_fmt(sm.get("mean_top1_preserved")),
                ab=sm.get("mean_advantage_sign_agreement_bucket"),
                pb=sm.get("mean_pairwise_fidelity_bucket"),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def _fmt(x) -> str:
    return "—" if x is None else f"{float(x):.3f}"


if __name__ == "__main__":
    main()
