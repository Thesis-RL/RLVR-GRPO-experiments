#!/usr/bin/env python3
"""Sample frozen rollouts, clean-grade, apply planned noise, measure signal survival.

Usage:
  uv run python scripts/run_offline_diagnostic.py --model Qwen/Qwen3-1.7B-Base
  uv run python scripts/run_offline_diagnostic.py --model Qwen/Qwen3-1.7B-Base --limit 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rlvr.config import load_config, resolve_model_config  # noqa: E402
from rlvr.data.mbpp import MBPPProblem, load_mbpp  # noqa: E402
from rlvr.exec.mbpp_executor import grade_codes  # noqa: E402
from rlvr.generate.sample import GenerationConfig, load_model, sample_completions  # noqa: E402
from rlvr.metrics.signal import (  # noqa: E402
    difficulty_stratum,
    group_signal_stats,
    stats_to_dict,
    summarize_groups,
)
from rlvr.noise.corruption import apply_noise, resolve_noise_specs  # noqa: E402
from rlvr.prompts.mbpp import build_prompt, extract_code  # noqa: E402
from rlvr.rewards.aggregation import (  # noqa: E402
    binary_rewards,
    fractional_rewards,
    resolve_reward_functions,
)
from rlvr.run_logging import RunContext, write_run_manifest  # noqa: E402


def _grade_job(args: tuple) -> dict:
    task_id, model_id, codes, test_list, test_setup_code, timeout_s = args
    problem = MBPPProblem(
        task_id=task_id,
        text="",
        code="",
        test_list=list(test_list),
        test_setup_code=str(test_setup_code or ""),
        challenge_test_list=[],
        split="train",
    )
    results = grade_codes(codes, problem, timeout_s=timeout_s)
    outcomes = [list(r.outcomes) for r in results]
    errors = [r.error for r in results]
    outcomes_arr = np.asarray(outcomes, dtype=np.int8)
    frac = fractional_rewards(outcomes_arr)
    bin_ = binary_rewards(outcomes_arr)
    mean_frac = float(frac.mean())
    return {
        "task_id": task_id,
        "model": model_id,
        "outcomes": outcomes,
        "errors": errors,
        "fractional_rewards": frac.tolist(),
        "binary_rewards": bin_.tolist(),
        "mean_fractional": mean_frac,
        "stratum": difficulty_stratum(mean_frac),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/offline_diagnostic.yaml")
    p.add_argument("--model", type=str, default=None, help="Override the model selected by config")
    p.add_argument("--role", type=str, default=None, help="Override the configured model role")
    p.add_argument("--limit", type=int, default=None, help="Optional cap on #training problems")
    p.add_argument("--skip-generate", action="store_true", help="Reuse saved rollouts JSONL")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--dtype", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model_cfg = resolve_model_config(cfg, model_id=args.model, role=args.role)
    model_id = model_cfg.id
    model_role = model_cfg.role
    safe_name = model_id.replace("/", "__")
    out_dir = args.out_dir or (ROOT / "outputs" / "offline_diagnostic" / safe_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    group_size = int(cfg["group_size"])
    noise_draws = int(cfg["noise_draws"])
    noise_base_seed = int(cfg["noise_base_seed"])
    batch_size = int(args.batch_size or cfg["batch_size"])
    dtype = args.dtype or model_cfg.dtype
    timeout_s = float(cfg["timeout_s"])
    noise_specs = resolve_noise_specs(cfg["noise_specs"])
    reward_functions = resolve_reward_functions(cfg["aggregations"])

    write_run_manifest(
        out_dir,
        context=RunContext(
            pipeline="offline_diagnostic",
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
                "limit": args.limit,
                "batch_size": args.batch_size,
                "dtype": args.dtype,
                "out_dir": str(args.out_dir) if args.out_dir else None,
                "skip_generate": args.skip_generate,
            }.items()
            if value is not None and value is not False
        },
    )

    train = load_mbpp("train")
    prompt_split = {p.task_id: p for p in load_mbpp("prompt")}
    few_shot_ids = list(cfg["few_shot_task_ids"])
    few_shot = [prompt_split[i] for i in few_shot_ids if i in prompt_split]
    if len(few_shot) < len(few_shot_ids):
        # Official prompt split may not contain all IDs; fall back to any prompt examples.
        few_shot = list(prompt_split.values())[:3]

    if args.limit is not None:
        train = train[: args.limit]

    rollouts_path = out_dir / "rollouts.jsonl"
    clean_path = out_dir / "clean_grades.jsonl"

    if not args.skip_generate:
        print(f"Loading model {model_id} ({dtype})…")
        t0 = time.time()
        model, tokenizer = load_model(model_id, dtype=dtype)
        print(f"Model ready in {time.time() - t0:.1f}s")
        gen = GenerationConfig(
            temperature=float(cfg["temperature"]),
            top_p=float(cfg["top_p"]),
            max_new_tokens=int(cfg["max_new_tokens"]),
        )
        with rollouts_path.open("w") as f:
            for problem in tqdm(train, desc="generate"):
                prompt = build_prompt(problem, few_shot)
                completions = sample_completions(
                    model, tokenizer, prompt, n=group_size, gen=gen, batch_size=batch_size
                )
                codes = [extract_code(c) for c in completions]
                row = {
                    "task_id": problem.task_id,
                    "model": model_id,
                    "prompt": prompt,
                    "completions": completions,
                    "codes": codes,
                }
                f.write(json.dumps(row) + "\n")
                f.flush()
        del model
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    elif not rollouts_path.exists():
        raise SystemExit(f"--skip-generate set but missing {rollouts_path}")

    # Clean grade (CPU-only; can resume / rerun without the model)
    problems = {p.task_id: p for p in train}
    difficulty: dict[int, dict] = {}
    # Sequential: each candidate already runs in a memory-capped child.
    # Parallel pools of nested processes previously tripped the OOM killer.
    jobs: list[tuple] = []
    with rollouts_path.open() as fin:
        for line in fin:
            row = json.loads(line)
            problem = problems[row["task_id"]]
            jobs.append(
                (
                    row["task_id"],
                    model_id,
                    row["codes"],
                    problem.test_list,
                    problem.test_setup_code,
                    timeout_s,
                )
            )

    with clean_path.open("w") as fout:
        for job in tqdm(jobs, desc="clean-grade"):
            out = _grade_job(job)
            difficulty[out["task_id"]] = {
                "mean_fractional": out["mean_fractional"],
                "mean_binary": float(np.mean(out["binary_rewards"])),
                "stratum": out["stratum"],
                "nonzero_variance_frac": bool(np.std(out["fractional_rewards"], ddof=0) > 1e-12),
            }
            fout.write(json.dumps(out) + "\n")
            fout.flush()

    (out_dir / "difficulty.json").write_text(json.dumps(difficulty, indent=2))

    # Offline noise + signal metrics
    signal_rows: list[dict] = []
    summaries: dict[str, dict] = {}

    clean_by_id: dict[int, np.ndarray] = {}
    with clean_path.open() as f:
        for line in f:
            row = json.loads(line)
            clean_by_id[row["task_id"]] = np.asarray(row["outcomes"], dtype=np.int8)

    for spec in noise_specs:
        spec_name = spec.name
        for agg_name, agg_fn in reward_functions.items():
            key = f"{spec_name}__{agg_name}"
            per_draw_summaries = []
            for draw in range(noise_draws):
                stats_list = []
                for task_id, clean in clean_by_id.items():
                    rng = np.random.default_rng(noise_base_seed + 1_000_003 * draw + task_id)
                    noisy, mask = apply_noise(clean, spec, rng)
                    clean_r = agg_fn(clean)
                    noisy_r = agg_fn(noisy)
                    stats = group_signal_stats(clean_r, noisy_r)
                    stats_list.append(stats)
                    signal_rows.append(
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
                per_draw_summaries.append(summarize_groups(stats_list))
            # Average numeric fields across draws
            summaries[key] = _mean_summaries(per_draw_summaries)

    signal_path = out_dir / "signal_by_group.jsonl"
    with signal_path.open("w") as f:
        for row in signal_rows:
            f.write(json.dumps(row) + "\n")

    # Model screening (pre-registered rule)
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

    # Prop. 1 offline deltas
    prop1 = {}
    for agg in ("fractional", "binary"):
        fp = summaries.get(f"fp_only__{agg}", {})
        fn = summaries.get(f"fn_only__{agg}", {})
        prop1[agg] = {
            "delta_advantage_sign_agreement": _sub(
                fn.get("mean_advantage_sign_agreement"),
                fp.get("mean_advantage_sign_agreement"),
            ),
            "delta_pairwise_fidelity": _sub(
                fn.get("mean_pairwise_fidelity"),
                fp.get("mean_pairwise_fidelity"),
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
            "group_size": group_size,
            "noise_draws": noise_draws,
            "noise_specs": [spec.name for spec in noise_specs],
            "aggregations": list(reward_functions),
            "n_problems": len(difficulty),
        },
        "screening": screening,
        "summaries": summaries,
        "proposition1_offline": prop1,
        "pre_registration": str(ROOT / "notes/pre_registration.md"),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    _write_markdown_report(out_dir / "report.md", report)
    print(json.dumps({"out_dir": str(out_dir), "screening": screening, "prop1": prop1}, indent=2))


def _sub(a, b):
    if a is None or b is None:
        return None
    return float(a) - float(b)


def _mean_summaries(items: list[dict]) -> dict:
    if not items:
        return {}
    keys = items[0].keys()
    out: dict = {}
    for k in keys:
        vals = [it[k] for it in items]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            out[k] = float(np.mean(vals))
        elif all(isinstance(v, bool) for v in vals):
            out[k] = bool(np.mean(vals) >= 0.5)
        elif all(isinstance(v, str) for v in vals):
            # majority bucket label
            out[k] = max(set(vals), key=vals.count)
        else:
            out[k] = vals[0]
    return out


def _write_markdown_report(path: Path, report: dict) -> None:
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
        lines.append(
            f"| {agg} | {_fmt(block['delta_advantage_sign_agreement'])} | {_fmt(block['delta_pairwise_fidelity'])} |"
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
    if x is None:
        return "—"
    return f"{float(x):.3f}"


if __name__ == "__main__":
    main()
