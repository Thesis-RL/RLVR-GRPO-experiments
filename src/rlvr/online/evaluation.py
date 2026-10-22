"""Resumable clean MBPP evaluation for smoke and scheduled checkpoints."""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlvr.data.mbpp import MBPPProblem
from rlvr.exec.mbpp_executor import grade_code
from rlvr.generate.sample import GenerationConfig, sample_completions
from rlvr.prompts.mbpp import build_prompt, extract_code
from rlvr.rewards.aggregation import binary_rewards, fractional_rewards
from rlvr.run_logging import RunContext, append_jsonl


def evaluate_clean_pass_at_1(
    model: Any,
    tokenizer: Any,
    problems: list[MBPPProblem],
    *,
    few_shot: list[MBPPProblem],
    context: RunContext,
    out_dir: str | Path,
    split_name: str,
    checkpoint: str,
    step: int,
    seed: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    timeout_s: float,
    mem_limit: int,
    difficulty_by_task: dict[int, str] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Generate one program per task, resuming safely at task granularity."""
    output = Path(out_dir) / "evaluations" / split_name / f"step-{step}"
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "predictions.jsonl"
    summary_path = output / "summary.json"
    existing = _read_predictions(predictions_path, step=step, checkpoint=checkpoint)
    expected_ids = {problem.task_id for problem in problems}
    completed = len(expected_ids.intersection(existing))
    if progress_callback is not None:
        progress_callback(completed, len(problems))

    if summary_path.exists() and expected_ids.issubset(existing):
        return json.loads(summary_path.read_text())

    generation = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )
    with _preserve_rng_state():
        was_training = bool(model.training)
        model.eval()
        try:
            for problem in problems:
                if problem.task_id in existing:
                    continue
                evaluation_seed = _task_evaluation_seed(seed, problem.task_id)
                _set_seed(evaluation_seed)
                prompt = build_prompt(problem, few_shot)
                completion = sample_completions(
                    model,
                    tokenizer,
                    prompt,
                    n=1,
                    gen=generation,
                    batch_size=1,
                )[0]
                code = extract_code(completion)
                result = grade_code(
                    code,
                    problem,
                    timeout_s=timeout_s,
                    mem_limit=mem_limit,
                )
                outcomes = np.asarray([result.outcomes], dtype=np.int8)
                row = {
                    "event": "clean_evaluation",
                    "split": split_name,
                    "step": step,
                    "checkpoint": checkpoint,
                    "task_id": problem.task_id,
                    "difficulty": (difficulty_by_task or {}).get(problem.task_id, "unknown"),
                    "evaluation_seed": evaluation_seed,
                    "prompt": prompt,
                    "completion": completion,
                    "code": code,
                    "exec_error": result.error,
                    "clean_outcomes": list(result.outcomes),
                    "clean_binary_reward": float(binary_rewards(outcomes)[0]),
                    "clean_fractional_reward": float(fractional_rewards(outcomes)[0]),
                }
                append_jsonl(predictions_path, row, context=context)
                existing[problem.task_id] = row
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, len(problems))
        finally:
            if was_training:
                model.train()

    selected = [existing[problem.task_id] for problem in problems]
    summary = _summarize_rows(
        selected,
        split_name=split_name,
        checkpoint=checkpoint,
        step=step,
        seed=seed,
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def evaluation_steps(max_steps: int, interval_fraction: float) -> tuple[int, ...]:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not 0.0 < interval_fraction <= 1.0:
        raise ValueError("interval_fraction must be in (0, 1]")
    count = max(1, int(round(1.0 / interval_fraction)))
    return tuple(
        sorted({max(1, round(max_steps * index / count)) for index in range(1, count + 1)})
    )


def fraction_steps(max_steps: int, fractions: list[float]) -> tuple[int, ...]:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not fractions or any(not 0.0 < fraction <= 1.0 for fraction in fractions):
        raise ValueError("evaluation fractions must be a non-empty list in (0, 1]")
    return tuple(sorted({max(1, round(max_steps * fraction)) for fraction in fractions}))


def summarize_primary_metric(
    out_dir: str | Path,
    *,
    split_name: str,
    metric: str,
    last_n: int,
    require_count: bool,
) -> dict[str, Any]:
    if last_n <= 0:
        raise ValueError("last_n must be positive")
    summaries: list[dict[str, Any]] = []
    root = Path(out_dir) / "evaluations" / split_name
    paths = root.glob("step-*/summary.json") if root.exists() else ()
    for path in paths:
        summaries.append(json.loads(path.read_text()))
    summaries.sort(key=lambda item: int(item["step"]))
    if require_count and len(summaries) < last_n:
        raise RuntimeError(
            f"Expected at least {last_n} {split_name} evaluations, found {len(summaries)}"
        )
    selected = summaries[-last_n:]
    values = [float(item[metric]) for item in selected]
    by_difficulty: dict[str, dict[str, Any]] = {}
    for stratum in ("hard", "medium", "easy"):
        stratum_values = [
            float(item["by_difficulty"][stratum][metric])
            for item in selected
            if stratum in item.get("by_difficulty", {})
        ]
        if not stratum_values:
            continue
        counts = {
            int(item["by_difficulty"][stratum]["n_problems"])
            for item in selected
            if stratum in item.get("by_difficulty", {})
        }
        if len(counts) != 1:
            raise RuntimeError(f"Difficulty stratum size changed across evaluations: {stratum}")
        by_difficulty[stratum] = {
            "n_problems": counts.pop(),
            "values": stratum_values,
            "mean": float(np.mean(stratum_values)),
        }
    result = {
        "split": split_name,
        "metric": metric,
        "last_n": last_n,
        "selected_steps": [int(item["step"]) for item in selected],
        "values": values,
        "mean": float(np.mean(values)) if values else None,
        "by_difficulty": by_difficulty,
    }
    (Path(out_dir) / "primary_metric_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _read_predictions(path: Path, *, step: int, checkpoint: str) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid evaluation JSONL at {path}:{line_number}") from exc
        if int(row.get("step", -1)) != step or row.get("checkpoint") != checkpoint:
            raise RuntimeError(f"Evaluation metadata mismatch in {path}:{line_number}")
        task_id = int(row["task_id"])
        if task_id in rows:
            raise RuntimeError(f"Duplicate task_id {task_id} in {path}")
        rows[task_id] = row
    return rows


def _summarize_rows(
    rows: list[dict[str, Any]],
    *,
    split_name: str,
    checkpoint: str,
    step: int,
    seed: int,
) -> dict[str, Any]:
    binary = [float(row["clean_binary_reward"]) for row in rows]
    fractional = [float(row["clean_fractional_reward"]) for row in rows]
    by_difficulty: dict[str, dict[str, float | int]] = {}
    for stratum in ("hard", "medium", "easy"):
        selected = [row for row in rows if row.get("difficulty") == stratum]
        if not selected:
            continue
        by_difficulty[stratum] = {
            "n_problems": len(selected),
            "pass_at_1": float(np.mean([row["clean_binary_reward"] for row in selected])),
            "mean_test_pass_rate": float(
                np.mean([row["clean_fractional_reward"] for row in selected])
            ),
        }
    return {
        "split": split_name,
        "checkpoint": checkpoint,
        "step": step,
        "seed": seed,
        "n_problems": len(rows),
        "pass_at_1": float(np.mean(binary)) if binary else None,
        "mean_test_pass_rate": float(np.mean(fractional)) if fractional else None,
        "by_difficulty": by_difficulty,
    }


def _task_evaluation_seed(seed: int, task_id: int) -> int:
    sequence = np.random.SeedSequence([seed, task_id])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def _preserve_rng_state() -> Iterator[None]:
    """Keep scheduled evaluation from perturbing subsequent training draws."""
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
