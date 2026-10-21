"""Clean execution, controlled corruption, rewards, and rollout logging."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rlvr.config import ExperimentCondition
from rlvr.data.mbpp import MBPPProblem
from rlvr.exec.mbpp_executor import grade_codes
from rlvr.metrics.signal import group_signal_stats, stats_to_dict
from rlvr.noise.corruption import apply_noise
from rlvr.prompts.mbpp import extract_code
from rlvr.rewards.aggregation import binary_rewards, fractional_rewards, grpo_advantages
from rlvr.run_logging import RunContext, append_jsonl_rows


@dataclass
class MBPPNoisyReward:
    """TRL-compatible reward callable for one configured online condition."""

    condition: ExperimentCondition
    seed: int
    group_size: int
    context: RunContext
    rollout_log_path: Path | None = None
    timeout_s: float = 0.5
    mem_limit: int = 64 * 1024 * 1024
    save_clean_outcomes: bool = True
    save_noisy_outcomes: bool = True
    save_both_rewards: bool = True
    save_advantages: bool = True
    reward_seconds: float = 0.0
    grading_seconds: float = 0.0
    generated_tokens: int = 0
    calls: int = 0

    def __post_init__(self) -> None:
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.rollout_log_path is not None and self.rollout_log_path.exists():
            self.calls, self.generated_tokens = _resume_rollout_state(self.rollout_log_path)

    def __call__(
        self,
        prompts: list[Any],
        completions: list[Any],
        *,
        task_id: list[int],
        test_list: list[list[str]],
        test_setup_code: list[str],
        completion_ids: list[list[int]] | None = None,
        difficulty: list[str] | None = None,
        trainer_state: Any | None = None,
        log_extra: Any | None = None,
        log_metric: Any | None = None,
        **_: Any,
    ) -> list[float]:
        call_started = time.perf_counter()
        n = len(completions)
        self._validate_batch(n, prompts, task_id, test_list, test_setup_code)
        if completion_ids is None:
            completion_ids = [[] for _ in range(n)]
        if difficulty is None:
            difficulty = ["unknown" for _ in range(n)]

        step = int(getattr(trainer_state, "global_step", 0))
        verification_call = self.calls
        codes = [extract_code(_completion_text(completion)) for completion in completions]
        self.generated_tokens += sum(len(ids) for ids in completion_ids)
        logs: list[dict[str, Any]] = []
        train_rewards: list[float] = []

        for group_start in range(0, n, self.group_size):
            group_end = group_start + self.group_size
            group_index = group_start // self.group_size
            group_task_ids = task_id[group_start:group_end]
            if len(set(group_task_ids)) != 1:
                raise ValueError(
                    "Expected each contiguous GRPO group to contain one task_id, "
                    f"got {group_task_ids}"
                )

            current_task_id = int(group_task_ids[0])
            problem = MBPPProblem(
                task_id=current_task_id,
                text="",
                code="",
                test_list=list(test_list[group_start]),
                test_setup_code=str(test_setup_code[group_start] or ""),
                challenge_test_list=[],
                split="train",
            )
            grading_started = time.perf_counter()
            try:
                graded = grade_codes(
                    codes[group_start:group_end],
                    problem,
                    timeout_s=self.timeout_s,
                    mem_limit=self.mem_limit,
                )
            finally:
                self.grading_seconds += time.perf_counter() - grading_started
            clean = np.asarray([result.outcomes for result in graded], dtype=np.int8)
            noise_seed = _noise_seed(
                self.seed,
                step,
                current_task_id,
                verification_call,
                group_index,
            )
            noisy, mask = apply_noise(
                clean,
                self.condition.noise,
                np.random.default_rng(noise_seed),
            )

            clean_binary = binary_rewards(clean)
            clean_fractional = fractional_rewards(clean)
            noisy_binary = binary_rewards(noisy)
            noisy_fractional = fractional_rewards(noisy)
            selected = noisy_binary if self.condition.reward == "binary" else noisy_fractional
            clean_selected = clean_binary if self.condition.reward == "binary" else clean_fractional

            clean_binary_adv = grpo_advantages(clean_binary, eps=1e-4)
            clean_fractional_adv = grpo_advantages(clean_fractional, eps=1e-4)
            noisy_binary_adv = grpo_advantages(noisy_binary, eps=1e-4)
            noisy_fractional_adv = grpo_advantages(noisy_fractional, eps=1e-4)
            selected_adv = (
                noisy_binary_adv if self.condition.reward == "binary" else noisy_fractional_adv
            )

            group_stats = _group_corruption_stats(
                clean,
                noisy,
                mask,
                alpha=self.condition.noise.alpha,
                beta=self.condition.noise.beta,
                class_conditional=self.condition.noise.mode == "class_conditional",
                noise_mode=self.condition.noise.mode,
                flip_p=self.condition.noise.flip_p,
            )
            signal_stats = {
                f"train_aggregation_{key}": value
                for key, value in stats_to_dict(
                    group_signal_stats(clean_selected, selected)
                ).items()
            }
            group_id = (
                f"step-{step}__call-{verification_call}__task-{current_task_id}"
                f"__group-{group_index}"
            )
            selected_zero_variance = bool(selected.std(ddof=0) < 1e-12)
            train_rewards.extend(float(value) for value in selected)

            for offset, result in enumerate(graded):
                index = group_start + offset
                row = {
                    "step": step,
                    "policy_step": step,
                    "verification_call": verification_call,
                    "group_id": group_id,
                    "group_index": group_index,
                    "completion_index": offset,
                    "task_id": current_task_id,
                    "difficulty": difficulty[index],
                    "prompt": prompts[index],
                    "completion": _completion_text(completions[index]),
                    "code": codes[index],
                    "completion_token_count": len(completion_ids[index]),
                    "exec_error": result.error,
                    "clean_outcomes": clean[offset].tolist(),
                    "corruption_mask": mask[offset].tolist(),
                    "noisy_outcomes": noisy[offset].tolist(),
                    "clean_binary_reward": float(clean_binary[offset]),
                    "clean_fractional_reward": float(clean_fractional[offset]),
                    "noisy_binary_reward": float(noisy_binary[offset]),
                    "noisy_fractional_reward": float(noisy_fractional[offset]),
                    "train_reward": float(selected[offset]),
                    "clean_binary_advantage": float(clean_binary_adv[offset]),
                    "clean_fractional_advantage": float(clean_fractional_adv[offset]),
                    "noisy_binary_advantage": float(noisy_binary_adv[offset]),
                    "noisy_fractional_advantage": float(noisy_fractional_adv[offset]),
                    "train_advantage": float(selected_adv[offset]),
                    "zero_variance_group": selected_zero_variance,
                    "noise_seed": noise_seed,
                    **group_stats,
                    **signal_stats,
                }
                if not self.save_clean_outcomes:
                    row.pop("clean_outcomes")
                if not self.save_noisy_outcomes:
                    row.pop("corruption_mask")
                    row.pop("noisy_outcomes")
                if not self.save_both_rewards:
                    selected_prefix = (
                        "noisy_binary" if self.condition.reward == "binary" else "noisy_fractional"
                    )
                    for key in (
                        "clean_binary_reward",
                        "clean_fractional_reward",
                        "noisy_binary_reward",
                        "noisy_fractional_reward",
                    ):
                        if not key.startswith(selected_prefix):
                            row.pop(key)
                if not self.save_advantages:
                    for key in tuple(row):
                        if key.endswith("_advantage"):
                            row.pop(key)
                logs.append(row)

        try:
            if self.rollout_log_path is not None:
                append_jsonl_rows(self.rollout_log_path, logs, context=self.context)
            _log_trl_extras(log_extra, logs)
            _log_trl_metrics(log_metric, logs)
            return train_rewards
        finally:
            self.calls += 1
            self.reward_seconds += time.perf_counter() - call_started

    def _validate_batch(self, n: int, *columns: list[Any]) -> None:
        if n == 0:
            raise ValueError("Reward batch must not be empty")
        if n % self.group_size != 0:
            raise ValueError(
                f"Completion count {n} must be divisible by group_size {self.group_size}"
            )
        if any(len(column) != n for column in columns):
            raise ValueError("All reward columns must match the number of completions")


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict) and "content" in last:
            return str(last["content"])
    raise TypeError(f"Unsupported completion format: {type(completion).__name__}")


def _noise_seed(
    seed: int,
    step: int,
    task_id: int,
    verification_call: int,
    group_index: int,
) -> int:
    sequence = np.random.SeedSequence([seed, step, task_id, verification_call, group_index])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _resume_rollout_state(path: Path) -> tuple[int, int]:
    """Recover verifier-call and token counters from a resumable rollout log."""
    latest = -1
    generated_tokens = 0
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid rollout JSONL at {path}:{line_number}") from exc
            if "verification_call" not in row:
                raise RuntimeError(
                    f"Cannot resume {path}: verification_call is missing at line {line_number}"
                )
            latest = max(latest, int(row["verification_call"]))
            generated_tokens += int(row.get("completion_token_count", 0))
    return latest + 1, generated_tokens


def _group_corruption_stats(
    clean: np.ndarray,
    noisy: np.ndarray,
    mask: np.ndarray,
    *,
    alpha: float,
    beta: float,
    class_conditional: bool,
    noise_mode: str,
    flip_p: float,
) -> dict[str, float | int | str | None]:
    clean_fail = int((clean == 0).sum())
    clean_pass = int((clean == 1).sum())
    false_positives = int(((clean == 0) & (noisy == 1)).sum())
    false_negatives = int(((clean == 1) & (noisy == 0)).sum())
    true_positives = int(((clean == 1) & (noisy == 1)).sum())
    true_negatives = int(((clean == 0) & (noisy == 0)).sum())
    realized_alpha = false_positives / clean_fail if clean_fail else None
    realized_beta = false_negatives / clean_pass if clean_pass else None
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else None
    )
    recall = true_positives / clean_pass if clean_pass else None
    return {
        "noise_mode": noise_mode,
        "changed_label_rate": float(mask.mean()),
        "true_positive_count": true_positives,
        "true_negative_count": true_negatives,
        "false_positive_count": false_positives,
        "false_negative_count": false_negatives,
        "verifier_precision": precision,
        "verifier_recall": recall,
        "realized_alpha": realized_alpha,
        "realized_beta": realized_beta,
        "configured_alpha": alpha if class_conditional else None,
        "configured_beta": beta if class_conditional else None,
        "configured_j": float(1.0 - alpha - beta) if class_conditional else None,
        "configured_flip_p": flip_p if not class_conditional else None,
    }


def _log_trl_extras(log_extra: Any | None, logs: list[dict[str, Any]]) -> None:
    if log_extra is None:
        return
    columns = (
        "task_id",
        "group_id",
        "code",
        "clean_outcomes",
        "corruption_mask",
        "noisy_outcomes",
        "clean_binary_reward",
        "clean_fractional_reward",
        "noisy_binary_reward",
        "noisy_fractional_reward",
        "train_reward",
        "zero_variance_group",
    )
    for column in columns:
        if not logs or column not in logs[0]:
            continue
        values = [row[column] for row in logs]
        if column in {"clean_outcomes", "corruption_mask", "noisy_outcomes"}:
            values = [json.dumps(value) for value in values]
        log_extra(column, values)


def _log_trl_metrics(log_metric: Any | None, logs: list[dict[str, Any]]) -> None:
    if log_metric is None:
        return
    log_metric(
        "verification/changed_label_rate",
        float(np.mean([row["changed_label_rate"] for row in logs])),
    )
    log_metric(
        "verification/zero_variance_group",
        float(np.mean([row["zero_variance_group"] for row in logs])),
    )
