"""Dataset preparation for online MBPP training and clean evaluation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from datasets import Dataset

from rlvr.data.mbpp import MBPPProblem, SplitName, load_mbpp
from rlvr.prompts.mbpp import build_prompt


def load_few_shot_examples(task_ids: Sequence[int]) -> list[MBPPProblem]:
    prompt_problems = {problem.task_id: problem for problem in load_mbpp("prompt")}
    examples = [prompt_problems[task_id] for task_id in task_ids if task_id in prompt_problems]
    if len(examples) < len(task_ids):
        return list(prompt_problems.values())[:3]
    return examples


def build_mbpp_dataset(
    split: SplitName,
    *,
    few_shot: list[MBPPProblem],
    limit: int | None = None,
    difficulty_by_task: dict[int, str] | None = None,
) -> Dataset:
    """Build the standard-format dataset expected by TRL's GRPOTrainer."""
    problems = load_mbpp(split)
    if limit is not None:
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        problems = problems[:limit]

    rows = [
        {
            "prompt": build_prompt(problem, few_shot),
            "task_id": problem.task_id,
            "test_list": problem.test_list,
            "test_setup_code": problem.test_setup_code,
            "difficulty": (difficulty_by_task or {}).get(problem.task_id, "unknown"),
        }
        for problem in problems
    ]
    return Dataset.from_list(rows)


def load_difficulty_map(
    path: str | Path | None,
    *,
    required_task_ids: Sequence[int] | None = None,
    expected_model: str | None = None,
    expected_revision: str | None = None,
    expected_split: str | None = None,
    expected_group_size: int | None = None,
    require_metadata: bool = False,
    allow_missing: bool = False,
) -> dict[int, str]:
    """Load the frozen offline difficulty strata used by later analyses."""
    if path is None:
        return {}
    source = Path(path)
    if not source.exists():
        if allow_missing:
            return {}
        raise FileNotFoundError(f"Difficulty file does not exist: {source}")
    raw = json.loads(source.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Difficulty file must contain a mapping: {source}")

    result: dict[int, str] = {}
    for task_id, value in raw.items():
        stratum = value.get("stratum") if isinstance(value, dict) else value
        if stratum not in {"hard", "medium", "easy"}:
            raise ValueError(f"Invalid difficulty for task {task_id}: {stratum!r}")
        result[int(task_id)] = str(stratum)

    metadata_path = source.with_name("difficulty_metadata.json")
    if require_metadata and not metadata_path.exists():
        raise ValueError(f"Difficulty metadata does not exist: {metadata_path}")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if require_metadata and metadata.get("schema_version") != 1:
            raise ValueError(f"Unsupported difficulty metadata schema in {metadata_path}")
        expected = {
            "model": expected_model,
            "model_revision": expected_revision,
            "split": expected_split,
            "group_size": expected_group_size,
        }
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if value is not None and metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Difficulty metadata mismatch in {metadata_path}: {mismatches}")

    if required_task_ids is not None:
        missing = sorted(set(int(task_id) for task_id in required_task_ids) - result.keys())
        if missing:
            preview = ", ".join(str(task_id) for task_id in missing[:5])
            raise ValueError(f"Difficulty file is missing {len(missing)} task IDs: {preview}")
    return result
