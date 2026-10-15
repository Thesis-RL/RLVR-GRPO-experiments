"""Official MBPP splits (Austin et al., 2021)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from datasets import Dataset, load_dataset

SplitName = Literal["train", "validation", "test", "prompt"]


@dataclass(frozen=True)
class MBPPProblem:
    task_id: int
    text: str
    code: str
    test_list: list[str]
    test_setup_code: str
    challenge_test_list: list[str]
    split: SplitName


def load_mbpp(
    split: SplitName | None = None,
    config: str = "full",
    *,
    local_dir: str | None = None,
) -> list[MBPPProblem]:
    """Load MBPP. Prefers local arrow files under data/mbpp/ when present."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]  # repo root
    local = Path(local_dir) if local_dir else root / "data" / "mbpp"
    if (local / "mbpp-train.arrow").exists():
        return _from_local_arrows(local, split)

    ds = load_dataset("google-research-datasets/mbpp", config)
    if split is None:
        problems: list[MBPPProblem] = []
        for name in ("train", "validation", "test", "prompt"):
            problems.extend(_from_split(ds[name], name))  # type: ignore[arg-type]
        return problems
    return _from_split(ds[split], split)


def _from_local_arrows(local: "Path", split: SplitName | None) -> list[MBPPProblem]:
    from datasets import Dataset

    mapping = {
        "train": "mbpp-train.arrow",
        "validation": "mbpp-validation.arrow",
        "test": "mbpp-test.arrow",
        "prompt": "mbpp-prompt.arrow",
    }
    names = list(mapping) if split is None else [split]
    problems: list[MBPPProblem] = []
    for name in names:
        problems.extend(_from_split(Dataset.from_file(str(local / mapping[name])), name))  # type: ignore[arg-type]
    return problems


def _from_split(split_ds: Dataset, split: SplitName) -> list[MBPPProblem]:
    out: list[MBPPProblem] = []
    for row in split_ds:
        out.append(
            MBPPProblem(
                task_id=int(row["task_id"]),
                text=str(row["text"]),
                code=str(row["code"]),
                test_list=list(row["test_list"]),
                test_setup_code=str(row.get("test_setup_code") or ""),
                challenge_test_list=list(row.get("challenge_test_list") or []),
                split=split,
            )
        )
    return out
