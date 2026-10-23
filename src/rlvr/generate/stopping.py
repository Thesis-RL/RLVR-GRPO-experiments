"""Generation controls shared by offline sampling and online GRPO."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from transformers import (
    PreTrainedTokenizerBase,
    StoppingCriteria,
    StoppingCriteriaList,
    StopStringCriteria,
)


def done_stopping_criteria(
    tokenizer: PreTrainedTokenizerBase,
    stop_sequence: str | None,
) -> StopStringCriteria | None:
    """Build a tokenizer-aware criterion for a textual completion marker.

    Matching decoded text rather than one fixed token-id suffix is important:
    BPE tokenization of ``[DONE]`` changes when it follows whitespace.
    """
    if stop_sequence is None:
        return None
    stop_sequence = str(stop_sequence)
    if not stop_sequence:
        raise ValueError("stop_sequence must not be empty")
    return StopStringCriteria(tokenizer=tokenizer, stop_strings=[stop_sequence])


def append_stopping_criteria(
    existing: Any,
    criterion: StoppingCriteria | None,
) -> StoppingCriteriaList | Any:
    """Return a fresh criteria list without mutating caller-owned state."""
    if criterion is None:
        return existing
    criteria = StoppingCriteriaList(list(existing) if existing is not None else [])
    criteria.append(criterion)
    return criteria


@contextmanager
def inject_generate_stopping_criteria(
    model: Any, criterion: StoppingCriteria | None
) -> Iterator[None]:
    """Temporarily inject a stopping criterion into one model's ``generate`` calls."""
    if criterion is None:
        yield
        return

    original_generate = model.generate

    def generate_with_criterion(*args: Any, **kwargs: Any) -> Any:
        kwargs["stopping_criteria"] = append_stopping_criteria(
            kwargs.get("stopping_criteria"),
            criterion,
        )
        return original_generate(*args, **kwargs)

    model.generate = generate_with_criterion
    try:
        yield
    finally:
        model.generate = original_generate
