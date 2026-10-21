"""Trainer callbacks that persist optimisation and checkpoint events."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from rlvr.run_logging import RunContext, append_jsonl


class JsonlTrainerCallback(TrainerCallback):
    def __init__(self, path: str | Path, context: RunContext) -> None:
        self.path = Path(path)
        self.context = context

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001, ANN201
        if logs:
            append_jsonl(
                self.path,
                {"event": "metrics", "step": int(state.global_step), "metrics": dict(logs)},
                context=self.context,
            )
        return control

    def on_save(self, args, state, control, **kwargs: Any):  # noqa: ANN001, ANN201
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        append_jsonl(
            self.path,
            {
                "event": "checkpoint",
                "step": int(state.global_step),
                "checkpoint": str(checkpoint),
            },
            context=self.context,
        )
        return control


class ScheduledCleanEvalCallback(TrainerCallback):
    """Force checkpoints at protocol steps and evaluate their clean policy."""

    def __init__(
        self,
        *,
        validation_steps: set[int],
        test_steps: set[int],
        evaluate: Callable[[str, Any, int, str], dict[str, Any]],
        event_path: str | Path,
        context: RunContext,
    ) -> None:
        self.validation_steps = validation_steps
        self.test_steps = test_steps
        self.evaluate = evaluate
        self.event_path = Path(event_path)
        self.context = context

    @property
    def scheduled_steps(self) -> set[int]:
        return self.validation_steps | self.test_steps

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001, ANN201, ARG002
        if int(state.global_step) in self.scheduled_steps:
            control.should_save = True
        return control

    def on_save(self, args, state, control, model=None, **kwargs):  # noqa: ANN001, ANN201, ARG002
        step = int(state.global_step)
        if not state.is_world_process_zero or step not in self.scheduled_steps:
            return control
        if model is None:
            raise RuntimeError(
                "Trainer did not provide a model to the scheduled evaluation callback"
            )

        checkpoint = str(Path(args.output_dir) / f"checkpoint-{step}")
        for split_name, steps in (
            ("validation", self.validation_steps),
            ("test", self.test_steps),
        ):
            if step not in steps:
                continue
            summary = self.evaluate(split_name, model, step, checkpoint)
            append_jsonl(
                self.event_path,
                {
                    "event": "clean_evaluation_summary",
                    "step": step,
                    "checkpoint": checkpoint,
                    "split": split_name,
                    "summary": summary,
                },
                context=self.context,
            )
        return control
