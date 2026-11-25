#!/usr/bin/env python3
"""Evaluate one base model or LoRA checkpoint with multiple samples per MBPP task."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rlvr.config import load_experiment_config  # noqa: E402
from rlvr.data.mbpp import load_mbpp, resolve_mbpp_split  # noqa: E402
from rlvr.evaluation.multisample import (  # noqa: E402
    MultiSampleTarget,
    evaluate_multisample_clean,
)
from rlvr.generate.sample import load_model  # noqa: E402
from rlvr.online.data import load_difficulty_map, load_few_shot_examples  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/priority_bridge.yaml")
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--kind", choices=("base", "adapter"), required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--checkpoint-label")
    parser.add_argument("--condition")
    parser.add_argument("--step", type=int)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--samples-per-task", type=int, required=True)
    parser.add_argument("--generation-chunk-size", type=int, default=4)
    parser.add_argument("--generation-chunk-offset", type=int, default=0)
    parser.add_argument("--generation-chunk-stride", type=int, default=1)
    parser.add_argument("--generation-seed", type=int)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_261_116)
    parser.add_argument("--difficulty-path", type=Path)
    parser.add_argument("--split")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = load_experiment_config(args.config)
    values = experiment.values
    model_values = values["model"]
    data_values = values["data"]
    generation_values = values["generation"]
    evaluation_values = values["evaluation"]
    model_revision = str(model_values.get("revision") or "").strip()
    if not model_revision:
        raise SystemExit("model.revision must be pinned for multi-sample evaluation")
    if args.kind == "adapter":
        if args.checkpoint_path is None or args.condition is None or args.step is None:
            raise SystemExit("adapter target requires checkpoint path, condition, and step")
        experiment.condition(args.condition)
        _validate_checkpoint(
            args.checkpoint_path,
            condition=args.condition,
            step=args.step,
            model=str(model_values["id"]),
            model_revision=model_revision,
        )
    elif any(
        value is not None
        for value in (args.checkpoint_path, args.condition, args.step, args.checkpoint_label)
    ):
        raise SystemExit("base target cannot have checkpoint, condition, or step arguments")

    split = resolve_mbpp_split(args.split or data_values["test_split"])
    problems = load_mbpp(split)
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be positive")
        problems = problems[: args.limit]
    few_shot = load_few_shot_examples([int(value) for value in data_values["few_shot_task_ids"]])
    difficulty_path = args.difficulty_path or ROOT / str(data_values["test_difficulty_path"])
    difficulty = load_difficulty_map(
        difficulty_path,
        required_task_ids=[problem.task_id for problem in problems],
        expected_model=str(model_values["id"]),
        expected_revision=model_revision,
        expected_split=split,
        expected_group_size=int(generation_values["group_size"]),
        expected_max_new_tokens=int(generation_values["max_new_tokens"]),
        expected_stop_sequence=str(generation_values["stop_sequence"]),
        require_metadata=True,
    )

    _validate_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_label = args.checkpoint_label
    if args.kind == "adapter" and checkpoint_label is None:
        checkpoint_label = f"{args.condition}/checkpoint-{args.step}"
    target = MultiSampleTarget(
        target_id=args.target_id,
        kind=args.kind,
        model=str(model_values["id"]),
        model_revision=model_revision,
        checkpoint=checkpoint_label,
        condition=args.condition,
        step=args.step,
    )
    _write_json_atomic(
        args.out_dir / "runtime_metadata.json",
        _runtime_metadata(args=args, target=target),
    )

    progress = _Progress(args.out_dir / "status.json", target_id=args.target_id)
    progress.update(0, len(problems) * args.samples_per_task, stage="model_load")
    model, tokenizer = load_model(
        str(model_values["id"]),
        dtype=str(model_values["dtype"]),
        device=args.device,
        revision=model_revision,
    )
    if args.kind == "adapter":
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            str(args.checkpoint_path),
            is_trainable=False,
        )
        model.eval()

    try:
        summary = evaluate_multisample_clean(
            model,
            tokenizer,
            problems,
            few_shot=few_shot,
            target=target,
            out_dir=args.out_dir,
            samples_per_task=args.samples_per_task,
            generation_seed=(
                int(evaluation_values["seed"])
                if args.generation_seed is None
                else args.generation_seed
            ),
            generation_chunk_size=args.generation_chunk_size,
            generation_chunk_offset=args.generation_chunk_offset,
            generation_chunk_stride=args.generation_chunk_stride,
            temperature=float(generation_values["eval_temperature"]),
            top_p=float(generation_values["top_p"]),
            max_new_tokens=int(generation_values["max_new_tokens"]),
            stop_sequence=str(generation_values["stop_sequence"]),
            timeout_s=float(evaluation_values["timeout_s"]),
            timeout_fallback_s=float(evaluation_values["timeout_fallback_s"]),
            mem_limit=int(evaluation_values["mem_mb"]) * 1024 * 1024,
            difficulty_by_task=difficulty,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
            progress_callback=progress,
        )
    except BaseException as exc:
        progress.fail(exc)
        raise
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    progress.complete(summary)
    print(json.dumps(summary, indent=2))


def _validate_device(device: str) -> None:
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
        if torch.cuda.device_count() != 1:
            raise SystemExit(
                "Exactly one GPU must be visible inside the evaluation container; "
                f"found {torch.cuda.device_count()}"
            )
        if not torch.cuda.is_bf16_supported():
            raise SystemExit("The selected GPU does not report bfloat16 support")


def _validate_checkpoint(
    checkpoint: Path,
    *,
    condition: str,
    step: int,
    model: str,
    model_revision: str,
) -> None:
    required = (checkpoint / "adapter_config.json", checkpoint / "adapter_model.safetensors")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"checkpoint is missing required files: {missing}")
    adapter = json.loads((checkpoint / "adapter_config.json").read_text())
    expected_adapter = {
        "base_model_name_or_path": model,
        "peft_type": "LORA",
        "r": 16,
        "lora_alpha": 32,
    }
    mismatches = {
        key: (adapter.get(key), expected)
        for key, expected in expected_adapter.items()
        if adapter.get(key) != expected
    }
    target_modules = adapter.get("target_modules")
    expected_target_modules = ["q_proj", "v_proj"]
    if not isinstance(target_modules, list) or sorted(target_modules) != expected_target_modules:
        mismatches["target_modules"] = (target_modules, expected_target_modules)
    if mismatches:
        raise SystemExit(f"adapter metadata mismatch in {checkpoint}: {mismatches}")

    run_metadata_path = checkpoint.parent / "run_metadata.json"
    if run_metadata_path.is_file():
        context = json.loads(run_metadata_path.read_text()).get("context", {})
        expected_context = {
            "condition": condition,
            "seed": 0,
            "profile": "main",
            "model": model,
            "model_revision": model_revision,
        }
        context_mismatches = {
            key: (context.get(key), expected)
            for key, expected in expected_context.items()
            if context.get(key) != expected
        }
        if context_mismatches:
            raise SystemExit(
                f"checkpoint run metadata mismatch in {run_metadata_path}: {context_mismatches}"
            )
    if checkpoint.name != f"checkpoint-{step}":
        raise SystemExit(f"checkpoint path {checkpoint} does not match requested step {step}")


def _runtime_metadata(*, args: argparse.Namespace, target: MultiSampleTarget) -> dict[str, Any]:
    packages = {}
    for name in ("torch", "transformers", "peft", "numpy", "datasets"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cuda: dict[str, Any] = {
        "available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        cuda["devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ]
    checkpoint_sha = None
    if args.checkpoint_path is not None:
        checkpoint_sha = _sha256(args.checkpoint_path / "adapter_model.safetensors")
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "target": target.__dict__,
        "checkpoint_path": str(args.checkpoint_path) if args.checkpoint_path else None,
        "adapter_model_sha256": checkpoint_sha,
        "packages": packages,
        "cuda": cuda,
        "environment": {
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }


class _Progress:
    def __init__(self, path: Path, *, target_id: str) -> None:
        self.path = path
        self.target_id = target_id
        self.started = time.monotonic()
        self.initial_completed: int | None = None

    def __call__(self, completed: int, total: int) -> None:
        self.update(completed, total, stage="evaluation")

    def update(self, completed: int, total: int, *, stage: str) -> None:
        if self.initial_completed is None and stage == "evaluation":
            self.initial_completed = completed
        elapsed = time.monotonic() - self.started
        new_completed = (
            max(0, completed - self.initial_completed) if self.initial_completed is not None else 0
        )
        rate = new_completed / elapsed if elapsed > 0 and new_completed > 0 else None
        remaining = total - completed
        eta = remaining / rate if rate else None
        _write_json_atomic(
            self.path,
            {
                "schema_version": 1,
                "status": "running",
                "stage": stage,
                "target_id": self.target_id,
                "completed": completed,
                "total": total,
                "progress_fraction": completed / total if total else 0.0,
                "elapsed_seconds": elapsed,
                "generations_per_second": rate,
                "estimated_seconds_remaining": eta,
                "updated_at_utc": datetime.now(UTC).isoformat(),
            },
        )

    def complete(self, summary: dict[str, Any]) -> None:
        _write_json_atomic(
            self.path,
            {
                "schema_version": 1,
                "status": "complete",
                "stage": "complete",
                "target_id": self.target_id,
                "completed": summary["n_generations"],
                "total": summary["n_generations"],
                "progress_fraction": 1.0,
                "elapsed_seconds": time.monotonic() - self.started,
                "estimated_seconds_remaining": 0.0,
                "updated_at_utc": datetime.now(UTC).isoformat(),
                "pass_at_1": summary["pass_at_1"],
            },
        )

    def fail(self, exc: BaseException) -> None:
        _write_json_atomic(
            self.path,
            {
                "schema_version": 1,
                "status": "failed",
                "stage": "failed",
                "target_id": self.target_id,
                "elapsed_seconds": time.monotonic() - self.started,
                "updated_at_utc": datetime.now(UTC).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


if __name__ == "__main__":
    main()
