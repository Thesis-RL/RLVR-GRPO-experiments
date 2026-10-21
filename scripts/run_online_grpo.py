#!/usr/bin/env python3
"""Run one configured MBPP condition with online LoRA GRPO.

Use ``--dry-run`` to validate configuration, execution, corruption, rewards,
and logging without loading a language model. Use ``--smoke`` for the short
GPU end-to-end pilot. Main runs are blocked until the protocol is frozen.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rlvr.config import ConfigError, load_experiment_config  # noqa: E402
from rlvr.data.mbpp import load_mbpp, resolve_mbpp_split  # noqa: E402
from rlvr.online.callbacks import (  # noqa: E402
    JsonlTrainerCallback,
    ScheduledCleanEvalCallback,
)
from rlvr.online.data import (  # noqa: E402
    build_mbpp_dataset,
    load_difficulty_map,
    load_few_shot_examples,
)
from rlvr.online.evaluation import (  # noqa: E402
    evaluate_clean_pass_at_1,
    evaluation_steps,
    fraction_steps,
    summarize_primary_metric,
)
from rlvr.online.lifecycle import prepare_run_directory  # noqa: E402
from rlvr.online.reward import MBPPNoisyReward  # noqa: E402
from rlvr.online.settings import (  # noqa: E402
    EvaluationSettings,
    resolve_evaluation_settings,
    resolve_logging_settings,
    resolve_training_settings,
)
from rlvr.online.telemetry import RunTelemetry  # noqa: E402
from rlvr.run_logging import (  # noqa: E402
    RunContext,
    append_jsonl,
    collect_environment_metadata,
    slugify,
    update_run_metadata,
    write_run_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/priority_bridge.yaml")
    parser.add_argument("--condition", help="Condition name, e.g. fp_only__fractional")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", help="Optional model ID override for a pilot")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--limit", type=int, help="Limit training problems")
    parser.add_argument("--max-steps", type=int, help="Override profile max_steps")
    parser.add_argument("--eval-limit", type=int, help="Limit clean evaluation problems")
    parser.add_argument("--timeout-s", type=float, help="Per-test execution timeout")
    parser.add_argument("--mem-mb", type=int, help="Per-completion executor memory cap")
    parser.add_argument("--smoke", action="store_true", help="Use the provisional smoke profile")
    parser.add_argument("--dry-run", action="store_true", help="Exercise rewards without a model")
    parser.add_argument("--skip-clean-eval", action="store_true")
    parser.add_argument("--list-conditions", action="store_true")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        help="Resume from the latest or a specified checkpoint",
    )
    parser.add_argument(
        "--use-vllm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the configured vLLM setting",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = load_experiment_config(args.config)
    if args.list_conditions:
        for condition in experiment.conditions:
            print(condition.name)
        return
    if not args.condition:
        raise SystemExit("--condition is required (or use --list-conditions)")
    if args.dry_run and args.smoke:
        raise ConfigError("--dry-run and --smoke are mutually exclusive")
    if args.dry_run and args.resume:
        raise ConfigError("A dry run cannot be resumed")

    profile = "dry-run" if args.dry_run else "smoke" if args.smoke else "main"
    condition = experiment.condition(args.condition)
    seed = experiment.require_seed(args.seed)
    model = experiment.model
    if args.model or args.dtype:
        model = replace(model, id=args.model or model.id, dtype=args.dtype or model.dtype)

    settings = resolve_training_settings(
        experiment.values,
        smoke=args.smoke or args.dry_run,
        max_steps=args.max_steps,
        use_vllm=args.use_vllm,
    )
    if args.dtype:
        settings = replace(settings, precision=args.dtype)
    if settings.precision != model.dtype:
        raise ConfigError(
            "model.dtype and training.precision must match for reproducible mixed precision"
        )
    evaluation = resolve_evaluation_settings(experiment.values)
    logging = resolve_logging_settings(experiment.values)
    _validate_cli_for_profile(args, profile=profile)
    _validate_single_gpu(profile=profile)

    if settings.use_vllm and importlib.util.find_spec("vllm") is None:
        raise ConfigError(
            "vLLM was requested but is not installed. Install it in the cloud GPU environment "
            "or run with --no-use-vllm."
        )

    data_config = _mapping(experiment.values, "data")
    if str(data_config.get("dataset")) != "mbpp":
        raise ConfigError("Only data.dataset: mbpp is supported")
    generation = _mapping(experiment.values, "generation")
    _validate_generation(generation, settings.group_size)
    few_shot = load_few_shot_examples([int(value) for value in data_config["few_shot_task_ids"]])

    try:
        train_split = resolve_mbpp_split(data_config["train_split"])
        resolve_mbpp_split(data_config["validation_split"])
        resolve_mbpp_split(data_config["test_split"])
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    train_problems = load_mbpp(train_split)
    train_difficulty_path = (
        data_config.get("train_difficulty_path") if model.id == experiment.model.id else None
    )
    train_difficulty = load_difficulty_map(
        _repo_path(train_difficulty_path),
        required_task_ids=[problem.task_id for problem in train_problems]
        if train_difficulty_path is not None
        else None,
    )
    train_limit = args.limit
    if args.smoke and train_limit is None:
        train_limit = 2
    train_dataset = build_mbpp_dataset(
        train_split,
        few_shot=few_shot,
        limit=train_limit,
        difficulty_by_task=train_difficulty,
    )

    configured_revision_value = experiment.values["model"].get("revision")
    configured_revision = (
        str(configured_revision_value).strip() if configured_revision_value is not None else None
    )
    revision = configured_revision if model.id == experiment.model.id else None
    out_dir = args.out_dir or _default_out_dir(
        logging_root=ROOT / logging.output_root,
        experiment_name=experiment.name,
        profile=profile,
        model_id=model.id,
        model_revision=revision,
        condition_name=condition.name,
        seed=seed,
        dry_run=args.dry_run,
    )
    context = RunContext(
        pipeline="online_grpo",
        experiment=experiment.name,
        study=experiment.study,
        model=model.id,
        model_role=model.role,
        condition=condition.name,
        reward=condition.reward,
        noise=condition.noise.name,
        seed=seed,
        profile=profile,
        model_revision=revision,
    )
    timeout_s = float(args.timeout_s if args.timeout_s is not None else evaluation.timeout_s)
    mem_mb = int(args.mem_mb if args.mem_mb is not None else evaluation.mem_mb)
    if timeout_s <= 0 or mem_mb <= 0:
        raise ConfigError("timeout-s and mem-mb must be positive")
    if profile == "main" and settings.save_total_limit < evaluation.primary_metric_last_n:
        raise ConfigError("training.main.save_total_limit must retain the primary evaluations")
    if profile == "main":
        if len(fraction_steps(settings.max_steps, list(evaluation.test_fractions))) < (
            evaluation.primary_metric_last_n
        ):
            raise ConfigError(
                "The main step budget collapses scheduled test evaluations onto too few steps"
            )
        _validate_main_test_difficulty(
            data_config=data_config,
            model_id=model.id,
            model_revision=revision,
            expected_group_size=int(generation["group_size"]),
        )

    resolved = deepcopy(experiment.values)
    resolved["resolved_run"] = {
        "condition": condition.name,
        "seed": seed,
        "profile": profile,
        "model": model.id,
        "model_revision": revision,
        "dtype": model.dtype,
        "training": settings.as_dict(),
        "evaluation": asdict(evaluation),
        "logging": asdict(logging),
        "train_limit": train_limit,
        "eval_limit": args.eval_limit,
        "dry_run": args.dry_run,
    }
    environment = collect_environment_metadata(ROOT)
    git_state = environment.get("git", {})
    if profile == "main" and (not git_state.get("commit") or git_state.get("dirty")):
        raise ConfigError("Main runs require a clean Git worktree at a recorded commit")
    resume_checkpoint = prepare_run_directory(
        out_dir,
        context=context,
        resolved_config=resolved,
        resume=args.resume,
        current_git_commit=git_state.get("commit"),
    )
    if resume_checkpoint is None:
        write_run_manifest(
            out_dir,
            context=context,
            config=resolved,
            config_path=args.config,
            overrides=_non_null_cli_overrides(args),
            environment=environment,
        )

    rank = int(os.environ.get("RANK", "0"))
    rollout_name = "rollouts.jsonl" if rank == 0 else f"rollouts_rank-{rank}.jsonl"
    reward = MBPPNoisyReward(
        condition=condition,
        seed=seed,
        group_size=2 if args.dry_run else settings.group_size,
        context=context,
        rollout_log_path=(out_dir / rollout_name) if logging.save_rollouts else None,
        timeout_s=timeout_s,
        mem_limit=mem_mb * 1024 * 1024,
        save_clean_outcomes=logging.save_clean_outcomes,
        save_noisy_outcomes=logging.save_noisy_outcomes,
        save_both_rewards=logging.save_both_rewards,
        save_advantages=logging.save_advantages,
    )

    if args.dry_run:
        result = _run_reward_dry_run(reward, train_dataset, split=train_split)
        print(json.dumps(result, indent=2))
        return

    _run_training(
        args=args,
        experiment=experiment,
        model=model,
        condition_reward=reward,
        settings=settings,
        train_dataset=train_dataset,
        few_shot=few_shot,
        context=context,
        out_dir=out_dir,
        generation=generation,
        evaluation=evaluation,
        data_config=data_config,
        timeout_s=timeout_s,
        mem_mb=mem_mb,
        rank=rank,
        resume_checkpoint=resume_checkpoint,
    )


def _run_training(
    *,
    args: argparse.Namespace,
    experiment: Any,
    model: Any,
    condition_reward: MBPPNoisyReward,
    settings: Any,
    train_dataset: Any,
    few_shot: list[Any],
    context: RunContext,
    out_dir: Path,
    generation: dict[str, Any],
    evaluation: EvaluationSettings,
    data_config: dict[str, Any],
    timeout_s: float,
    mem_mb: int,
    rank: int,
    resume_checkpoint: str | bool | None,
) -> None:
    from peft import LoraConfig, TaskType
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    telemetry = RunTelemetry(out_dir)
    trainer: Any | None = None
    train_metrics: dict[str, Any] = {}
    telemetry.reset_gpu_peak()
    try:
        revision = context.model_revision
        with telemetry.stage("tokenizer_load"):
            tokenizer = AutoTokenizer.from_pretrained(
                model.id,
                revision=revision,
                trust_remote_code=True,
            )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        actual_revision = (
            revision
            or tokenizer.init_kwargs.get("_commit_hash")
            or getattr(tokenizer, "_commit_hash", None)
        )
        runtime_context = replace(context, model_revision=actual_revision)
        condition_reward.context = runtime_context
        if rank == 0:
            update_run_metadata(out_dir, {"resolved_model_revision": actual_revision})

        trainer_args = GRPOConfig(
            output_dir=str(out_dir),
            run_name=runtime_context.run_id,
            seed=runtime_context.seed,
            data_seed=runtime_context.seed,
            learning_rate=settings.learning_rate,
            per_device_train_batch_size=settings.per_device_train_batch_size,
            gradient_accumulation_steps=settings.gradient_accumulation_steps,
            max_steps=settings.max_steps,
            optim=settings.optimizer,
            lr_scheduler_type=settings.lr_scheduler_type,
            warmup_steps=settings.warmup_steps,
            weight_decay=settings.weight_decay,
            max_grad_norm=settings.max_grad_norm,
            adam_beta1=settings.adam_beta1,
            adam_beta2=settings.adam_beta2,
            adam_epsilon=settings.adam_epsilon,
            gradient_checkpointing=settings.gradient_checkpointing,
            bf16=settings.precision == "bfloat16",
            fp16=settings.precision == "float16",
            model_init_kwargs={
                "dtype": model.dtype,
                "revision": actual_revision,
                "trust_remote_code": True,
            },
            num_generations=settings.group_size,
            generation_batch_size=settings.group_size,
            max_completion_length=int(generation["max_new_tokens"]),
            temperature=float(generation["train_temperature"]),
            top_p=float(generation["top_p"]),
            beta=settings.kl_coefficient,
            num_iterations=settings.num_iterations,
            scale_rewards=settings.scale_rewards,
            loss_type=settings.loss_type,
            use_vllm=settings.use_vllm,
            logging_steps=settings.logging_steps,
            logging_first_step=True,
            log_completions=True,
            save_strategy="steps",
            save_steps=settings.save_steps,
            save_total_limit=settings.save_total_limit,
            save_only_model=False,
            report_to="none",
            dataloader_num_workers=0,
            remove_unused_columns=False,
        )
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=settings.lora_rank,
            lora_alpha=settings.lora_alpha,
            lora_dropout=settings.lora_dropout,
            target_modules=list(settings.lora_target_modules),
            bias="none",
        )
        event_path = out_dir / "events.jsonl"
        callbacks: list[Any] = [JsonlTrainerCallback(event_path, runtime_context)]

        if settings.profile == "main":
            callbacks.append(
                _scheduled_evaluation_callback(
                    settings=settings,
                    tokenizer=tokenizer,
                    few_shot=few_shot,
                    context=runtime_context,
                    out_dir=out_dir,
                    generation=generation,
                    evaluation=evaluation,
                    data_config=data_config,
                    timeout_s=timeout_s,
                    mem_mb=mem_mb,
                    event_path=event_path,
                    telemetry=telemetry,
                )
            )

        class InstrumentedGRPOTrainer(GRPOTrainer):
            generation_and_scoring_seconds = 0.0

            def _generate_and_score_completions(self, *call_args, **call_kwargs):
                started = time.perf_counter()
                try:
                    return super()._generate_and_score_completions(*call_args, **call_kwargs)
                finally:
                    self.generation_and_scoring_seconds += time.perf_counter() - started

        with telemetry.stage("trainer_initialization"):
            trainer = InstrumentedGRPOTrainer(
                model=model.id,
                reward_funcs=condition_reward,
                args=trainer_args,
                train_dataset=train_dataset,
                processing_class=tokenizer,
                peft_config=lora,
                callbacks=callbacks,
            )
        with telemetry.stage("training"):
            train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        train_metrics = dict(train_result.metrics)

        with telemetry.stage("final_save"):
            trainer.save_model(out_dir / "final_adapter")
            trainer.save_state()
            trainer.accelerator.wait_for_everyone()

        primary_summary = None
        if not args.skip_clean_eval and rank == 0:
            if settings.profile == "smoke":
                test_split = resolve_mbpp_split(data_config["test_split"])
                problems = load_mbpp(test_split)
                eval_limit = args.eval_limit if args.eval_limit is not None else 2
                problems = problems[:eval_limit]
                difficulty_path = (
                    data_config.get("test_difficulty_path")
                    if model.id == experiment.model.id
                    else None
                )
                test_difficulty = load_difficulty_map(_repo_path(difficulty_path))
                unwrapped = trainer.accelerator.unwrap_model(trainer.model)
                with telemetry.stage("evaluation_test"):
                    evaluate_clean_pass_at_1(
                        unwrapped,
                        tokenizer,
                        problems,
                        few_shot=few_shot,
                        context=runtime_context,
                        out_dir=out_dir,
                        split_name="test",
                        checkpoint="final_adapter",
                        step=int(trainer.state.global_step),
                        seed=evaluation.seed,
                        temperature=float(generation["eval_temperature"]),
                        top_p=float(generation["top_p"]),
                        max_new_tokens=int(generation["max_new_tokens"]),
                        timeout_s=timeout_s,
                        mem_limit=mem_mb * 1024 * 1024,
                        difficulty_by_task=test_difficulty,
                    )
                primary_summary = summarize_primary_metric(
                    out_dir,
                    split_name="test",
                    metric=evaluation.primary_metric,
                    last_n=1,
                    require_count=True,
                )
            else:
                primary_summary = summarize_primary_metric(
                    out_dir,
                    split_name="test",
                    metric=evaluation.primary_metric,
                    last_n=evaluation.primary_metric_last_n,
                    require_count=True,
                )

        resource_report = telemetry.write(
            status="complete",
            reward_seconds=condition_reward.reward_seconds,
            grading_seconds=condition_reward.grading_seconds,
            generation_and_scoring_seconds=float(
                getattr(trainer, "generation_and_scoring_seconds", 0.0)
            ),
            generated_tokens=condition_reward.generated_tokens,
            train_metrics=train_metrics,
        )
        append_jsonl(
            out_dir / "events.jsonl",
            {"event": "run_complete", "step": int(trainer.state.global_step)},
            context=runtime_context,
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "primary_metric": primary_summary,
                    "resource_report": resource_report,
                },
                indent=2,
            )
        )
    except Exception as exc:
        telemetry.write(
            status="failed",
            reward_seconds=condition_reward.reward_seconds,
            grading_seconds=condition_reward.grading_seconds,
            generation_and_scoring_seconds=float(
                getattr(trainer, "generation_and_scoring_seconds", 0.0)
                if trainer is not None
                else 0.0
            ),
            generated_tokens=condition_reward.generated_tokens,
            train_metrics=train_metrics,
            error=f"{type(exc).__name__}: {exc}",
        )
        append_jsonl(
            out_dir / "events.jsonl",
            {"event": "run_failed", "error": f"{type(exc).__name__}: {exc}"},
            context=condition_reward.context,
        )
        raise


def _scheduled_evaluation_callback(
    *,
    settings: Any,
    tokenizer: Any,
    few_shot: list[Any],
    context: RunContext,
    out_dir: Path,
    generation: dict[str, Any],
    evaluation: EvaluationSettings,
    data_config: dict[str, Any],
    timeout_s: float,
    mem_mb: int,
    event_path: Path,
    telemetry: RunTelemetry,
) -> ScheduledCleanEvalCallback:
    validation_split = resolve_mbpp_split(data_config["validation_split"])
    test_split = resolve_mbpp_split(data_config["test_split"])
    validation_problems = load_mbpp(validation_split)
    test_problems = load_mbpp(test_split)
    test_difficulty_path = _repo_path(data_config.get("test_difficulty_path"))
    if test_difficulty_path is None:
        raise ConfigError("data.test_difficulty_path must be frozen before a main run")
    test_difficulty = load_difficulty_map(
        test_difficulty_path,
        required_task_ids=[problem.task_id for problem in test_problems],
        expected_model=context.model,
        expected_revision=context.model_revision,
        expected_split=test_split,
        expected_group_size=int(generation["group_size"]),
        require_metadata=True,
    )
    validation_schedule = set(
        evaluation_steps(settings.max_steps, evaluation.validation_interval_fraction)
    )
    test_schedule = set(fraction_steps(settings.max_steps, list(evaluation.test_fractions)))

    def run_evaluation(split_name: str, model: Any, step: int, checkpoint: str) -> dict[str, Any]:
        problems = validation_problems if split_name == "validation" else test_problems
        difficulties = {} if split_name == "validation" else test_difficulty
        with telemetry.stage(f"evaluation_{split_name}"):
            return evaluate_clean_pass_at_1(
                model,
                tokenizer,
                problems,
                few_shot=few_shot,
                context=context,
                out_dir=out_dir,
                split_name=split_name,
                checkpoint=checkpoint,
                step=step,
                seed=evaluation.seed,
                temperature=float(generation["eval_temperature"]),
                top_p=float(generation["top_p"]),
                max_new_tokens=int(generation["max_new_tokens"]),
                timeout_s=timeout_s,
                mem_limit=mem_mb * 1024 * 1024,
                difficulty_by_task=difficulties,
            )

    return ScheduledCleanEvalCallback(
        validation_steps=validation_schedule,
        test_steps=test_schedule,
        evaluate=run_evaluation,
        event_path=event_path,
        context=context,
    )


def _run_reward_dry_run(
    reward: MBPPNoisyReward,
    train_dataset: Any,
    *,
    split: str,
) -> dict[str, Any]:
    problems = {problem.task_id: problem for problem in load_mbpp(split)}
    rows = [train_dataset[index] for index in range(min(2, len(train_dataset)))]
    prompts: list[str] = []
    completions: list[str] = []
    task_ids: list[int] = []
    tests: list[list[str]] = []
    setup: list[str] = []
    difficulty: list[str] = []

    for row in rows:
        problem = problems[int(row["task_id"])]
        prompts.extend([str(row["prompt"])] * reward.group_size)
        completions.extend([problem.code, *([""] * (reward.group_size - 1))])
        task_ids.extend([problem.task_id] * reward.group_size)
        tests.extend([problem.test_list] * reward.group_size)
        setup.extend([problem.test_setup_code] * reward.group_size)
        difficulty.extend([str(row["difficulty"])] * reward.group_size)

    rewards = reward(
        prompts=prompts,
        completions=completions,
        task_id=task_ids,
        test_list=tests,
        test_setup_code=setup,
        completion_ids=[[] for _ in completions],
        difficulty=difficulty,
        trainer_state=SimpleNamespace(global_step=0),
    )
    return {
        "status": "dry_run_ok",
        "condition": reward.condition.name,
        "n_completions": len(completions),
        "rewards": rewards,
        "rollout_log": str(reward.rollout_log_path) if reward.rollout_log_path else None,
    }


def _default_out_dir(
    *,
    logging_root: Path,
    experiment_name: str,
    profile: str,
    model_id: str,
    model_revision: str | None,
    condition_name: str,
    seed: int,
    dry_run: bool,
) -> Path:
    kind = "dry_run" if dry_run else "online_grpo"
    model_slug = slugify(model_id.replace("/", "__"))
    if model_revision:
        model_slug = f"{model_slug}__rev-{slugify(model_revision[:12])}"
    return (
        logging_root
        / kind
        / experiment_name
        / profile
        / model_slug
        / condition_name
        / f"seed-{seed}"
    )


def _repo_path(value: Any) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def _mapping(values: dict[str, Any], key: str) -> dict[str, Any]:
    value = values.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"{key} must be a mapping")
    return value


def _validate_generation(generation: dict[str, Any], group_size: int) -> None:
    if group_size < 2:
        raise ConfigError("GRPO group_size must be at least 2")
    if float(generation.get("train_temperature", 0)) <= 0:
        raise ConfigError("generation.train_temperature must be positive")
    if float(generation.get("eval_temperature", 0)) <= 0:
        raise ConfigError("generation.eval_temperature must be positive")
    if not 0.0 < float(generation.get("top_p", 0)) <= 1.0:
        raise ConfigError("generation.top_p must be in (0, 1]")
    if int(generation.get("max_new_tokens", 0)) <= 0:
        raise ConfigError("generation.max_new_tokens must be positive")


def _validate_cli_for_profile(args: argparse.Namespace, *, profile: str) -> None:
    if profile != "main":
        return
    forbidden = {
        "--model": args.model,
        "--dtype": args.dtype,
        "--limit": args.limit,
        "--max-steps": args.max_steps,
        "--eval-limit": args.eval_limit,
        "--timeout-s": args.timeout_s,
        "--mem-mb": args.mem_mb,
        "--skip-clean-eval": args.skip_clean_eval,
    }
    used = [name for name, value in forbidden.items() if value is not None and value is not False]
    if args.use_vllm is not None:
        used.append("--use-vllm/--no-use-vllm")
    if used:
        raise ConfigError(
            "Main runs must use only the frozen YAML configuration; remove: " + ", ".join(used)
        )


def _validate_single_gpu(*, profile: str) -> None:
    if profile == "dry-run":
        return
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 1:
        raise ConfigError("The current reward logger supports exactly one GPU/process per run")


def _validate_main_test_difficulty(
    *,
    data_config: dict[str, Any],
    model_id: str,
    model_revision: str | None,
    expected_group_size: int,
) -> None:
    test_split = resolve_mbpp_split(data_config["test_split"])
    problems = load_mbpp(test_split)
    path = _repo_path(data_config.get("test_difficulty_path"))
    if path is None:
        raise ConfigError("data.test_difficulty_path must be frozen before a main run")
    try:
        load_difficulty_map(
            path,
            required_task_ids=[problem.task_id for problem in problems],
            expected_model=model_id,
            expected_revision=model_revision,
            expected_split=test_split,
            expected_group_size=expected_group_size,
            require_metadata=True,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc


def _non_null_cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if value is not None and value is not False
    }


if __name__ == "__main__":
    try:
        main()
    except ConfigError as exc:
        raise SystemExit(f"configuration error: {exc}") from exc
