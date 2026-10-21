"""Validated training settings for smoke and frozen main runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from rlvr.config import ConfigError


@dataclass(frozen=True)
class TrainingSettings:
    profile: str
    group_size: int
    learning_rate: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    kl_coefficient: float
    logging_steps: int
    save_steps: int
    save_total_limit: int
    optimizer: str
    lr_scheduler_type: str
    warmup_steps: int
    weight_decay: float
    max_grad_norm: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    precision: str
    gradient_checkpointing: bool
    use_vllm: bool
    scale_rewards: str
    loss_type: str
    num_iterations: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationSettings:
    clean_only: bool
    primary_metric: str
    primary_metric_last_n: int
    validation_interval_fraction: float
    test_fractions: tuple[float, ...]
    seed: int
    timeout_s: float
    mem_mb: int


@dataclass(frozen=True)
class LoggingSettings:
    output_root: str
    save_rollouts: bool
    save_clean_outcomes: bool
    save_noisy_outcomes: bool
    save_both_rewards: bool
    save_advantages: bool


def resolve_training_settings(
    values: Mapping[str, Any],
    *,
    smoke: bool,
    max_steps: int | None = None,
    use_vllm: bool | None = None,
) -> TrainingSettings:
    training = _mapping(values, "training")
    generation = _mapping(values, "generation")
    profile_name = "smoke" if smoke else "main"
    profile = _mapping(training, profile_name)

    if not smoke and str(training.get("protocol_status")) != "frozen":
        raise ConfigError(
            "Main training is blocked while training.protocol_status is not 'frozen'. "
            "Run --smoke first, then freeze the pilot-selected hyperparameters."
        )
    if not smoke:
        model = _mapping(values, "model")
        if not str(model.get("revision") or "").strip():
            raise ConfigError("model.revision must be pinned before a main run")

    required = (
        "learning_rate",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "max_steps",
        "kl_coefficient",
        "logging_steps",
        "save_steps",
        "save_total_limit",
    )
    missing = [key for key in required if profile.get(key) is None]
    if missing:
        raise ConfigError(f"training.{profile_name} has unset fields: {', '.join(missing)}")

    lora = _mapping(training, "lora")
    target_modules = lora.get("target_modules")
    if not isinstance(target_modules, list) or not target_modules:
        raise ConfigError("training.lora.target_modules must be a non-empty list")

    configured_group_size = profile.get("group_size", generation.get("group_size"))
    settings = TrainingSettings(
        profile=profile_name,
        group_size=int(configured_group_size),
        learning_rate=float(profile["learning_rate"]),
        per_device_train_batch_size=int(profile["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(profile["gradient_accumulation_steps"]),
        max_steps=int(max_steps if max_steps is not None else profile["max_steps"]),
        kl_coefficient=float(profile["kl_coefficient"]),
        logging_steps=int(profile["logging_steps"]),
        save_steps=int(profile["save_steps"]),
        save_total_limit=int(profile["save_total_limit"]),
        optimizer=str(training["optimizer"]),
        lr_scheduler_type=str(training["lr_scheduler_type"]),
        warmup_steps=int(training["warmup_steps"]),
        weight_decay=float(training["weight_decay"]),
        max_grad_norm=float(training["max_grad_norm"]),
        adam_beta1=float(training["adam_beta1"]),
        adam_beta2=float(training["adam_beta2"]),
        adam_epsilon=float(training["adam_epsilon"]),
        precision=str(training["precision"]),
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
        use_vllm=bool(training["use_vllm"] if use_vllm is None else use_vllm),
        scale_rewards=str(training["scale_rewards"]),
        loss_type=str(training["loss_type"]),
        num_iterations=int(training["num_iterations"]),
        lora_rank=int(lora["rank"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        lora_target_modules=tuple(str(module) for module in target_modules),
    )
    _validate_positive(settings)
    return settings


def resolve_evaluation_settings(values: Mapping[str, Any]) -> EvaluationSettings:
    evaluation = _mapping(values, "evaluation")
    test_fractions_raw = evaluation.get("test_fractions")
    if not isinstance(test_fractions_raw, list):
        raise ConfigError("evaluation.test_fractions must be a list")
    settings = EvaluationSettings(
        clean_only=bool(evaluation.get("clean_only")),
        primary_metric=str(evaluation.get("primary_metric")),
        primary_metric_last_n=int(evaluation.get("primary_metric_last_n", 0)),
        validation_interval_fraction=float(evaluation.get("validation_interval_fraction", 0)),
        test_fractions=tuple(float(value) for value in test_fractions_raw),
        seed=int(evaluation.get("seed", 0)),
        timeout_s=float(evaluation.get("timeout_s", 0)),
        mem_mb=int(evaluation.get("mem_mb", 0)),
    )
    if not settings.clean_only:
        raise ConfigError("Only clean evaluation is supported")
    if settings.primary_metric != "pass_at_1":
        raise ConfigError("evaluation.primary_metric must be pass_at_1")
    if settings.primary_metric_last_n <= 0:
        raise ConfigError("evaluation.primary_metric_last_n must be positive")
    if settings.primary_metric_last_n > len(settings.test_fractions):
        raise ConfigError("primary_metric_last_n exceeds the number of test evaluations")
    if not 0.0 < settings.validation_interval_fraction <= 1.0:
        raise ConfigError("evaluation.validation_interval_fraction must be in (0, 1]")
    if not settings.test_fractions or any(
        not 0.0 < fraction <= 1.0 for fraction in settings.test_fractions
    ):
        raise ConfigError("evaluation.test_fractions must be non-empty values in (0, 1]")
    if settings.timeout_s <= 0 or settings.mem_mb <= 0:
        raise ConfigError("evaluation timeout_s and mem_mb must be positive")
    return settings


def resolve_logging_settings(values: Mapping[str, Any]) -> LoggingSettings:
    logging = _mapping(values, "logging")
    output_root = str(logging.get("output_root", "")).strip()
    if not output_root:
        raise ConfigError("logging.output_root must be non-empty")
    keys = (
        "save_rollouts",
        "save_clean_outcomes",
        "save_noisy_outcomes",
        "save_both_rewards",
        "save_advantages",
    )
    for key in keys:
        if not isinstance(logging.get(key), bool):
            raise ConfigError(f"logging.{key} must be a boolean")
    return LoggingSettings(
        output_root=output_root,
        save_rollouts=bool(logging["save_rollouts"]),
        save_clean_outcomes=bool(logging["save_clean_outcomes"]),
        save_noisy_outcomes=bool(logging["save_noisy_outcomes"]),
        save_both_rewards=bool(logging["save_both_rewards"]),
        save_advantages=bool(logging["save_advantages"]),
    )


def _mapping(values: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = values.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{key} must be a mapping")
    return value


def _validate_positive(settings: TrainingSettings) -> None:
    positive = {
        "group_size": settings.group_size,
        "learning_rate": settings.learning_rate,
        "per_device_train_batch_size": settings.per_device_train_batch_size,
        "gradient_accumulation_steps": settings.gradient_accumulation_steps,
        "max_steps": settings.max_steps,
        "logging_steps": settings.logging_steps,
        "save_steps": settings.save_steps,
        "save_total_limit": settings.save_total_limit,
        "lora_rank": settings.lora_rank,
        "lora_alpha": settings.lora_alpha,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ConfigError(f"Training settings must be positive: {', '.join(invalid)}")
    if settings.kl_coefficient < 0:
        raise ConfigError("kl_coefficient must be non-negative")
    if settings.warmup_steps < 0:
        raise ConfigError("training.warmup_steps must be non-negative")
    if settings.weight_decay < 0 or settings.max_grad_norm <= 0:
        raise ConfigError("weight_decay must be non-negative and max_grad_norm positive")
    if not 0.0 <= settings.adam_beta1 < 1.0 or not 0.0 <= settings.adam_beta2 < 1.0:
        raise ConfigError("Adam beta values must be in [0, 1)")
    if settings.adam_epsilon <= 0:
        raise ConfigError("training.adam_epsilon must be positive")
    if not settings.lr_scheduler_type.strip() or not settings.optimizer.strip():
        raise ConfigError("optimizer and lr_scheduler_type must be non-empty")
    if settings.precision not in {"bfloat16", "float16", "float32"}:
        raise ConfigError("training.precision must be bfloat16, float16, or float32")
    if not 0.0 <= settings.lora_dropout < 1.0:
        raise ConfigError("lora_dropout must be in [0, 1)")
    if settings.scale_rewards != "group":
        raise ConfigError(
            "training.scale_rewards must be group so logged advantages match the GRPO update"
        )
    if settings.loss_type not in {"grpo", "bnpo", "dr_grpo", "dapo", "cispo", "luspo", "vespo"}:
        raise ConfigError(f"Unsupported training.loss_type: {settings.loss_type}")
    if settings.num_iterations <= 0:
        raise ConfigError("training.num_iterations must be positive")
    if any(not module.strip() for module in settings.lora_target_modules):
        raise ConfigError("training.lora.target_modules must contain non-empty names")
    if settings.group_size % settings.per_device_train_batch_size != 0:
        raise ConfigError("group_size must be divisible by per_device_train_batch_size on one GPU")
    steps_per_generation = settings.group_size // settings.per_device_train_batch_size
    if settings.gradient_accumulation_steps % steps_per_generation != 0:
        raise ConfigError(
            "gradient_accumulation_steps must be divisible by steps_per_generation "
            "so policy generation and optimizer updates stay aligned"
        )
