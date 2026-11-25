#!/usr/bin/env python3
"""Audit clean/noisy LoRA policy-gradient alignment on frozen rollout groups."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rlvr.config import load_experiment_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-config",
        type=Path,
        default=ROOT / "configs/gradient_audit_v1_1.yaml",
    )
    parser.add_argument("--matrix-root", type=Path)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--phases", nargs="+")
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = _load_yaml(args.audit_config.resolve())
    experiment_path = _resolve_config_path(args.audit_config.resolve(), audit["source_config"])
    experiment = load_experiment_config(experiment_path)
    matrix_root = (args.matrix_root or Path(audit["matrix_root"])).resolve()
    out_root = (args.out_root or Path(audit["output_root"])).resolve()
    targets = build_targets(
        audit=audit,
        experiment=experiment,
        matrix_root=matrix_root,
        conditions=args.conditions,
        seeds=args.seeds,
        phases=args.phases,
    )
    if args.max_targets is not None:
        if args.max_targets <= 0:
            raise SystemExit("--max-targets must be positive")
        targets = targets[: args.max_targets]

    protocol = _protocol(
        audit=audit,
        audit_config=args.audit_config.resolve(),
        experiment_path=experiment_path,
        experiment=experiment,
        matrix_root=matrix_root,
        targets=targets,
    )
    fingerprint = _fingerprint(protocol)
    preflight = preflight_targets(targets, protocol_fingerprint=fingerprint)
    if args.preflight_only:
        out_root.mkdir(parents=True, exist_ok=True)
        report = {
            "schema_version": 1,
            "kind": "gradient_audit_preflight",
            "status": "passed",
            "protocol_fingerprint": fingerprint,
            **preflight,
        }
        _write_json_atomic(out_root / "preflight.json", report)
        print(json.dumps(report, indent=2))
        return

    out_root.mkdir(parents=True, exist_ok=True)
    _prepare_output(out_root, protocol, fingerprint, resume=args.resume)
    _write_status(
        out_root,
        status="running",
        fingerprint=fingerprint,
        completed=_completed_target_count(out_root, fingerprint),
        total=len(targets),
    )
    try:
        result = run_audit(
            audit=audit,
            experiment=experiment,
            targets=targets,
            out_root=out_root,
            protocol_fingerprint=fingerprint,
        )
        write_aggregate_reports(out_root, result)
        _write_status(
            out_root,
            status="complete",
            fingerprint=fingerprint,
            completed=len(targets),
            total=len(targets),
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "out_root": str(out_root),
                    "target_count": len(targets),
                    "result_rows": len(result["targets"]),
                },
                indent=2,
            )
        )
    except BaseException as exc:
        _write_status(
            out_root,
            status="failed",
            fingerprint=fingerprint,
            completed=_completed_target_count(out_root, fingerprint),
            total=len(targets),
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


def build_targets(
    *,
    audit: dict[str, Any],
    experiment: Any,
    matrix_root: Path,
    conditions: list[str] | None = None,
    seeds: list[int] | None = None,
    phases: list[str] | None = None,
) -> list[dict[str, Any]]:
    selected_conditions = conditions or [str(value) for value in audit["conditions"]]
    selected_seeds = seeds or [int(value) for value in audit["seeds"]]
    selected_phases = phases or [str(value) for value in audit["phases"]]
    configured_phases = {str(name): int(step) for name, step in audit["phases"].items()}
    allowed_seeds = {int(value) for value in audit["seeds"]}
    targets = []
    for condition in selected_conditions:
        condition_config = experiment.condition(condition)
        if condition not in audit["conditions"]:
            raise ValueError(f"condition {condition!r} is outside the frozen audit")
        for seed in selected_seeds:
            if seed not in allowed_seeds:
                raise ValueError(f"seed {seed} is outside the frozen audit")
            for phase in selected_phases:
                if phase not in configured_phases:
                    raise ValueError(f"phase {phase!r} is outside the frozen audit")
                step = configured_phases[phase]
                run_dir = matrix_root / f"seed-{seed}" / condition
                targets.append(
                    {
                        "target_id": f"seed-{seed}__{condition}__{phase}-step-{step}",
                        "seed": seed,
                        "condition": condition,
                        "noise": condition_config.noise.name,
                        "train_aggregation": condition_config.reward,
                        "phase": phase,
                        "step": step,
                        "run_dir": run_dir,
                        "checkpoint": run_dir / f"checkpoint-{step}",
                        "rollouts": run_dir / "rollouts.jsonl",
                    }
                )
    return targets


def preflight_targets(
    targets: list[dict[str, Any]], *, protocol_fingerprint: str
) -> dict[str, Any]:
    if not targets:
        raise ValueError("the gradient audit target list is empty")
    by_run: dict[Path, dict[int, list[dict[str, Any]]]] = {}
    rows_per_target: dict[str, int] = {}
    for target in targets:
        checkpoint = target["checkpoint"]
        for name in ("adapter_config.json", "adapter_model.safetensors", "tokenizer.json"):
            path = checkpoint / name
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"missing checkpoint artifact: {path}")
        run_dir = target["run_dir"]
        if run_dir not in by_run:
            by_run[run_dir] = _read_selected_groups(
                target["rollouts"],
                {item["step"] for item in targets if item["run_dir"] == run_dir},
            )
        group = by_run[run_dir].get(target["step"], [])
        _validate_group(group, target)
        rows_per_target[target["target_id"]] = len(group)
    return {
        "target_count": len(targets),
        "run_count": len({target["run_dir"] for target in targets}),
        "checkpoint_count": len({target["checkpoint"] for target in targets}),
        "rollout_rows_per_target": rows_per_target,
        "protocol_fingerprint": protocol_fingerprint,
    }


def run_audit(
    *,
    audit: dict[str, Any],
    experiment: Any,
    targets: list[dict[str, Any]],
    out_root: Path,
    protocol_fingerprint: str,
) -> dict[str, Any]:
    import peft
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("gradient audit requires one CUDA device")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"expected exactly one visible CUDA device, got {torch.cuda.device_count()}"
        )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    memory_fraction = float(os.environ.get("RLVR_GPU_MEMORY_FRACTION", "0.35"))
    if not 0 < memory_fraction <= 1:
        raise ValueError("RLVR_GPU_MEMORY_FRACTION must be in (0, 1]")
    torch.cuda.set_per_process_memory_fraction(memory_fraction, device=0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    first_checkpoint = targets[0]["checkpoint"]
    tokenizer = AutoTokenizer.from_pretrained(first_checkpoint, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_config = experiment.model
    revision = str(experiment.values["model"]["revision"])
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[model_config.dtype]
    base = AutoModelForCausalLM.from_pretrained(
        model_config.id,
        revision=revision,
        dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
        device_map={"": device},
    )
    base.config.use_cache = False
    model: Any | None = None
    group_cache: dict[Path, dict[int, list[dict[str, Any]]]] = {}

    package_versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
    }
    started = time.perf_counter()
    for index, target in enumerate(targets, start=1):
        target_path = _target_result_path(out_root, target)
        existing = _read_valid_target(target_path, protocol_fingerprint)
        if existing is not None:
            print(f"[{index}/{len(targets)}] reuse {target['target_id']}", flush=True)
            continue

        run_dir = target["run_dir"]
        if run_dir not in group_cache:
            group_cache[run_dir] = _read_selected_groups(
                target["rollouts"],
                {item["step"] for item in targets if item["run_dir"] == run_dir},
            )
        group = group_cache[run_dir][target["step"]]
        _validate_group(group, target)
        adapter_name = f"audit_{index}"
        if model is None:
            model = PeftModel.from_pretrained(
                base,
                target["checkpoint"],
                adapter_name=adapter_name,
                is_trainable=True,
            )
        else:
            model.load_adapter(
                target["checkpoint"],
                adapter_name=adapter_name,
                is_trainable=True,
            )
        model.set_adapter(adapter_name)
        model.eval()
        trainable = _trainable_parameter_metadata(model)
        if not trainable["names"] or any("lora_" not in name for name in trainable["names"]):
            raise RuntimeError("gradient audit exposed non-LoRA trainable parameters")

        print(
            f"[{index}/{len(targets)}] audit {target['target_id']} task={group[0]['task_id']}",
            flush=True,
        )
        target_started = time.perf_counter()
        vectors, token_info = _gradient_vectors(
            model=model,
            tokenizer=tokenizer,
            rows=group,
            device=device,
        )
        aggregations = {}
        for aggregation in audit["aggregations"]:
            clean_advantages = np.asarray(
                [float(row[f"clean_{aggregation}_advantage"]) for row in group],
                dtype=np.float64,
            )
            noisy_advantages = np.asarray(
                [float(row[f"noisy_{aggregation}_advantage"]) for row in group],
                dtype=np.float64,
            )
            aggregations[str(aggregation)] = {
                "actual_training_aggregation": aggregation == target["train_aggregation"],
                "advantage": _advantage_metrics(clean_advantages, noisy_advantages),
                "gradient": _vector_metrics(
                    vectors[f"clean_{aggregation}"], vectors[f"noisy_{aggregation}"]
                ),
                "clip_rate": 0.0,
            }

        result = {
            "schema_version": 1,
            "kind": "lora_gradient_audit_target",
            "protocol_fingerprint": protocol_fingerprint,
            "target_id": target["target_id"],
            "seed": target["seed"],
            "condition": target["condition"],
            "noise": target["noise"],
            "train_aggregation": target["train_aggregation"],
            "phase": target["phase"],
            "step": target["step"],
            "task_id": int(group[0]["task_id"]),
            "difficulty": str(group[0]["difficulty"]),
            "group_id": str(group[0]["group_id"]),
            "checkpoint": str(target["checkpoint"]),
            "adapter_sha256": _sha256(target["checkpoint"] / "adapter_model.safetensors"),
            "completion_count": len(group),
            "token_reconstruction": token_info,
            "trainable_lora_parameters": trainable["count"],
            "trainable_lora_tensors": len(trainable["names"]),
            "aggregations": aggregations,
            "duration_seconds": time.perf_counter() - target_started,
            "package_versions": package_versions,
        }
        _write_json_atomic(target_path, result)
        _write_status(
            out_root,
            status="running",
            fingerprint=protocol_fingerprint,
            completed=_completed_target_count(out_root, protocol_fingerprint),
            total=len(targets),
            elapsed_seconds=time.perf_counter() - started,
            current_target=target["target_id"],
        )
        torch.cuda.empty_cache()

    rows = _load_all_target_results(out_root, targets, protocol_fingerprint)
    return {
        "schema_version": 1,
        "kind": "lora_gradient_audit",
        "protocol_fingerprint": protocol_fingerprint,
        "package_versions": package_versions,
        "targets": rows,
        "summary": summarize_targets(rows),
    }


def _gradient_vectors(
    *,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    device: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    import torch.nn.functional as functional

    prompt = str(rows[0]["prompt"])
    if any(str(row["prompt"]) != prompt for row in rows):
        raise ValueError("all completions in an audited group must share one prompt")
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    completion_ids, reconstruction = _reconstruct_completion_ids(tokenizer, rows)
    total_tokens = sum(len(ids) for ids in completion_ids)
    if total_tokens <= 0:
        raise ValueError("audited group has no completion tokens")

    keys = ("clean_binary", "noisy_binary", "clean_fractional", "noisy_fractional")
    advantages = {key: [float(row[f"{key}_advantage"]) for row in rows] for key in keys}
    trainable_parameters = [
        parameter for _, parameter in model.named_parameters() if parameter.requires_grad
    ]
    vector_size = sum(parameter.numel() for parameter in trainable_parameters)
    vectors = {key: torch.zeros(vector_size, dtype=torch.float32) for key in keys}
    nonzero_basis_count = 0
    for completion_index, ids in enumerate(completion_ids):
        weights = {key: -advantages[key][completion_index] / total_tokens for key in keys}
        if all(abs(value) < 1e-15 for value in weights.values()):
            continue
        model.zero_grad(set_to_none=True)
        input_ids = torch.tensor([prompt_ids + ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.autocast(device_type="cuda", dtype=next(model.parameters()).dtype):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=len(ids) + 1,
            )
            completion_logits = outputs.logits[:, -(len(ids) + 1) : -1, :]
            completion_targets = input_ids[:, -len(ids) :]
            negative_log_likelihood = functional.cross_entropy(
                completion_logits.float().reshape(-1, completion_logits.size(-1)),
                completion_targets.reshape(-1),
                reduction="sum",
            )
            log_probability_sum = -negative_log_likelihood
        log_probability_sum.backward()
        basis_parts = []
        for parameter in trainable_parameters:
            if parameter.grad is None:
                basis_parts.append(torch.zeros(parameter.numel(), dtype=torch.float32))
            else:
                basis_parts.append(parameter.grad.detach().float().cpu().reshape(-1))
        basis = torch.cat(basis_parts)
        if not torch.isfinite(basis).all():
            raise FloatingPointError("non-finite LoRA gradient basis")
        nonzero_basis_count += int(float(torch.linalg.vector_norm(basis)) > 0)
        for key, weight in weights.items():
            if abs(weight) >= 1e-15:
                vectors[key].add_(basis, alpha=weight)
        del (
            basis,
            basis_parts,
            input_ids,
            attention_mask,
            outputs,
            completion_logits,
            completion_targets,
            negative_log_likelihood,
            log_probability_sum,
        )

    return vectors, {
        "prompt_tokens": len(prompt_ids),
        "completion_tokens_total": total_tokens,
        "completion_tokens_min": min(map(len, completion_ids)),
        "completion_tokens_max": max(map(len, completion_ids)),
        **reconstruction,
        "nonzero_basis_completions": nonzero_basis_count,
    }


def _reconstruct_completion_ids(
    tokenizer: Any, rows: list[dict[str, Any]]
) -> tuple[list[list[int]], dict[str, int]]:
    completion_ids = []
    appended_eos = 0
    unreconstructed = 0
    logged_total = 0
    canonical_visible_total = 0
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer has no EOS token for completion reconstruction")
    for row in rows:
        ids = tokenizer.encode(str(row["completion"]), add_special_tokens=False)
        expected = int(row["completion_token_count"])
        hidden_count = expected - len(ids)
        if hidden_count < 0:
            raise ValueError(
                f"canonical completion is longer than the logged token count for "
                f"{row['group_id']} index {row['completion_index']}: "
                f"encoded={len(ids)}, logged={expected}"
            )
        canonical_visible_total += len(ids)
        logged_total += expected
        if hidden_count >= 1:
            ids = [*ids, int(tokenizer.eos_token_id)]
            appended_eos += 1
            unreconstructed += hidden_count - 1
        completion_ids.append(ids)
    return completion_ids, {
        "logged_completion_tokens_total": logged_total,
        "canonical_visible_tokens_total": canonical_visible_total,
        "eos_tokens_reconstructed": appended_eos,
        "unreconstructed_noncanonical_tokens": unreconstructed,
    }


def _advantage_metrics(clean: np.ndarray, noisy: np.ndarray) -> dict[str, Any]:
    clean_norm = float(np.linalg.norm(clean))
    noisy_norm = float(np.linalg.norm(noisy))
    cosine = None
    if clean_norm > 0 and noisy_norm > 0:
        cosine = float(np.dot(clean, noisy) / (clean_norm * noisy_norm))
    mask = (np.abs(clean) > 1e-12) | (np.abs(noisy) > 1e-12)
    sign_agreement = None
    if mask.any():
        sign_agreement = float(np.mean(np.sign(clean[mask]) == np.sign(noisy[mask])))
    return {
        "clean_norm": clean_norm,
        "noisy_norm": noisy_norm,
        "cosine": cosine,
        "sign_agreement": sign_agreement,
        "clean_zero": clean_norm <= 1e-12,
        "noisy_zero": noisy_norm <= 1e-12,
    }


def _vector_metrics(clean: Any, noisy: Any) -> dict[str, Any]:
    import torch

    clean64 = clean.to(torch.float64)
    noisy64 = noisy.to(torch.float64)
    clean_norm = float(torch.linalg.vector_norm(clean64))
    noisy_norm = float(torch.linalg.vector_norm(noisy64))
    dot = float(torch.dot(clean64, noisy64))
    cosine = None
    if clean_norm > 0 and noisy_norm > 0:
        cosine = max(-1.0, min(1.0, dot / (clean_norm * noisy_norm)))
    return {
        "clean_norm": clean_norm,
        "noisy_norm": noisy_norm,
        "dot_product": dot,
        "cosine": cosine,
        "norm_ratio_noisy_to_clean": noisy_norm / clean_norm if clean_norm > 0 else None,
        "negative_alignment": cosine < 0 if cosine is not None else None,
        "clean_zero": clean_norm <= 1e-12,
        "noisy_zero": noisy_norm <= 1e-12,
    }


def summarize_targets(targets: list[dict[str, Any]]) -> dict[str, Any]:
    flat = []
    for target in targets:
        for aggregation, values in target["aggregations"].items():
            flat.append(
                {
                    "condition": target["condition"],
                    "seed": target["seed"],
                    "phase": target["phase"],
                    "step": target["step"],
                    "aggregation": aggregation,
                    "actual_training_aggregation": values["actual_training_aggregation"],
                    "advantage_cosine": values["advantage"]["cosine"],
                    "advantage_sign_agreement": values["advantage"]["sign_agreement"],
                    "gradient_cosine": values["gradient"]["cosine"],
                    "gradient_dot_product": values["gradient"]["dot_product"],
                    "gradient_clean_norm": values["gradient"]["clean_norm"],
                    "gradient_noisy_norm": values["gradient"]["noisy_norm"],
                    "gradient_norm_ratio": values["gradient"]["norm_ratio_noisy_to_clean"],
                    "negative_alignment": values["gradient"]["negative_alignment"],
                    "clean_zero_gradient": values["gradient"]["clean_zero"],
                    "noisy_zero_gradient": values["gradient"]["noisy_zero"],
                    "clip_rate": values["clip_rate"],
                }
            )

    groupings: dict[str, Any] = {}
    for scope, selected in (
        ("actual", [row for row in flat if row["actual_training_aggregation"]]),
        ("counterfactual", [row for row in flat if not row["actual_training_aggregation"]]),
    ):
        groupings[scope] = {
            "by_condition": _group_summaries(selected, ("condition",)),
            "by_condition_phase": _group_summaries(selected, ("condition", "phase")),
            "by_condition_seed": _group_summaries(selected, ("condition", "seed")),
        }
    return {
        "rows": flat,
        **groupings,
        "paired_fractional_minus_binary": _paired_aggregation_summaries(targets),
    }


def _paired_aggregation_summaries(targets: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for target in targets:
        binary = target["aggregations"]["binary"]
        fractional = target["aggregations"]["fractional"]
        binary_gradient = binary["gradient"]["cosine"]
        fractional_gradient = fractional["gradient"]["cosine"]
        binary_advantage = binary["advantage"]["cosine"]
        fractional_advantage = fractional["advantage"]["cosine"]
        rows.append(
            {
                "noise": target["noise"],
                "condition": target["condition"],
                "seed": target["seed"],
                "phase": target["phase"],
                "gradient_delta": (
                    float(fractional_gradient) - float(binary_gradient)
                    if binary_gradient is not None and fractional_gradient is not None
                    else None
                ),
                "advantage_delta": (
                    float(fractional_advantage) - float(binary_advantage)
                    if binary_advantage is not None and fractional_advantage is not None
                    else None
                ),
            }
        )
    return {
        "rows": rows,
        "by_noise": _paired_group_summaries(rows, ("noise",)),
        "by_condition": _paired_group_summaries(rows, ("condition",)),
        "by_noise_phase": _paired_group_summaries(rows, ("noise", "phase")),
    }


def _paired_group_summaries(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    result = {}
    for identity, values in grouped.items():
        label = "__".join(str(value) for value in identity)
        gradient = [
            float(row["gradient_delta"]) for row in values if row["gradient_delta"] is not None
        ]
        advantage = [
            float(row["advantage_delta"]) for row in values if row["advantage_delta"] is not None
        ]
        result[label] = {
            **{key: value for key, value in zip(keys, identity, strict=True)},
            "target_count": len(values),
            "paired_gradient_count": len(gradient),
            "mean_gradient_cosine_delta": _mean_or_none(gradient),
            "median_gradient_cosine_delta": _median_or_none(gradient),
            "fraction_fractional_gradient_higher": _mean_or_none(
                [float(value > 1e-12) for value in gradient]
            ),
            "mean_advantage_cosine_delta": _mean_or_none(advantage),
        }
    return result


def _group_summaries(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    result = {}
    for identity, values in grouped.items():
        label = "__".join(str(value) for value in identity)
        cosines = [
            float(row["gradient_cosine"]) for row in values if row["gradient_cosine"] is not None
        ]
        ratios = [
            float(row["gradient_norm_ratio"])
            for row in values
            if row["gradient_norm_ratio"] is not None
        ]
        dots = [float(row["gradient_dot_product"]) for row in values]
        advantage_cosines = [
            float(row["advantage_cosine"]) for row in values if row["advantage_cosine"] is not None
        ]
        result[label] = {
            **{key: value for key, value in zip(keys, identity, strict=True)},
            "target_count": len(values),
            "defined_gradient_cosine_count": len(cosines),
            "mean_gradient_cosine": _mean_or_none(cosines),
            "median_gradient_cosine": _median_or_none(cosines),
            "mean_gradient_dot_product": _mean_or_none(dots),
            "mean_gradient_norm_ratio": _mean_or_none(ratios),
            "negative_alignment_rate_defined": _mean_or_none(
                [
                    float(row["negative_alignment"])
                    for row in values
                    if row["negative_alignment"] is not None
                ]
            ),
            "clean_zero_gradient_rate": float(
                np.mean([row["clean_zero_gradient"] for row in values])
            ),
            "noisy_zero_gradient_rate": float(
                np.mean([row["noisy_zero_gradient"] for row in values])
            ),
            "mean_advantage_cosine": _mean_or_none(advantage_cosines),
            "clip_rate": float(np.mean([row["clip_rate"] for row in values])),
        }
    return result


def write_aggregate_reports(out_root: Path, result: dict[str, Any]) -> None:
    _write_json_atomic(out_root / "analysis.json", result)
    rows = result["summary"]["rows"]
    _write_csv(out_root / "target_metrics.csv", rows)
    (out_root / "analysis.md").write_text(_markdown(result))


def _markdown(result: dict[str, Any]) -> str:
    actual = result["summary"]["actual"]["by_condition"]
    condition_order = (
        "clean__binary",
        "clean__fractional",
        "fp_only__binary",
        "fp_only__fractional",
        "fn_only__binary",
        "fn_only__fractional",
    )
    conditions = [condition for condition in condition_order if condition in actual]
    lines = [
        "# LoRA gradient-direction audit",
        "",
        "Frozen groups at checkpoints 112 (early), 561 (middle), and 1010 (late); "
        "three seeds; no optimizer step.",
        "",
        "## Actual training aggregation",
        "",
        "| Condition | Defined / total | Mean gradient cosine | Negative alignment | Noisy zero-gradient | Mean norm ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in conditions:
        values = actual[condition]
        lines.append(
            f"| `{condition}` | {values['defined_gradient_cosine_count']} / {values['target_count']} | "
            f"{_fmt(values['mean_gradient_cosine'])} | "
            f"{_pct(values['negative_alignment_rate_defined'])} | "
            f"{_pct(values['noisy_zero_gradient_rate'])} | "
            f"{_fmt(values['mean_gradient_norm_ratio'])} |"
        )
    lines.extend(
        [
            "",
            "## Actual aggregation by phase",
            "",
            "| Condition | Phase | Mean cosine | Negative alignment | Noisy zero-gradient |",
            "|---|---|---:|---:|---:|",
        ]
    )
    phased = result["summary"]["actual"]["by_condition_phase"]
    for condition in conditions:
        for phase in ("early", "middle", "late"):
            key = f"{condition}__{phase}"
            if key not in phased:
                continue
            values = phased[key]
            lines.append(
                f"| `{condition}` | {phase} | {_fmt(values['mean_gradient_cosine'])} | "
                f"{_pct(values['negative_alignment_rate_defined'])} | "
                f"{_pct(values['noisy_zero_gradient_rate'])} |"
            )
    lines.extend(
        [
            "",
            "## Paired aggregation contrast",
            "",
            "Fractional minus binary cosine on the same checkpoint and frozen group.",
            "",
            "| Noise | Defined / total | Mean gradient-cosine delta | Median delta | Fraction fractional higher |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    paired = result["summary"]["paired_fractional_minus_binary"]["by_noise"]
    for noise in ("clean", "fp_only", "fn_only"):
        if noise not in paired:
            continue
        values = paired[noise]
        lines.append(
            f"| `{noise}` | {values['paired_gradient_count']} / {values['target_count']} | "
            f"{_fmt(values['mean_gradient_cosine_delta'])} | "
            f"{_fmt(values['median_gradient_cosine_delta'])} | "
            f"{_pct(values['fraction_fractional_gradient_higher'])} |"
        )
    lines.extend(
        [
            "",
            "Cosines with a zero clean or noisy gradient are undefined and excluded from the "
            "cosine/negative-alignment denominator; zero-gradient rates are reported separately. "
            "The clipping rate is zero by the frozen on-policy ratio=1 audit definition.",
            "",
        ]
    )
    return "\n".join(lines)


def _protocol(
    *,
    audit: dict[str, Any],
    audit_config: Path,
    experiment_path: Path,
    experiment: Any,
    matrix_root: Path,
    targets: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "lora_gradient_audit_protocol",
        "audit_config": str(audit_config),
        "audit_config_sha256": _sha256(audit_config),
        "source_config": str(experiment_path),
        "source_config_sha256": _sha256(experiment_path),
        "experiment": experiment.name,
        "model": experiment.model.id,
        "model_revision": str(experiment.values["model"]["revision"]),
        "matrix_root": str(matrix_root),
        "seeds": sorted({target["seed"] for target in targets}),
        "conditions": list(dict.fromkeys(target["condition"] for target in targets)),
        "phases": {target["phase"]: target["step"] for target in targets},
        "aggregations": list(audit["aggregations"]),
        "target_ids": [target["target_id"] for target in targets],
        "objective": audit["loss"],
        "dropout_mode": audit["dropout_mode"],
        "gradient_subspace": "trainable LoRA parameters only",
        "gradient_vectors_persisted": False,
        "completion_source": "immutable saved rollout groups",
        "token_reconstruction": audit["token_reconstruction"],
    }


def _prepare_output(
    out_root: Path, protocol: dict[str, Any], fingerprint: str, *, resume: bool
) -> None:
    manifest_path = out_root / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("protocol_fingerprint") != fingerprint:
            raise ValueError("existing gradient-audit output has a different protocol")
        if not resume:
            raise ValueError("gradient-audit output already exists; pass --resume")
        return
    manifest = {
        **protocol,
        "protocol_fingerprint": fingerprint,
        "created_at_utc": _now(),
    }
    _write_json_atomic(manifest_path, manifest)


def _read_selected_groups(path: Path, steps: set[int]) -> dict[int, list[dict[str, Any]]]:
    if not path.is_file():
        raise ValueError(f"missing rollout log: {path}")
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            step = int(row["policy_step"])
            if step in steps:
                result[step].append(row)
    return dict(result)


def _validate_group(rows: list[dict[str, Any]], target: dict[str, Any]) -> None:
    if len(rows) != 16:
        raise ValueError(f"{target['target_id']} has {len(rows)} rollout rows, expected 16")
    expected_indices = set(range(16))
    if {int(row["completion_index"]) for row in rows} != expected_indices:
        raise ValueError(f"{target['target_id']} has invalid completion indices")
    if len({row["group_id"] for row in rows}) != 1:
        raise ValueError(f"{target['target_id']} contains multiple groups")
    if len({int(row["task_id"]) for row in rows}) != 1:
        raise ValueError(f"{target['target_id']} contains multiple tasks")
    for row in rows:
        if int(row["seed"]) != target["seed"]:
            raise ValueError(f"{target['target_id']} seed mismatch")
        if row["condition"] != target["condition"]:
            raise ValueError(f"{target['target_id']} condition mismatch")
        if int(row["policy_step"]) != target["step"]:
            raise ValueError(f"{target['target_id']} policy-step mismatch")
        for aggregation in ("binary", "fractional"):
            for prefix in ("clean", "noisy"):
                value = float(row[f"{prefix}_{aggregation}_advantage"])
                if not math.isfinite(value):
                    raise ValueError(f"{target['target_id']} contains non-finite advantages")
    rows.sort(key=lambda row: int(row["completion_index"]))


def _trainable_parameter_metadata(model: Any) -> dict[str, Any]:
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    count = sum(
        parameter.numel() for _, parameter in model.named_parameters() if parameter.requires_grad
    )
    return {"names": names, "count": count}


def _target_result_path(out_root: Path, target: dict[str, Any]) -> Path:
    return (
        out_root
        / "targets"
        / f"seed-{target['seed']}"
        / target["condition"]
        / f"step-{target['step']}.json"
    )


def _read_valid_target(path: Path, fingerprint: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    if value.get("protocol_fingerprint") != fingerprint:
        raise ValueError(f"target result has a different protocol: {path}")
    return value


def _load_all_target_results(
    out_root: Path, targets: list[dict[str, Any]], fingerprint: str
) -> list[dict[str, Any]]:
    rows = []
    for target in targets:
        value = _read_valid_target(_target_result_path(out_root, target), fingerprint)
        if value is None:
            raise ValueError(f"missing completed target: {target['target_id']}")
        rows.append(value)
    return rows


def _completed_target_count(out_root: Path, fingerprint: str) -> int:
    count = 0
    for path in (out_root / "targets").glob("seed-*/*/step-*.json"):
        try:
            count += int(json.loads(path.read_text()).get("protocol_fingerprint") == fingerprint)
        except (OSError, json.JSONDecodeError):
            continue
    return count


def _write_status(
    out_root: Path,
    *,
    status: str,
    fingerprint: str,
    completed: int,
    total: int,
    elapsed_seconds: float | None = None,
    current_target: str | None = None,
    error: str | None = None,
) -> None:
    _write_json_atomic(
        out_root / "status.json",
        {
            "schema_version": 1,
            "kind": "lora_gradient_audit_status",
            "status": status,
            "updated_at_utc": _now(),
            "protocol_fingerprint": fingerprint,
            "completed_targets": completed,
            "total_targets": total,
            "current_target": current_target,
            "elapsed_seconds": elapsed_seconds,
            "error": error,
        },
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping in {path}")
    return value


def _resolve_config_path(audit_config: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    candidate = (ROOT / path).resolve()
    if candidate.is_file():
        return candidate
    return (audit_config.parent / path).resolve()


def _fingerprint(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _median_or_none(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def _now() -> str:
    return datetime.now(UTC).isoformat()


if __name__ == "__main__":
    main()
