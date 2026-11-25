from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.audit_lora_gradients import (
    _advantage_metrics,
    _markdown,
    _reconstruct_completion_ids,
    _vector_metrics,
    summarize_targets,
)


class FakeTokenizer:
    eos_token_id = 99

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [ord(character) for character in text]


def test_completion_reconstruction_restores_dropped_eos() -> None:
    rows = [
        {
            "completion": "ab",
            "completion_token_count": 3,
            "group_id": "group",
            "completion_index": 0,
        },
        {
            "completion": "c",
            "completion_token_count": 1,
            "group_id": "group",
            "completion_index": 1,
        },
    ]

    ids, reconstruction = _reconstruct_completion_ids(FakeTokenizer(), rows)

    assert ids == [[ord("a"), ord("b"), 99], [ord("c")]]
    assert reconstruction == {
        "logged_completion_tokens_total": 4,
        "canonical_visible_tokens_total": 3,
        "eos_tokens_reconstructed": 1,
        "unreconstructed_noncanonical_tokens": 0,
    }


def test_completion_reconstruction_reports_unrecoverable_subtoken() -> None:
    rows = [
        {
            "completion": "ab",
            "completion_token_count": 4,
            "group_id": "group",
            "completion_index": 0,
        }
    ]

    ids, reconstruction = _reconstruct_completion_ids(FakeTokenizer(), rows)

    assert ids == [[ord("a"), ord("b"), 99]]
    assert reconstruction["unreconstructed_noncanonical_tokens"] == 1


def test_advantage_and_gradient_alignment_metrics() -> None:
    clean_advantage = np.asarray([-1.0, 0.0, 1.0])
    noisy_advantage = -clean_advantage
    advantage = _advantage_metrics(clean_advantage, noisy_advantage)
    gradient = _vector_metrics(torch.tensor([1.0, 2.0]), torch.tensor([-1.0, -2.0]))

    assert advantage["cosine"] == pytest.approx(-1.0)
    assert advantage["sign_agreement"] == 0.0
    assert gradient["cosine"] == pytest.approx(-1.0)
    assert gradient["negative_alignment"] is True
    assert gradient["norm_ratio_noisy_to_clean"] == pytest.approx(1.0)


def test_summary_excludes_undefined_cosine_but_counts_zero_gradient() -> None:
    def target(condition: str, cosine: float | None, noisy_zero: bool) -> dict:
        return {
            "condition": condition,
            "noise": "fn_only",
            "seed": 0,
            "phase": "early",
            "step": 112,
            "aggregations": {
                "binary": {
                    "actual_training_aggregation": True,
                    "advantage": {"cosine": cosine, "sign_agreement": cosine},
                    "gradient": {
                        "cosine": cosine,
                        "dot_product": 0.0,
                        "clean_norm": 1.0,
                        "noisy_norm": 0.0 if noisy_zero else 1.0,
                        "norm_ratio_noisy_to_clean": 0.0 if noisy_zero else 1.0,
                        "negative_alignment": None if cosine is None else cosine < 0,
                        "clean_zero": False,
                        "noisy_zero": noisy_zero,
                    },
                    "clip_rate": 0.0,
                },
                "fractional": {
                    "actual_training_aggregation": False,
                    "advantage": {"cosine": 1.0, "sign_agreement": 1.0},
                    "gradient": {
                        "cosine": 1.0,
                        "dot_product": 1.0,
                        "clean_norm": 1.0,
                        "noisy_norm": 1.0,
                        "norm_ratio_noisy_to_clean": 1.0,
                        "negative_alignment": False,
                        "clean_zero": False,
                        "noisy_zero": False,
                    },
                    "clip_rate": 0.0,
                },
            },
        }

    summary = summarize_targets(
        [target("fn_only__binary", None, True), target("fn_only__binary", -0.5, False)]
    )
    values = summary["actual"]["by_condition"]["fn_only__binary"]
    paired = summary["paired_fractional_minus_binary"]["by_noise"]["fn_only"]

    assert values["target_count"] == 2
    assert values["defined_gradient_cosine_count"] == 1
    assert values["mean_gradient_cosine"] == -0.5
    assert values["negative_alignment_rate_defined"] == 1.0
    assert values["noisy_zero_gradient_rate"] == 0.5
    assert paired["paired_gradient_count"] == 1
    assert paired["mean_gradient_cosine_delta"] == pytest.approx(1.5)

    report = _markdown({"summary": summary})
    assert "`fn_only__binary`" in report
    assert "`clean__binary`" not in report
