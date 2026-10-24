from __future__ import annotations

import json

import pytest

from scripts.analyze_online_seed import _bar_svg, _summarize_rollouts


def test_summarize_rollouts_counts_signal_and_zero_variance(tmp_path) -> None:
    path = tmp_path / "rollouts.jsonl"
    rows = []
    for group_index, zero_variance in enumerate((False, True)):
        for completion_index in range(16):
            clean = float(completion_index % 2)
            noisy = clean if group_index == 0 else 0.0
            rows.append(
                {
                    "condition": "clean__binary",
                    "group_id": f"group-{group_index}",
                    "clean_binary_reward": clean,
                    "noisy_binary_reward": noisy,
                    "train_reward": noisy,
                    "train_advantage": clean - 0.5 if not zero_variance else 0.0,
                    "zero_variance_group": zero_variance,
                    "changed_label_rate": 0.0,
                    "train_aggregation_pairwise_fidelity": 1.0,
                    "train_aggregation_advantage_sign_agreement": 1.0,
                }
            )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    summary = _summarize_rollouts(path, "binary")

    assert summary["row_count"] == 32
    assert summary["group_count"] == 2
    assert summary["invalid_group_sizes"] == []
    assert summary["zero_variance_group_fraction"] == pytest.approx(0.5)
    assert summary["nonzero_train_advantage_fraction"] == pytest.approx(0.5)
    assert summary["mean_abs_train_advantage"] == pytest.approx(0.25)


def test_bar_svg_escapes_text_and_contains_bars() -> None:
    svg = _bar_svg(
        title="A & B",
        subtitle="one <seed>",
        labels=["clean\nbinary"],
        series=[("metric", [0.5], ["#123456"])],
        y_max=1.0,
        value_format=lambda value: f"{value:.1f}",
    )

    assert svg.startswith("<svg")
    assert "A &amp; B" in svg
    assert "one &lt;seed&gt;" in svg
    assert 'fill="#123456"' in svg
