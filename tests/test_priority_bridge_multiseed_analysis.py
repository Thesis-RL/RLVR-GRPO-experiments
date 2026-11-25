from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_priority_bridge_multiseed import (
    _base_slice,
    _condition_result,
    _seed_t_interval,
    _seed_task_summary,
    _validate_matched_generation,
    _write_csv,
)


def test_seed_t_interval_uses_three_seed_uncertainty() -> None:
    low, high = _seed_t_interval(np.asarray([0.01, 0.02, 0.03]))

    assert (low + high) / 2 == pytest.approx(0.02)
    assert low < 0.0 < high


def test_condition_result_preserves_paired_seed_deltas() -> None:
    base = np.asarray([0.0, 1.0, 0.0, 1.0])
    trained = np.stack([base + delta for delta in (0.1, 0.2, 0.3)])

    result = _condition_result(
        trained=trained,
        base=base,
        seeds=(0, 1, 2),
        replicates=200,
        bootstrap_seed=7,
    )

    delta = result["delta_trained_minus_base"]
    assert delta["estimate"] == pytest.approx(0.2)
    assert delta["by_seed"] == pytest.approx({"0": 0.1, "1": 0.2, "2": 0.3})


def test_two_way_bootstrap_is_deterministic() -> None:
    values = np.asarray([[0.0, 1.0], [0.5, 1.0], [1.0, 1.0]])

    first = _seed_task_summary(
        values,
        seeds=(0, 1, 2),
        replicates=300,
        seed=11,
    )
    second = _seed_task_summary(
        values,
        seeds=(0, 1, 2),
        replicates=300,
        seed=11,
    )

    assert first == second


def test_csv_accepts_heterogeneous_summary_columns(tmp_path) -> None:
    path = tmp_path / "summary.csv"

    _write_csv(path, [{"series": "base", "task_ci": 0.1}, {"series": "trained", "seed_ci": 0.2}])

    header, *rows = path.read_text().splitlines()
    assert header == "series,task_ci,seed_ci"
    assert len(rows) == 2


def test_analysis_rejects_unmatched_generation_seed() -> None:
    base = [{"task_id": 1, "generation_chunk_seed_slot": 0, "generation_seed": 123}]
    trained = [{"task_id": 1, "generation_chunk_seed_slot": 4, "generation_seed": 456}]

    with pytest.raises(ValueError, match="generation seeds do not match"):
        _validate_matched_generation(base, trained, condition="clean__binary", seed=0, step=1010)


def test_analysis_aligns_the_shared_base_slice_with_the_checkpoint() -> None:
    base = [
        {"task_id": task_id, "sample_index": sample_index}
        for task_id in (11, 12, 13)
        for sample_index in range(8)
    ]

    selected = _base_slice(base, checkpoint_index=1, samples=4, task_ids={11, 12})

    assert len(selected) == 8
    assert {row["task_id"] for row in selected} == {11, 12}
    assert {row["sample_index"] for row in selected} == {0, 1, 2, 3}
