# Reproducing Priority Bridge v2

Run these commands from the repository root. Use
`configs/priority_bridge_v2.yaml` for the completed experiment and a separate
directory for newly generated outputs.

## Regenerate the analysis from saved data

Copy the minimum set of inputs to the listed paths. [Artifact inventory](ARTIFACTS.md)

```bash
python3 -B scripts/analyze_priority_bridge_offline.py \
  --archive outputs/online_matrix/priority_bridge_v2 \
  --out-dir outputs/analysis_recheck
```

This command works with Python 3.11 or later. It reads the saved evaluations,
rollouts, and gradient metrics and produces `analysis.json` and three SVGs.
The published results are in
[analyses/priority_bridge_v2_2026-11-25](../analyses/priority_bridge_v2_2026-11-25/README.md).

To rebuild the matched-$J$ **FN−FP** tables from those published artifacts
(without re-reading raw rollouts), run:

```bash
python3 -B scripts/extract_fp_vs_fn_contrasts.py \
  --analysis-json analyses/priority_bridge_v2_2026-11-25/analysis.json \
  --stage6-json analyses/priority_bridge_v2_2026-11-25/provenance/stage6_multiseed_reference.json \
  --out-dir analyses/priority_bridge_v2_2026-11-25
```

This writes [FP_VS_FN.md](../analyses/priority_bridge_v2_2026-11-25/FP_VS_FN.md) and
[fp_vs_fn.json](../analyses/priority_bridge_v2_2026-11-25/fp_vs_fn.json).

To recompute the primary endpoint analysis independently, run:

```bash
uv sync
uv run python scripts/analyze_priority_bridge_multiseed.py \
  --config configs/priority_bridge_v2.yaml \
  --matrix-root outputs/online_matrix/priority_bridge_v2/main \
  --base-root outputs/online_matrix/priority_bridge_v2/common-base \
  --out-dir outputs/multiseed_recheck
```

## Check the code

```bash
uv sync
uv run pytest -q
uv run ruff check src scripts tests
uv run ruff format --check src scripts tests
uv run python scripts/run_online_grpo.py --config configs/priority_bridge_v2.yaml --list-conditions
```

## Reproduce training on a GPU

Training requires a CUDA-compatible PyTorch build, TRL and PEFT
(`uv sync --extra train`), the pinned model, and the frozen difficulty maps.
The [experiment protocol](EXPERIMENT_PROTOCOL.md) describes the settings.
Check the environment with the
[two-step smoke configuration](../configs/priority_bridge_v2_integration_smoke.yaml)
before running the full matrix.

This command runs all six conditions for one seed. To repeat the full matrix,
use seeds 0, 1, and 2 with separate output directories.

```bash
uv run python scripts/run_online_matrix.py \
  --config configs/priority_bridge_v2.yaml \
  --seed 0 --out-root outputs/online_matrix/priority_bridge_v2_replay/seed-0
```

To rerun the LoRA gradient calculations on a GPU, see the
[audit protocol](GRADIENT_AUDIT_PROTOCOL.md) and
[configuration](../configs/gradient_audit_v1_1.yaml). The
[provenance notes](../analyses/priority_bridge_v2_2026-11-25/provenance/README.md)
identify the exact code and environment used for the completed runs.
