# Priority bridge — seed 0 report

> **Scope:** one seed only. All comparisons are descriptive and must not be interpreted as uncertainty-calibrated effects.

## Completion and integrity

- Matrix status: **complete** (6/6 conditions).
- Validation: **passed**; 0 errors and 0 warnings.
- Each condition contains 374 optimizer steps, 10 scheduled checkpoints, 5984 rollout rows, ten 90-problem validation evaluations, and two 500-problem test evaluations.
- Adapter hashes differ between checkpoint 37 and checkpoint 374 in every run, and every run records non-zero gradient steps.

## Results

| Condition | Clean Pass@1 | Δ vs clean | Mean clean reward | Mean noisy reward | Mean abs. advantage | Zero-var groups | Changed labels |
|---|---|---|---|---|---|---|---|
| clean__binary | 0.4680 | 0.0000 | 0.4027 | 0.4027 | 0.6245 | 0.2032 | 0.0000 |
| clean__fractional | 0.4730 | 0.0000 | 0.4801 | 0.4801 | 0.7130 | 0.0829 | 0.0000 |
| fp_only__binary | 0.4630 | -0.0050 | 0.4134 | 0.4514 | 0.7068 | 0.0936 | 0.1580 |
| fp_only__fractional | 0.4700 | -0.0030 | 0.4748 | 0.6316 | 0.7703 | 0.0214 | 0.1568 |
| fn_only__binary | 0.4680 | 0.0000 | 0.4098 | 0.1405 | 0.5024 | 0.3128 | 0.1392 |
| fn_only__fractional | 0.4740 | 0.0010 | 0.4692 | 0.3256 | 0.7314 | 0.0775 | 0.1436 |

The primary metric is the mean clean test Pass@1 over checkpoints 337 and 374.

## Descriptive conclusions

- Across this single seed, the largest absolute clean Pass@1 change from 30% one-sided label noise was 0.5 percentage points.
- Fractional reward exceeded binary reward by 0.5 to 0.7 percentage points across the three noise regimes.
- The corruption moved training rewards in the intended direction: FP-only raised the noisy-minus-clean mean by 0.038 (binary) and 0.157 (fractional), while FN-only changed it by -0.269 and -0.144.
- Fractional reward reduced zero-variance groups relative to binary by 12.0 points for clean, 7.2 for FP-only, and 23.5 for FN-only; the largest signal-preservation difference appeared under FN noise.
- These are descriptive results from seed 0 only. The differences are small relative to checkpoint-to-checkpoint variation and are not evidence of a population-level effect.

## Artifacts

- `analysis.json`: complete machine-readable analysis and validation details.
- `artifact_manifest.json`: SHA-256 hashes and sizes for source artifacts and checkpoints.
- `condition_summary.csv` / `.md`: compact result tables.
- `pass_at_1.svg`: primary-metric comparison.
- `training_diagnostics.svg`: advantage signal and zero-variance groups.
