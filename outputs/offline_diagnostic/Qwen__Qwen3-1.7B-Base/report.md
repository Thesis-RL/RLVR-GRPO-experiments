# Offline diagnostic — `Qwen/Qwen3-1.7B-Base`

## Model screening (pre-registered)

- Mean fractional reward: **0.4817** (need ∈ [0.05, 0.60])
- Frac problems with nonzero clean variance: **0.904** (need ≥ 0.15)
- Eligible as Phase-1 main: **True**
- Strata: {'easy': 132, 'medium': 99, 'hard': 143}

## Proposition 1 offline (FN − FP)

| Aggregation | Δ agreement (FN−FP) | Δ pairwise (FN−FP) |
|---|---:|---:|
| fractional | -0.013 | 0.005 |
| binary | -0.322 | -0.598 |

## Signal summaries

| Condition | Agr. | Cos | Pairwise | Top-1 | Agr. bucket | Pair bucket |
|---|---:|---:|---:|---:|---|---|
| clean__binary | 1.000 | 1.000 | 1.000 | 1.000 | survives | survives |
| clean__fractional | 1.000 | 1.000 | 1.000 | 1.000 | survives | survives |
| fn_only__binary | 0.646 | 0.450 | 0.349 | 0.500 | attenuated | destroyed |
| fn_only__fractional | 0.866 | 0.815 | 0.889 | 0.492 | survives | survives |
| fp_only__binary | 0.968 | 0.918 | 0.947 | 1.000 | survives | survives |
| fp_only__fractional | 0.879 | 0.819 | 0.884 | 0.945 | survives | survives |
| group_flip_10__binary | 0.902 | 0.801 | 0.889 | 0.891 | survives | survives |
| group_flip_10__fractional | 0.889 | 0.777 | 0.889 | 0.889 | survives | survives |
| ind_flip_10__binary | 0.861 | 0.709 | 0.731 | 0.757 | survives | attenuated |
| ind_flip_10__fractional | 0.853 | 0.839 | 0.914 | 0.740 | survives | survives |
