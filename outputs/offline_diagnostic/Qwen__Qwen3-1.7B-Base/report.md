# Offline diagnostic — `Qwen/Qwen3-1.7B-Base`

## Model screening (pre-registered)

- Mean fractional reward: **0.4879** (need ∈ [0.05, 0.60])
- Frac problems with nonzero clean variance: **0.917** (need ≥ 0.15)
- Eligible as Phase-1 main: **True**
- Strata: {'easy': 134, 'medium': 103, 'hard': 137}

## Proposition 1 offline (FN − FP)

| Aggregation | Δ agreement (FN−FP) | Δ pairwise (FN−FP) |
|---|---:|---:|
| fractional | -0.015 | 0.009 |
| binary | -0.321 | -0.587 |

## Signal summaries

| Condition | Agr. | Cos | Pairwise | Top-1 | Agr. bucket | Pair bucket |
|---|---:|---:|---:|---:|---|---|
| clean__binary | 1.000 | 1.000 | 1.000 | 1.000 | survives | survives |
| clean__fractional | 1.000 | 1.000 | 1.000 | 1.000 | survives | survives |
| fn_only__binary | 0.647 | 0.453 | 0.352 | 0.488 | attenuated | destroyed |
| fn_only__fractional | 0.862 | 0.815 | 0.891 | 0.473 | survives | survives |
| fp_only__binary | 0.968 | 0.919 | 0.939 | 1.000 | survives | survives |
| fp_only__fractional | 0.877 | 0.820 | 0.882 | 0.955 | survives | survives |
| group_flip_10__binary | 0.902 | 0.802 | 0.889 | 0.892 | survives | survives |
| group_flip_10__fractional | 0.889 | 0.778 | 0.889 | 0.889 | survives | survives |
| ind_flip_10__binary | 0.858 | 0.706 | 0.722 | 0.756 | survives | attenuated |
| ind_flip_10__fractional | 0.849 | 0.838 | 0.915 | 0.748 | survives | survives |
