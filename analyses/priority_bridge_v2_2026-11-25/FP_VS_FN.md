# FP vs FN at matched J · Priority Bridge v2

## Sources

| Input | Contents | SHA-256 |
| --- | --- | --- |
| `analyses/priority_bridge_v2_2026-11-25/provenance/stage6_multiseed_reference.json` | Pass@1 planned contrasts; training-signal condition means | `1b21b0aed6b3…` |
| `analyses/priority_bridge_v2_2026-11-25/analysis.json` | Group erasure / pairwise preservation; LoRA audit | `8519f7cc27ec…` |

`scripts/extract_fp_vs_fn_contrasts.py` ← `scripts/analyze_priority_bridge_multiseed.py`, `scripts/analyze_priority_bridge_offline.py`

```bash
python3 -B scripts/extract_fp_vs_fn_contrasts.py \
  --analysis-json analyses/priority_bridge_v2_2026-11-25/analysis.json \
  --stage6-json analyses/priority_bridge_v2_2026-11-25/provenance/stage6_multiseed_reference.json \
  --out-dir analyses/priority_bridge_v2_2026-11-25
```

## 1. Endpoint · clean test Pass@1 (FN − FP)

| Contrast | Estimate (pp) | Seed-t 95% CI (pp) | By seed (pp) |
| --- | ---: | --- | --- |
| `binary_fn_minus_fp` | -0.18 | [-1.62, +1.27] | 0: -0.70, 1: +0.45, 2: -0.27 |
| `fractional_fn_minus_fp` | +0.01 | [-0.62, +0.63] | 0: +0.27, 1: -0.22, 2: -0.03 |
| `fn_vs_fp_aggregation_interaction` | +0.18 | [-1.87, +2.24] | 0: +0.97, 1: -0.68, 2: +0.25 |

Gains over the matched untrained baseline:

| Condition | Gain (pp) | Seed-t 95% CI (pp) |
| --- | ---: | --- |
| `fp_only__binary` | +0.93 | [+0.05, +1.82] |
| `fn_only__binary` | +0.76 | [-0.02, +1.53] |
| `fp_only__fractional` | +1.22 | [+0.71, +1.72] |
| `fn_only__fractional` | +1.23 | [+0.79, +1.66] |
| `clean__binary` | +1.19 | [+0.83, +1.56] |
| `clean__fractional` | +0.97 | [+0.54, +1.41] |

## 2. Training-signal means and FN − FP

Condition means over training seeds 0, 1, 2.

| Metric | FP/binary | FN/binary | FN−FP | FP/frac | FN/frac | FN−FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mean_pairwise_fidelity` | 93.76% | 34.66% | -59.11 pp | 87.67% | 88.62% | +0.95 pp |
| `mean_advantage_sign_agreement` | 96.66% | 65.38% | -31.28 pp | 87.18% | 86.07% | -1.11 pp |
| `mean_advantage_cosine` | 91.58% | 45.49% | -46.09 pp | 81.08% | 81.18% | +0.10 pp |
| `mean_top1_preserved` | 100.00% | 49.38% | -50.62 pp | 94.08% | 48.23% | -45.85 pp |
| `mean_spearman` | 91.58% | 45.49% | -46.09 pp | 80.50% | 79.39% | -1.11 pp |
| `zero_variance_group_fraction` | 11.56% | 28.34% | +16.79 pp | 4.07% | 6.63% | +2.55 pp |
| `nonzero_train_advantage` | 88.44% | 71.66% | -16.79 pp | 94.33% | 91.97% | -2.36 pp |
| `mean_changed_label_rate` | 15.06% | 14.43% | -0.62 pp | 15.17% | 15.03% | -0.14 pp |

### FN−FP threshold checks

Thresholds: abs(FN−FP) ≤ 0.05 for near-equality; FN worse than FP by ≥ 0.05.

| Aggregation | Metric | FN−FP | abs(Δ) ≤ 0.05 | FN worse ≥ 0.05 |
| --- | --- | ---: | :---: | :---: |
| binary | `mean_advantage_sign_agreement` | -31.28 pp | no | yes |
| binary | `mean_pairwise_fidelity` | -59.11 pp | no | yes |
| binary | `mean_top1_preserved` | -50.62 pp | no | yes |
| fractional | `mean_advantage_sign_agreement` | -1.11 pp | yes | no |
| fractional | `mean_pairwise_fidelity` | +0.95 pp | yes | no |
| fractional | `mean_top1_preserved` | -45.85 pp | no | yes |

## 3. Within-group rollout contrast (`analysis.json`)

| Noise | Aggregation | Erased rate | Pairwise preserved | Created contrast |
| --- | --- | ---: | ---: | ---: |
| fn_only | binary | 15.29% (832/5440) | 34.51% | 223 |
| fn_only | fractional | 0.61% (37/6095) | 87.11% | 223 |
| fp_only | binary | 0.37% (20/5442) | 93.54% | 546 |
| fp_only | fractional | 0.33% (20/6057) | 86.05% | 416 |

Hard tasks under binary (erasure):

- `fn_only`: 36.10% (557/1543)
- `fp_only`: 0.00% (0/1579)

## 4. LoRA audit · actual training aggregation

| Train aggregation · metric | FN mean | FP mean | FN−FP | n FN / n FP |
| --- | ---: | ---: | ---: | ---: |
| `binary__advantage_cosine` | 0.423 | 0.913 | -0.491 | 6 / 7 |
| `binary__gradient_cosine` | 0.479 | 0.904 | -0.425 | 6 / 7 |
| `fractional__advantage_cosine` | 0.805 | 0.839 | -0.034 | 6 / 6 |
| `fractional__gradient_cosine` | 0.861 | 0.907 | -0.046 | 6 / 6 |

## Summary

- Endpoint: FN−FP is -0.18 pp under binary and +0.01 pp under fractional; both seed-level intervals span zero.
- Signal, binary: Under binary reward, FN is much worse than FP on fidelity, advantage agreement/cosine, top-1 preservation, and contrast erasure.
- Signal, fractional: Under fractional reward, FN≈FP on fidelity and advantage metrics (|Δ|≲0.01), but top-1 preservation remains much lower under FN.
- Gradients: On audited groups with the actual training aggregation, clean→noisy cosine is substantially lower for FN than FP under binary, and nearly matched under fractional.

[fp_vs_fn.json](fp_vs_fn.json)
