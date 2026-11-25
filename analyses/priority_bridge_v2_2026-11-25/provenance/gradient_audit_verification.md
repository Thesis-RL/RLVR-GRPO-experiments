# Independent completeness check

This check was performed after the user requested confirmation that the fast
gradient audit covered the complete planned analysis.

## Requirement-to-artifact check

| Frozen requirement | Verification |
|---|---|
| early / middle / late checkpoints | steps 112 / 561 / 1010 present for every condition and seed |
| all bridge conditions and three seeds | 6 × 3 × 3 = 54 unique targets |
| frozen saved groups | exactly one 16-completion group per target; 864 rows total |
| recompute clean/noisy advantages | independently recomputed from rewards with sample std and epsilon 1e-4; max absolute error 0.0 |
| actual and alternative aggregation | binary and fractional present in every target; exactly one marked actual |
| LoRA policy-gradient direction | 3,211,264 trainable LoRA parameters; no non-LoRA trainable parameters |
| no optimizer step | frozen protocol and implementation perform backward only |
| cosine, dot, norms, norm ratio | present for both aggregations in all target JSON files |
| negative alignment and zero-gradient handling | present; undefined cosine remains null and is counted separately |
| clipping rate | present as 0.0 under the frozen on-policy ratio=1 definition |
| phase and seed breakdowns | actual/counterfactual condition-phase and condition-seed summaries present |
| positive control | all defined clean-condition gradient cosines are approximately 1 |

All four reward vectors and all four advantage vectors were independently
recomputed for all 54 groups from `clean_outcomes` and `noisy_outcomes`.
Maximum absolute differences from the saved values were exactly zero. The
saved `train_advantage` also matched the noisy advantage selected by the
condition with maximum absolute error zero.

Each of the 54 stored adapter hashes was independently compared with the
corresponding local `adapter_model.safetensors`; there were no mismatches. The
local audit script hash equals the script in the executed devbox release:
`35850279ced3011c3d7c30ddda4e562281c29f2bf2f6784e04ea08cbd28b7a59`.
The executed Docker image ID is
`sha256:6eed5f68f64452dff1106c73e0a25f93766a97ea0f3dfab438e28236e52398b4`.

## Why execution was fast

The audit did not train or generate programs. The base model was loaded once;
then only 54 already saved groups were replayed, with backward restricted to
the small LoRA subspace. The main GPU calculation therefore took roughly 2.5
minutes after model loading. Speed is not evidence that targets were skipped:
the output contains all 54 expected target files.

## What this audit does not establish

- It samples one exact group per condition/seed/phase, not all 1,122 groups.
- It is a mechanistic diagnostic and has no population confidence interval.
- Dropout is disabled for deterministic comparison, so it is not a replay of
  stochastic training-mode dropout.
- Original completion token IDs were not logged. Three of 864 completions each
  omit one unreconstructable non-canonical subtoken; this is reported rather
  than guessed.
- Clipping is zero by the executed `num_iterations=1`, on-policy audit
  definition; it is not estimated from separately saved old-policy log-probs.
- Full Study A and Study B cells that were never trained are outside this
  bridge audit.

These limitations do not invalidate the observed FN/binary versus
FN/fractional mechanism, but they prevent describing it as an average effect
over every training step or as a statistically conclusive endpoint result.
