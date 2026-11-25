# Progress Report · Priority Bridge v2

25 November 2026

The experiment examines how binary and fractional rewards respond to false
positives and false negatives in RLVR with GRPO. After correcting the verifier
and adding a matched multi-sample baseline, the original one-pass run was
re-evaluated. Its gains in clean test Pass@1 ranged from −0.10 to +0.43
percentage points, with confidence intervals spanning zero. A separate
validation-only pilot selected the longer training protocol for the main run.

The full matrix is now complete for three seeds. Qwen3-1.7B-Base was trained on
MBPP for 1,122 steps with LoRA under six conditions combining clean, FP-only
and FN-only verification with binary or fractional reward. Test evaluation
used four responses per task at each of the final two checkpoints, paired
with the corresponding samples from a shared untrained baseline.

Against the 48.45% baseline, the six conditions gained 0.76 to 1.23
percentage points in mean clean test Pass@1. The pre-specified interaction
between error type and reward aggregation was +0.18 percentage points, with a
95% seed-level interval of −1.87 to +2.24 points. Its direction remains
uncertain.

The saved rollouts give a clearer account of the training signal. Under false
negatives, binary reward lost all contrast in 832 of 5,440 groups that had an
informative clean binary signal. Fractional reward lost all contrast in one of
the same groups. Loss of binary contrast was most frequent on hard tasks,
where fewer responses passed all tests before noise. In the LoRA audit,
fractional reward gave higher clean-to-noisy gradient alignment in 10 of 11
FN groups where both metrics were defined. The gradient result describes the
audited groups. The endpoint result remains uncertain across training seeds.

The [analysis package](README.md) contains the detailed results and figures.
