# Priority Bridge v2 · 25 November 2026

The main experiment combined six verification and reward conditions with three
training seeds per condition. Start with the [progress report](PROGRESS_REPORT_EN.md)
or the [experiment overview](../../README.md).

| File | Contents |
| --- | --- |
| [Progress report](PROGRESS_REPORT_EN.md) | Summary of the work and findings |
| [Detailed analysis](REPORT.md) | Evaluation, test Pass@1, rewards, advantages, and LoRA gradients |
| [FP vs FN contrasts](FP_VS_FN.md) | Matched-$J$ FN−FP on Pass@1, training signal, group contrast, and LoRA audit |
| [FP vs FN (JSON)](fp_vs_fn.json) | Machine-readable twin of the FP vs FN note, with source paths |
| [Validation curves](validation_curves.svg) | Pass@1 and the fraction of unit tests passed over training |
| [Test checkpoints](test_checkpoints.svg) | Gain over the matched baseline at steps 1010 and 1122 |
| [Signal loss under false negatives](fn_signal_erasure.svg) | How often false negatives erased within-group reward differences, by difficulty and training phase |
| [Machine-readable results](analysis.json) | Estimates, sample sizes, and audit metrics |
| [Provenance](provenance/README.md) | Code versions, pilot, original v2 summaries, and verification results |

To regenerate the results from the saved rollouts and predictions, use the
[artifact inventory](../../docs/ARTIFACTS.md) and the
[reproduction guide](../../docs/REPRODUCING_V2.md).
