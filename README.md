# RLVR-GRPO experiments

This project compares two ways to reward a model for solving Python problems.
Binary reward gives full credit when all three tests pass. Fractional reward
gives credit for the fraction of tests that pass. The GRPO runs compared both
rewards under clean verification and two types of simulated verifier errors.
A false positive (FP) marks a failed test as passed. A false negative (FN)
marks a passed test as failed. During training, each eligible test had a 30%
chance of being flipped. Evaluation always used clean verification.

The main experiment, **Priority Bridge v2**, trained Qwen3-1.7B-Base with LoRA
on MBPP under six conditions, with three training seeds per condition and
1,122 steps per run. False negatives erased reward differences more often
under binary aggregation. With three seeds, the effect of reward aggregation
on the FN–FP difference in test Pass@1 remains uncertain.

## Where to start

1. The [progress report](analyses/priority_bridge_v2_2026-11-25/PROGRESS_REPORT_EN.md)
   summarizes the experiment and its results.
2. The [analysis and figures](analyses/priority_bridge_v2_2026-11-25/README.md)
   explain how the metrics were computed and where the reward signal was lost.
3. The [artifact inventory](docs/ARTIFACTS.md) lists the raw runs and the files
   needed to reproduce the analysis. The reports and figures are available
   directly in this branch. Raw runs are shared separately.

## Where to find the code

| Task | File |
| --- | --- |
| Configuration for the completed experiment | [configs/priority_bridge_v2.yaml](configs/priority_bridge_v2.yaml) |
| Run the matrix or a single condition | [run_online_matrix.py](scripts/run_online_matrix.py), [run_online_grpo.py](scripts/run_online_grpo.py) |
| Verify solutions, compute rewards, and evaluate | [src/rlvr/](src/rlvr/) |
| Generate a matched base-model evaluation | [evaluate_multisample.py](scripts/evaluate_multisample.py) |
| Generate or regrade the inputs for the frozen difficulty maps | [run_offline_diagnostic.py](scripts/run_offline_diagnostic.py), [grade_and_analyze_offline.py](scripts/grade_and_analyze_offline.py) |
| Regenerate the analysis and figures | [analyze_priority_bridge_offline.py](scripts/analyze_priority_bridge_offline.py) |
| Aggregate evaluation results across seeds | [analyze_priority_bridge_multiseed.py](scripts/analyze_priority_bridge_multiseed.py) |
| Audit LoRA gradients | [audit_lora_gradients.py](scripts/audit_lora_gradients.py) |

The [experiment protocol](docs/EXPERIMENT_PROTOCOL.md) records the training and
evaluation setup. The [reproduction guide](docs/REPRODUCING_V2.md) provides
commands for offline analysis and for rerunning training.
`configs/priority_bridge.yaml` defines the inherited condition matrix. Pass
`configs/priority_bridge_v2.yaml` explicitly for v2. The repository includes
the MBPP splits and frozen difficulty maps.
