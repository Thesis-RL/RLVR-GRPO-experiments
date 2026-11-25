# Priority Bridge v2 experiment protocol

The protocol was frozen on 17 November 2026, before the main matrix, and used
for all 18 runs. [Configuration](../configs/priority_bridge_v2.yaml)

## Why training ran for 1,122 steps

A separate pilot compared clean/binary and clean/fractional with seed 9001.
Across three late checkpoints (898, 1010, and 1122), mean Pass@1 on the 90
validation tasks rose by 3.70 percentage points over step 0 for binary and by
1.48 points for fractional. Binary cleared the pre-specified 3-point
improvement threshold, and both conditions met the criterion for avoiding
substantial degradation. The pilot supported three passes through the
training set. Test evaluation followed protocol selection.
[Pilot criteria and decision](../analyses/priority_bridge_v2_2026-11-25/provenance/stage5_pilot_gate.json)

## Training

- Qwen3-1.7B-Base with LoRA, revision `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`.
- Six conditions paired clean, FP-only at 30%, and FN-only at 30% with binary
  or fractional reward. Each condition used training seeds 0, 1, and 2.
- Training ran for 1,122 optimizer steps, three passes through 374 tasks,
  with 16 completions per task. The learning rate was 2e-6, with 56 warmup
  steps followed by linear decay.
- Training decoding used temperature 1.0, top-p 0.95, and a limit of 512 new
  tokens. Verifier v2.1 used a 256 MiB memory limit and one retry with a
  five-second timeout after an initial 0.5-second timeout.

## Evaluation

Validation used 90 tasks and one response per task. It ran at step 0 and at
10% intervals through training. Test used 500 tasks and four responses per
task at each of two checkpoints, steps 1010 and 1122. The untrained model
generated eight responses per test task. The first checkpoint was paired
with the first four baseline decoding slots and the second with the remaining
four. The primary Pass@1 gives equal weight to both checkpoints and estimates
the probability that one response passes all tests. The random number
generator state was restored after integrated evaluation so it would leave
subsequent training unaffected.

Before the main matrix, the configuration, frozen difficulty maps, and
4,000 baseline responses were checked. A two-step smoke run verified test
output logging and decoding-slot alignment. All 18 runs then completed under
the same protocol. The [detailed analysis](../analyses/priority_bridge_v2_2026-11-25/REPORT.md)
gives uncertainty intervals and per-seed results.
