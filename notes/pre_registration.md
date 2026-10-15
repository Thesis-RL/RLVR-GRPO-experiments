# Pre-registration — locked before offline diagnostic / training results

Locked: 2026-10-14. Do not revise thresholds after looking at empirical numbers from this project.

## One primary metric (all studies)

**Clean Pass@1 on the 500 MBPP test problems**, averaged over the **last two** scheduled clean evaluations of a run.

- Evaluation temperature: 0.7; one sample per problem for Pass@1 (same protocol for every condition).
- No synthetic noise on evaluation.
- Always also report the primary metric **by difficulty stratum** (see below).

## One main comparison per study

| Study | Main contrast (same model, hparams, seed protocol) | Expected under Prop. 1 / design |
|---|---|---|
| **A** | Binary − fractional under the **same** noise schedule (priority: clean / ind.10% / group.10%) | Binary more fragile to the same flips than fractional |
| **B** | FN-only − FP-only at matched $J=0.7$ (fractional train) | ≈ 0 under fractional (**H1**) |
| **Bridge (priority block)** | Same FN−FP contrast under **binary** vs under **fractional** | Binary: FN harms more than FP (**H2**); fractional: FN ≈ FP |

Secondary contrasts (not used to pick the headline): each noisy cell − clean; FP-heavy vs FN-heavy at $J=0.6$.

## Outcome labels: preserved / delayed / distorted

Compare each noisy run to the **matched clean run** (same model, reward aggregation, seed index) on the primary metric. Let:

- $P_{\mathrm{clean}}$ = primary metric of the clean run
- $P$ = primary metric of the noisy run
- $P_0$ = primary metric of the shared step-0 / untrained baseline (same model; one value reused)
- $N_{\mathrm{clean}}$, $N$ = number of training rollouts until the run first reaches target $T$ (defined below); $\infty$ if never

**Target** $T$ (for “delayed” timing only):

$$
T = P_0 + 0.05
$$

(absolute Pass@1 points). If $P_{\mathrm{clean}} < T$, replace $T$ with $P_{\mathrm{clean}}$ for that seed pair (clean never hit +5pp → use clean’s own final as the bar).

| Label | Numeric rule (first match wins, checked in this order) |
|---|---|
| **Distorted** | (i) $P < P_0 + 0.01$ (no meaningful clean gain), **or** (ii) $P \le P_{\mathrm{clean}} - 0.05$, **or** (iii) over the last three clean evals, clean Pass@1 is non-increasing while mean noisy train reward is non-decreasing and the noisy−clean reward gap widens by ≥ 0.05 |
| **Delayed** | Not distorted, and either (i) $N > 1.5 \cdot N_{\mathrm{clean}}$ with both finite, **or** (ii) $P_{\mathrm{clean}} - 0.05 < P \le P_{\mathrm{clean}} - 0.02$ |
| **Preserved** | Not distorted and not delayed, and $P \ge P_{\mathrm{clean}} - 0.02$ |

If a run is stopped early for non-scientific reasons (OOM, crash), exclude from labeling; do not reassign.

## Offline frozen-rollout diagnostic (this step)

No optimiser. Per training problem: $G=16$ frozen answers; clean grade; apply planned noise offline; GRPO-style group advantages.

**Noise cells (priority):** clean; FP-only $(\alpha,\beta)=(0.3,0)$; FN-only $(0,0.3)$; plus Study A flips: independent 10%, group-level 10%. Aggregate each with **binary** and **fractional**. Noise seed fixed per (problem, condition, noise_draw); report mean over **3** independent noise draws.

**Signal-survival summaries** (means over problems with non-zero clean reward variance; zero-variance groups reported separately):

| Quantity | “Survives” | “Attenuated” | “Destroyed / reversed” |
|---|---|---|---|
| Advantage sign agreement | ≥ 0.70 | [0.55, 0.70) | < 0.55 |
| Advantage cosine (clean vs noisy vector) | ≥ 0.70 | [0.40, 0.70) | < 0.40 |
| Pairwise ranking fidelity (pairs with clean reward gap > 0) | ≥ 0.80 | [0.60, 0.80) | < 0.60 |
| Top-1 preserved | ≥ 0.70 | [0.50, 0.70) | < 0.50 |

Prop. 1 offline check: under fractional, FN-only vs FP-only summaries should be within **0.05** absolute on advantage agreement and pairwise fidelity; under binary, FN-only should be **worse** than FP-only by ≥ **0.05** on at least one of those two.

## Difficulty strata

From the offline diagnostic clean fractional pass-rate per problem (mean over $G$ rollouts), fixed once and reused for all later reporting:

| Stratum | Clean mean fractional reward |
|---|---|
| Easy | ≥ 2/3 |
| Medium | in (1/3, 2/3) |
| Hard | ≤ 1/3 |

## Model screening rule (from this diagnostic)

Confirm **Qwen3-1.7B-Base** as Phase-1 main if, on the training split under clean grading: (i) mean fractional reward is in **[0.05, 0.60]** (not near-solved, not near-dead), and (ii) ≥ **15%** of problems have non-zero within-group clean reward variance under $G=16$. Otherwise screen the next candidate with the same rule. Qwen2.5-Coder-1.5B-Instruct is pipeline-debug only and is **not** eligible as the scientific main model.
