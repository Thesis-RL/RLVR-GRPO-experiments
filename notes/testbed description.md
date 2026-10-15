# Week 2 Notes — Testbed Definition

**Structure of future work:** two separate studies on one shared testbed; merged into one paper.

| Study | Focus |
|---|---|
| **A** | Binary vs fractional rewards under the same verification errors |
| **B** | Asymmetric FP/FN noise (identification-corrected) |

Both studies share one formal vocabulary and the same connecting metrics.

---

## Shared formal vocabulary (“good enough”)

A verifier is good enough if training under its noisy reward still improves under a **clean** evaluation oracle. Outcomes (pre-specified from learning curves):

| Outcome | Meaning |
|---|---|
| **Preserved** | Clean primary metric rises; clean–noisy advantages/rankings usually agree |
| **Delayed** | Directionally correct but slower / higher variance; needs more steps or tokens to hit a pre-set target |
| **Distorted** | Ranking or gradient flips often enough that the policy plateaus below clean baseline, collapses, or shows rising noisy train reward with falling clean val |

**Connecting quantities:**

1. Clean–noisy **advantage agreement** (sign match, cosine / norm ratio of group advantage vectors)
2. Rollout **ranking fidelity** (Spearman / pairwise agreement, inversion rate, tie create/remove, top-1 preserved)
3. **Policy-gradient alignment** on audited minibatches (cosine, negative-alignment rate; LoRA subspace only)
4. **Reward margins** within each group (best−second, min positive gap, near-tie rate)

Target for theory: shared definitions + a few propositions/bounds or counterexamples + empirical checks. **Proposition 1** (equal-$J$ FP vs FN under binary vs fractional) is the first claim to write up; see Theory plan.

---

## Shared testbed

| Item | Choice |
|---|---|
| Domain | Short Python function synthesis |
| Data | Official MBPP (374 / 90 / 500); 3 unit tests per problem |
| Eval | Clean executor; MBPP+ (EvalPlus v0.2.0) on matching test-split IDs as stricter check |
| Algorithm | GRPO; same within-group normalization |
| Group size | 16 rollouts (8 for pipeline debug) |
| Train setup | LoRA/QLoRA; TRL + PEFT + vLLM; Adam; temps: train 1.0, eval 0.7 (pilot-confirm) |
| Seeds | 3 per main condition |
| Held fixed across studies | tasks, prompt template, model (within a phase), optimiser, eval protocol, hyperparams (frozen after clean pilot) |

**Reward values with 3 tests.** Fractional reward is discrete: $0,\ \tfrac13,\ \tfrac23,\ 1$. Sanity check: with **1 test**, binary and fractional must give identical rewards and identical training dynamics. If MBPP+ supplies enough covered training problems, run a small follow-up with **more tests** on a few cells.

**Models — two phases**

| Phase | Models | Role |
|---|---|---|
| **1. Implementation** | **Main:** Qwen3-1.7B-Base (lock after offline diagnostic). **Pipeline debug only:** Qwen2.5-Coder-1.5B-Instruct — not for scientific conclusions (see note below). **Non-Qwen check:** GLM-Z1-9B-0414 on priority-block cells if GPU allows (as in the target paper); else screen a smaller *general* (non-Coder) model | Build and debug the full pipeline; run the experimental matrices; analyse connecting metrics and freeze the protocol |
| **2. Scale-up** | Qwen3-4B-Base (paper-primary scale); Qwen3-8B-Base if resources allow | Repeat key (or full) conditions at paper-relevant scale once GPU access is available |

Phase 1 is the working baseline: results there validate the testbed and can support preliminary conclusions, with the caveat that effect sizes may change at 4B/8B. Phase 2 is confirmation at larger scale. The non-Qwen key-cell check guards against Qwen-specific RLVR artefacts.

Noise always hits **individual test outcomes before aggregation**. Evaluation uses **no synthetic noise**.

### Offline frozen-rollout diagnostic (before any RL)

Before training, for each candidate model: sample ~16 answers per **training** problem, grade with the clean tests, then apply the planned noise schedules offline and measure how much signal survives (advantage agreement, ranking fidelity, reward margins). Cheap; also used to **choose / confirm the main model**.

---

## Four decisions

### 1. Type of errors to inject

| Study | Errors |
|---|---|
| **A** | Symmetric flips at test level: (i) independent 10%; (ii) group-level 10% (all tests × all rollouts of a prompt). Optional follow-up at 30%. |
| **B** | Class-conditional: FPR $\alpha$ and FNR $\beta$ set independently on each test outcome. Resampled each verification call. |

**Study B matrix (identification-corrected; not group-shared FP/FN as in paper v2):**

| Condition | $\alpha$ (FPR) | $\beta$ (FNR) | $J=1-\alpha-\beta$ |
|---|---:|---:|---:|
| Clean | 0 | 0 | 1.0 |
| FP-only | 0.3 | 0 | 0.7 |
| FN-only | 0 | 0.3 | 0.7 |
| FP-heavy | 0.3 | 0.1 | 0.6 |
| FN-heavy | 0.1 | 0.3 | 0.6 |

Matched-$J$ pairs isolate **error type** from **error amount**. Under fractional rewards, both FP and FN shrink expected score by a factor involving $J$, so FP-only ≈ FN-only is the null. Under binary (all 3 tests must pass), they diverge: a correct answer keeps reward 1 only if none of 3 tests is FN-flipped ($0.7^3\approx 34\%$ at $\beta=0.3$), while a fully failing wrong answer gets reward 1 only if every test is FP-flipped ($0.3^3\approx 3\%$ at $\alpha=0.3$). So at the same $J$, binary scoring makes false negatives far more damaging than false positives (Proposition 1).

Malformed / timeout → all clean tests fail; no uncorrupted format bonus.

### 2. Binary or fractional reward

| Study | Reward |
|---|---|
| **A** | **Both.** Paired train runs: only the map criterion→scalar changes (all-tests-pass binary vs fraction of tests passed). Same tests, same noise realization. |
| **B** | **Train with fractional.** Every logged group is also **rescored offline with binary** (clean and noisy) so reward construction is comparable without doubling the B matrix. |

**Priority block (run first, not last).** Clean + FP-only + FN-only, each with binary and fractional, 3 seeds (**core of the bridge**; + clean shared). Same hparams/model. These are the cells where Proposition 1 predicts a binary FP≠FN gap and a fractional FP≈FN null. Running them first avoids discovering only “FP ≈ FN” from a fractional-only B matrix.

### 3. Stochastic or the same each time

Noise is **resampled** on every verification call in both studies (same protocol as the original paper’s controlled noise).

### 4. How the corruption happens

1. Generate rollouts → execute tests → get clean outcome matrix $z$.
2. Apply noise → noisy matrix $\tilde z$.
3. Aggregate $\tilde z$ → training reward (binary and/or fractional).
4. GRPO sees only the noisy reward; clean outcomes/rewards saved for analysis.
5. Validate/test with clean executor only.

**Study A — corruption modes:** independent Bernoulli flip per test outcome; or one group-level flip decision that complements the whole group matrix.

**Study B — corruption:** per outcome, $P(\tilde z=1\mid z=0)=\alpha$, $P(\tilde z=0\mid z=1)=\beta$; independent across tests/rollouts (not one shared group draw for FP and FN).

---

## Logging

Save **every rollout** (JSONL/Parquet); W&B for aggregates. TRL `log_completions` alone is insufficient.

| Field | Use |
|---|---|
| run/seed, step, checkpoint, prompt/group IDs | reproducibility |
| code, tokens, log-probs, length, exec errors | generation vs optimisation failures |
| clean test outcomes, corruption mask, noisy outcomes | realised error rates |
| clean & noisy rewards for **both** binary and fractional (train uses one; the other always logged) | distortion by reward type; cross-study bridge |
| clean & noisy advantages; group mean/std; zero-variance flag | advantage agreement |
| **Reward margins:** best−second, min positive gap, near-tie rate (clean and noisy) | when noise erases or invents contrasts |
| ranking stats: pairwise fidelity, inversions, tie create/remove, top-1 preserved | ranking fidelity |
| loss, grad norm; KL/entropy if available | optimisation health |
| Study B (+ cross): realised $\alpha,\beta$, $J$, precision/recall, changed-label rate | verifier behaviour online |
| noisy train reward vs clean val gap over time | exploitation / distortion flag |
| **problem difficulty** (e.g. clean solve rate / pass rate bucket from the offline diagnostic or a fixed clean baseline) | Report main results **split by difficulty** — hypothesis depends on how often the model already solves the problem |

### Required gradient audit

At early / mid / late checkpoints, on **frozen saved groups**:

1. Recompute clean and noisy advantages under the study’s train aggregation (and under the other aggregation for offline comparison).
2. Compute clean vs noisy policy-gradient direction **without** an optimiser step (LoRA adapter subspace).
3. Log cosine, dot product, norm ratio, negative-alignment rate, clipping rate.

Same audit protocol and metric definitions in Study A, Study B, and the binary cross.

### Curve metrics

Pass@1 and mean unit-test pass rate; peak and final clean val; area under clean learning curve; rollouts/tokens to a **pre-specified** target (or explicit failure to reach it). Classify each run as preserved / delayed / distorted using the pre-registered rules above. Always report the primary metric **overall and by difficulty stratum**.

---

## Theory plan

1. Formalise GRPO update in terms of criterion vector → reward map → advantages.
2. Define signal preservation / attenuation / reversal via advantage agreement and gradient alignment.
3. **Proposition 1 (write first, ~1 page):** at matched $J$, FP-only and FN-only are approximately exchangeable under **fractional** aggregation (both shrink expected score), but **not** under **binary** with $K=3$ tests — FN destroys correct programs’ all-pass reward with probability $1-(1-\beta)^K$, while FP invents all-pass on a fully failing program only with probability $\alpha^K$. Predicts: fractional FP≈FN; binary FN ≫ FP harm. Connects Study A (aggregation) and Study B (asymmetry).
4. Further checkable claims, e.g. fractional averaging can damp independent criterion noise relative to binary under the same $\tilde z$.
5. Support with counterexamples on frozen groups (offline diagnostic) and with the main runs.

---

## Run budget

Counts below are for **one model** (Phase 1 main first). Non-Qwen repeats the **priority block** only. Scale-up repeats selected cells.

| Order | Block | Rough matrix |
|---|---|---|
| **1st** | **Priority / bridge** | Clean + (FP-only, FN-only) × (binary, fractional) × 3 seeds ≈ **15** (clean shared once if hparams match) |
| **2nd** | **A** | 2 rewards × 3 verify conditions (clean / ind.10% / group.10%) × 3 seeds = **18**; +12 if 30% sweep; drop cells already covered by priority where identical |
| **3rd** | **B remainder** | FP-heavy & FN-heavy × fractional × 3 seeds = **6** (FP/FN-only fractional already in priority) |
| **Sanity / follow-up** | 1-test binary≡fractional check; optional more-tests (MBPP+) on a few cells | small |
| **Shared** | One clean baseline if model/hparams/data match | — |
