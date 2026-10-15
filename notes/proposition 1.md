# Proposition 1 — Equal-$J$ FP vs FN under Binary vs Fractional Rewards

## Setup

- $K=3$ unit tests (MBPP). Clean outcomes $z\in\{0,1\}^K$; noisy $\tilde z$ with FPR $\alpha$, FNR $\beta$ (independent per test).
- $J=1-\alpha-\beta$. Matched-$J$ pairs isolate **error type** from **error amount**.
- **Fractional:** $r_{\mathrm{frac}}=\tfrac1K\sum_k\tilde z_k\in\{0,\tfrac13,\tfrac23,1\}$.
- **Binary:** $r_{\mathrm{bin}}=\mathbf{1}\{\tilde z=(1,1,1)\}$.
- Key cells: FP-only $(0.3,0)$ and FN-only $(0,0.3)$; both $J=0.7$. Clean $=(\alpha,\beta)=(0,0)$.

## Proposition

At matched $J$, FP-only and FN-only are approximately **exchangeable under fractional** rewards, but **not under binary** rewards ($K=3$). Binary makes FN much more harmful than FP; fractional largely erases that asymmetry.

**Why (binary, extremes).** Correct program (all pass) keeps $r_{\mathrm{bin}}=1$ under FN-only only with $(1-\beta)^K=0.7^3\approx 34\%$ — ~2/3 of correct answers scored 0. Fully failing wrong program gets $r_{\mathrm{bin}}=1$ under FP-only only with $\alpha^K=0.3^3\approx 3\%$. Same $J$; binary asymmetry is large.

**Why (fractional).** Both error types shrink expected score by a factor involving $J$; at matched $J$, FP≈FN is the null.

**Link.** Study A varies the reward map; Study B varies error type at matched $J$.

## Run order

**Do the crossed cells first**, then the rest of A/B. If Study B is run alone with only fractional train rewards, the natural finding is “FP ≈ FN” — Claim A of Prop. 1, silent on Claim B.

| Step | Experiments | Hypothesis |
|---|---|---|
| **0** | Offline: ~16 frozen answers / train problem; apply FP-only & FN-only offline under both aggregations | Same reward-level pattern as below, without RL; also screens the model |
| **1** | Clean (binary + fractional if needed; share one clean when identical) | Reference learning curve under noise-free reward |
| **2** | FP-only × fractional; FN-only × fractional ($J=0.7$) | **H1:** clean learning ≈ matched (FP ≈ FN) |
| **3** | FP-only × binary; FN-only × binary ($J=0.7$) | **H2:** FN worse than FP (lower / slower clean gains; more distortion) |

## Metrics

**Primary (pre-registered):** clean Pass@1 on the 500 MBPP test problems, average of the **last two** evaluations.

**Main contrasts for Prop. 1:**

| Contrast | Aggregation | Expected under Prop. 1 |
|---|---|---|
| FN-only − FP-only (same seed protocol, matched $J$) | Fractional | ≈ 0 (**H1**) |
| FN-only − FP-only | Binary | < 0 for FN (**H2**: FN harms more) |
| Each noisy cell − clean | Both | Secondary: size of damage vs clean |

**Supporting (same definitions as testbed notes):**

- Outcome label per run: preserved / delayed / distorted (rules locked before results).
- Connecting: clean–noisy advantage agreement; ranking fidelity; reward margins; gradient alignment on audited groups.
- Report primary metric **overall and by problem difficulty**.
- Offline: same reward / ranking / advantage stats on frozen groups — no optimiser.
