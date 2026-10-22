# RLVR-GRPO experiments

## Repository structure

```
RLVR-GRPO-experiments/
├── README.md
├── pyproject.toml              # package + deps (uv)
├── uv.lock
├── .gitignore
│
├── notes/                      # thesis / experiment design (not runtime code)
│   ├── testbed description.md  # shared testbed, matrices, logging
│   ├── proposition 1.md        # FP vs FN × binary/fractional claim
│   └── pre_registration.md     # locked metrics & thresholds (before results)
│
├── configs/
│   ├── shared.yaml             # model, data, generation, eval, logging
│   ├── study_a.yaml            # binary vs fractional × shared-noise matrix
│   ├── study_b.yaml            # fractional × matched-J FP/FN matrix
│   ├── priority_bridge.yaml    # clean / FP / FN × both rewards (run first)
│   ├── offline_diagnostic.yaml # frozen-rollout diagnostic settings
│   └── test_difficulty.yaml    # frozen initial-checkpoint test strata
│
├── data/
│   └── mbpp/                   # local MBPP arrow splits (train/val/test/prompt)
│                               # avoids HuggingFace cache locks; used by load_mbpp()
│
├── src/rlvr/                   # installable library
│   ├── data/mbpp.py            # MBPPProblem + load_mbpp()
│   ├── prompts/mbpp.py         # few-shot prompts + code extraction
│   ├── generate/sample.py      # HF model load + G completions / prompt
│   ├── exec/mbpp_executor.py   # clean grading (credential-scrubbed, resource-capped child)
│   ├── noise/corruption.py     # FP/FN / flip schedules
│   ├── rewards/aggregation.py  # binary, fractional, GRPO advantages
│   ├── metrics/signal.py       # advantage agreement, ranking fidelity, strata
│   └── online/                 # GRPO dataset, noisy reward, callbacks, clean eval
│
├── scripts/
│   ├── run_online_grpo.py           # one online condition / seed, smoke and resume
│   ├── run_online_matrix.py         # sequential condition matrix + aggregate status
│   ├── show_progress.py             # readable run / matrix progress snapshots
│   ├── run_offline_diagnostic.py      # end-to-end: generate (+ optional grade/noise)
│   ├── grade_and_analyze_offline.py   # safe grade / analyze over saved rollouts (CPU)
│   └── _grade_one_problem_worker.py   # isolated child: exec one candidate under resource caps
│
├── tests/
│   ├── test_offline_core.py    # noise / rewards / Prop.1 sanity (no GPU)
│   ├── test_online_reward.py   # online reward path and config guards (no GPU)
│   ├── test_online_runtime.py  # resume, evaluation, difficulty, schedules (no GPU)
│   └── test_online_matrix.py   # completed-run reuse and matrix ETA scope (no GPU)
│
└── outputs/                    # frozen diagnostics + gitignored online runs
    └── offline_diagnostic/
        └── <org>__<model>/
            ├── rollouts.jsonl         # prompts + raw completions + extracted code
            ├── clean_grades.jsonl     # per-test outcomes, clean rewards, stratum
            ├── difficulty.json        # per-problem difficulty buckets
            ├── signal_by_group.jsonl  # per-group metrics under each noise draw
            ├── report.json            # machine-readable summary
            └── report.md              # human screening or difficulty summary
```

### What each package module does

| Module | Role |
|---|---|
| `rlvr.data` | Official MBPP splits (374 / 90 / 500 / 10). Prefers `data/mbpp/*.arrow`. |
| `rlvr.prompts` | Austin-style few-shot prompt; strips `[BEGIN]`/`[DONE]` / fences from completions. |
| `rlvr.generate` | Loads causal LM (bf16/fp16), samples `G` answers at train temperature. |
| `rlvr.exec` | Runs each candidate in a separate temporary working directory with a scrubbed environment and resource-capped Python child. This is defense in depth, not a VM-grade security boundary. |
| `rlvr.noise` | Planned schedules: `clean`, `fp_only`, `fn_only`, `ind_flip_10`, `group_flip_10`, … |
| `rlvr.rewards` | `fractional` ∈ {0,⅓,⅔,1}, `binary` = all-tests-pass; GRPO within-group advantages. |
| `rlvr.metrics` | Clean–noisy advantage agreement / cosine, pairwise fidelity, top-1, margins, difficulty strata. |
| `rlvr.config` | Layered YAML loading plus validated model, reward, noise, seed, and condition selection. |
| `rlvr.run_logging` | Stable run IDs, resolved-config snapshots, metadata manifests, and JSONL events. |
| `rlvr.online` | TRL dataset, clean execution + noisy reward, trainer events, and scheduled clean evaluation. |

### Shared experiment configs

`configs/shared.yaml` is the common testbed contract. The two thesis studies and
the priority bridge extend it and only replace their experiment identity and
condition matrix:

| Config | Expanded conditions per seed |
|---|---:|
| `configs/study_a.yaml` | 3 verification conditions × 2 reward maps = 6 |
| `configs/study_b.yaml` | 5 matched/asymmetric FP/FN conditions × fractional = 5 |
| `configs/priority_bridge.yaml` | clean / FP-only / FN-only × binary / fractional = 6 |

Noise rates live in the study YAML rather than in training code. Named legacy
noise schedules remain supported by the offline diagnostic. The config loader
validates names/rates and expands a deterministic condition order for the
online runner. Every runner can use `rlvr.run_logging.write_run_manifest()` to
store `resolved_config.yaml` and `run_metadata.json` before expensive work.

---

## Setup

Requirements:

- Python ≥ 3.11
- [`uv`](https://github.com/astral-sh/uv)
- CUDA-capable GPU + working PyTorch CUDA build for **generation** (grading and analysis are CPU-only)

```bash
git clone <repo-url> RLVR-GRPO-experiments
cd RLVR-GRPO-experiments
uv sync
uv run pytest tests/ -q
uv run ruff check .
uv run ruff format --check .
```

Install the additional training stack only in an environment that will run
online GRPO:

```bash
uv sync --extra train
```

The `train` extra installs TRL and PEFT. GPU-specific accelerators such as
vLLM are deliberately not forced into the local macOS environment.

Optional environment variables:

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | Hugging Face Hub auth / higher rate limits when downloading models or datasets |
| `CUDA_VISIBLE_DEVICES` | Select which GPU(s) the generation step may use |

Check that PyTorch sees a GPU before generating:

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

---

## Offline diagnostic — how to run

Pre-registered rules (do not change after looking at numbers): [`notes/pre_registration.md`](notes/pre_registration.md).

Recommended workflow: generate on GPU, then grade/analyze on CPU (isolated per-code workers).

### 1) Generate rollouts (GPU)

```bash
# Smoke (4 train problems × 16 samples)
uv run python scripts/run_offline_diagnostic.py \
  --model Qwen/Qwen3-1.7B-Base \
  --role main_candidate \
  --limit 4 \
  --batch-size 8 \
  --dtype bfloat16

# Full training split (374 problems × 16)
uv run python scripts/run_offline_diagnostic.py \
  --model Qwen/Qwen3-1.7B-Base \
  --role main_candidate \
  --batch-size 8 \
  --dtype bfloat16
```

`run_offline_diagnostic.py` can also grade+analyze in one go; the split below is useful when you already have `rollouts.jsonl` or want to resume grading.

#### Parameters — `run_offline_diagnostic.py`

| Flag | Default | Meaning |
|---|---|---|
| `--model` | selected from config | Optional HF model override, e.g. `Qwen/Qwen3-1.7B-Base` |
| `--role` | selected from config | Optional report label override (`main_candidate` / `pipeline_debug` / …) |
| `--revision` | selected from config | Immutable HF revision (required for the test split) |
| `--config` | `configs/offline_diagnostic.yaml` | YAML with G, temp, noise list, etc. |
| `--limit` | entire configured split | Cap number of problems (smoke tests) |
| `--batch-size` | from config (`4`) | `num_return_sequences` chunk size; raise if VRAM allows (e.g. `8`) |
| `--dtype` | from config (`bfloat16`) | `bfloat16` / `float16` / `float32` |
| `--skip-generate` | off | Reuse existing `rollouts.jsonl` (no GPU) |
| `--out-dir` | `outputs/offline_diagnostic/<org>__<model>[/<split>]/` | Override output directory |

#### Config keys — `configs/offline_diagnostic.yaml`

| Key | Current | Meaning |
|---|---|---|
| `group_size` | `16` | Rollouts per problem (`G`) |
| `temperature` / `top_p` | `1.0` / `0.95` | Train-time sampling |
| `max_new_tokens` | `256` | Generation length |
| `batch_size` | `4` | Default chunk size (CLI overrides) |
| `noise_draws` | `3` | Independent noise resamples for offline metrics |
| `noise_base_seed` | `0` | Base seed for noise RNGs |
| `generation_seed` | `0` | Per-task deterministic generation seed base |
| `few_shot_task_ids` | `[2,3,4]` | Prompt examples (fallback: first 3 of prompt split) |
| `noise_specs` | clean, fp_only, fn_only, ind_flip_10, group_flip_10 | Schedules to apply offline |
| `aggregations` | fractional, binary | Reward maps for signal metrics |
| `dtype` | `bfloat16` | Model dtype |
| `split` | `train` | Which MBPP split to sample |

### Freeze test difficulty before main runs

The frozen map for the current Qwen3-1.7B-Base revision is already present and
validated for all 500 held-out test problems. If the model revision changes,
copy the new immutable revision into `model.revision` in `configs/shared.yaml`
and `model_revision` in `configs/test_difficulty.yaml`, then generate a new
initial-checkpoint map from 16-answer groups:

```bash
uv run python scripts/run_offline_diagnostic.py \
  --config configs/test_difficulty.yaml
```

This writes `difficulty.json` and `difficulty_metadata.json` under
`outputs/offline_diagnostic/Qwen__Qwen3-1.7B-Base/test/`. Main training checks
the model ID, immutable revision, split, and complete 500-task coverage before
starting. A fresh test generation refuses to overwrite this directory. These
strata use the fractional-reward thresholds locked in `pre_registration.md`;
they are for reporting only and must not be used for tuning.

### 2) Grade + analyze (CPU)

Uses saved `rollouts.jsonl`. Each candidate code runs in a credential-scrubbed,
resource-capped subprocess (default `--mem-mb 64`). Resumes if
`clean_grades.jsonl` already has some task_ids.

```bash
# Grade only (resume-friendly)
uv run python scripts/grade_and_analyze_offline.py \
  --model Qwen/Qwen3-1.7B-Base \
  --mem-mb 64 \
  --timeout-s 0.5 \
  --grade-only

# Then build report / Prop.1 deltas (no code execution)
uv run python scripts/grade_and_analyze_offline.py \
  --model Qwen/Qwen3-1.7B-Base \
  --analyze-only

# Or both in one command
uv run python scripts/grade_and_analyze_offline.py \
  --model Qwen/Qwen3-1.7B-Base \
  --mem-mb 64 \
  --timeout-s 0.5
```

#### Parameters — `grade_and_analyze_offline.py`

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `Qwen/Qwen3-1.7B-Base` | Must match folder name under `outputs/…` |
| `--role` | `main_candidate` | Written into `report.json` |
| `--revision` | selected from config | Immutable HF revision (required for the test split) |
| `--config` | `configs/offline_diagnostic.yaml` | Noise specs / draws / aggregations |
| `--out-dir` | auto from model id | Directory containing `rollouts.jsonl` |
| `--mem-mb` | `64` | Per-code child address-space cap (MiB) |
| `--timeout-s` | `0.5` | Per body / per-test timer inside the worker |
| `--grade-only` | off | Stop after `clean_grades.jsonl` |
| `--analyze-only` | off | Only recompute `report.*` / `signal_by_group.jsonl` from grades |

### 3) Read results

```bash
# macOS / Linux
cat outputs/offline_diagnostic/Qwen__Qwen3-1.7B-Base/report.md
# Windows (PowerShell)
Get-Content outputs/offline_diagnostic/Qwen__Qwen3-1.7B-Base/report.md
```

Screening rule (from pre-registration): main model is eligible if mean fractional reward ∈ **[0.05, 0.60]** and ≥ **15%** of problems have nonzero within-group clean reward variance.

---

## Online GRPO — priority bridge

The first online block uses `configs/priority_bridge.yaml`: clean, FP-only
30%, and FN-only 30%, each with binary and fractional reward. Each command
runs exactly one condition and one predeclared seed.

List the six condition names:

```bash
uv run python scripts/run_online_grpo.py --list-conditions
```

Validate config, execution, noise, rewards, advantages, and logging without
loading a model:

```bash
uv run python scripts/run_online_grpo.py \
  --condition fp_only__fractional \
  --seed 0 \
  --dry-run \
  --limit 2
```

Run the short end-to-end GPU smoke pilot (two training steps and two clean
evaluation problems by default):

```bash
uv run python scripts/run_online_grpo.py \
  --condition clean__fractional \
  --seed 0 \
  --smoke
```

Resume its latest checkpoint:

```bash
uv run python scripts/run_online_grpo.py \
  --condition clean__fractional \
  --seed 0 \
  --smoke \
  --resume
```

Useful controls are `--limit`, `--max-steps`, `--eval-limit`,
`--skip-clean-eval`, `--model`, `--dtype`, and `--out-dir`. vLLM can be
enabled explicitly with `--use-vllm` after it is installed in the cloud GPU
environment. These overrides are pilot-only: a frozen main run refuses them so
that all six conditions use exactly the YAML protocol.

Run directories are immutable by default. Re-running a command into a non-empty
directory fails instead of appending mixed observations. `--resume` validates
the saved context and resolved config, locates the latest checkpoint, preserves
the original manifest, and appends a resume event.

The main profile is frozen to the immutable Qwen3-1.7B-Base revision and a
374-step budget (one 16-completion group per MBPP train problem). Main runs
record the current Git commit and dirty state for provenance, but do not require
a clean worktree. They do require the frozen test-set difficulty map configured
at `data.test_difficulty_path`.

Each run writes its resolved config and metadata before training, then records
rollouts, clean/noisy test outcomes, corruption masks, both reward variants,
selected training reward, advantages, zero-variance groups, optimizer events,
checkpoints, and final clean evaluation under:

```
outputs/online_grpo/priority_bridge/<profile>/<model>[__rev-<revision>]/<condition>/seed-<n>/
```

The manifest includes Git state, package versions, platform, CUDA and GPU
details. `resource_report.json` records peak GPU/RAM use, parent and grading
subprocess CPU time, generated completion tokens, disk size, and stage timings;
`resource_report_attempts.jsonl` preserves every failed or resumed attempt.
Every rollout includes a policy step and verifier-call index; evaluation writes
one resumable JSONL row per task without changing the training RNG stream.

Each online run also atomically refreshes `status.json`. It records the current
stage, optimizer step, total steps, epoch, elapsed time, observed seconds per
step, scoped ETA, expected finish time, latest checkpoint, recent metrics, and
task-level clean-evaluation progress. The ETA is explicitly labelled as
training-only or training plus the evaluation currently in progress; future
scheduled evaluations are not silently included. Read the latest run with:

```bash
uv run python scripts/show_progress.py
```

Use `--out-dir <run-directory>` for one known run or `--all` to list every
recorded run. The command also reports whether the recorded process is still
alive on the current host.

Run a condition matrix sequentially with an aggregate `matrix_status.json`:

```bash
uv run python scripts/run_online_matrix.py \
  --config configs/priority_bridge.yaml \
  --seed 0
```

The matrix status records completed, current, and pending conditions plus an
explicitly scoped ETA projection. `--resume` resumes an interrupted condition
from its latest valid checkpoint and reuses a completed condition only after
its experiment, profile, condition, seed, metadata, and Git commit have been
validated. In a remote session, launch this command through a persistent
process manager (`tmux`, `systemd`, or equivalent) so an SSH disconnect cannot
stop the experiment.

For a frozen main run, clean validation runs every 10% of the optimizer budget.
Clean test evaluation runs at 90% and 100%; `primary_metric_summary.json`
averages these last two Pass@1 values. Difficulty-stratified test scores are
included when the frozen test difficulty map is available.

---

## Tests

```bash
uv run pytest tests/ -q
```

Covers fractional/binary maps, GRPO advantages, Prop.1 Monte Carlo sanity (FN
hurts binary correct answers more than FP invents wrong ones), signal buckets,
the online reward path, immutable run directories, exact resume validation,
resumable evaluation, difficulty maps, and evaluation schedules — no GPU.

---

## Current artifacts (example)

After a full Qwen3-1.7B-Base diagnostic:

```
outputs/offline_diagnostic/Qwen__Qwen3-1.7B-Base/
  rollouts.jsonl      # 374 × 16 frozen answers
  clean_grades.jsonl
  difficulty.json
  signal_by_group.jsonl
  report.json / report.md
```

The priority GRPO runner and main hyperparameters are frozen, and the
initial-checkpoint test difficulty map has been generated and validated for all
500 test problems. The next operational step is seed 0 of the six-condition
priority matrix.

---

## Notes index

| File | Contents |
|---|---|
| [`notes/testbed description.md`](notes/testbed%20description.md) | Shared vocabulary, Study A/B matrices, logging, run budget |
| [`notes/proposition 1.md`](notes/proposition%201.md) | Equal-$J$ FP vs FN under binary vs fractional |
| [`notes/pre_registration.md`](notes/pre_registration.md) | Primary metric, outcome cutoffs, offline signal buckets |
