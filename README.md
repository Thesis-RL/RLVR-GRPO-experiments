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
│   └── offline_diagnostic.yaml # frozen-rollout diagnostic settings
│
├── data/
│   └── mbpp/                   # local MBPP arrow splits (train/val/test/prompt)
│                               # avoids HuggingFace cache locks; used by load_mbpp()
│
├── src/rlvr/                   # installable library
│   ├── data/mbpp.py            # MBPPProblem + load_mbpp()
│   ├── prompts/mbpp.py         # few-shot prompts + code extraction
│   ├── generate/sample.py      # HF model load + G completions / prompt
│   ├── exec/mbpp_executor.py   # clean unit-test grading (isolated subprocess)
│   ├── noise/corruption.py     # FP/FN / flip schedules
│   ├── rewards/aggregation.py  # binary, fractional, GRPO advantages
│   └── metrics/signal.py       # advantage agreement, ranking fidelity, strata
│
├── scripts/
│   ├── run_offline_diagnostic.py      # end-to-end: generate (+ optional grade/noise)
│   ├── grade_and_analyze_offline.py   # safe grade / analyze over saved rollouts (CPU)
│   └── _grade_one_problem_worker.py   # isolated child: exec one candidate under resource caps
│
├── tests/
│   └── test_offline_core.py    # noise / rewards / Prop.1 sanity (no GPU)
│
└── outputs/                    # gitignored run artifacts
    └── offline_diagnostic/
        └── <org>__<model>/
            ├── rollouts.jsonl         # prompts + raw completions + extracted code
            ├── clean_grades.jsonl     # per-test outcomes, clean rewards, stratum
            ├── difficulty.json        # per-problem difficulty buckets
            ├── signal_by_group.jsonl  # per-group metrics under each noise draw
            ├── report.json            # machine-readable summary
            └── report.md              # human summary (screening + Prop.1)
```

### What each package module does

| Module | Role |
|---|---|
| `rlvr.data` | Official MBPP splits (374 / 90 / 500 / 10). Prefers `data/mbpp/*.arrow`. |
| `rlvr.prompts` | Austin-style few-shot prompt; strips `[BEGIN]`/`[DONE]` / fences from completions. |
| `rlvr.generate` | Loads causal LM (bf16/fp16), samples `G` answers at train temperature. |
| `rlvr.exec` | Runs each candidate in a **separate** Python process with resource caps (address space, CPU time, process count). |
| `rlvr.noise` | Planned schedules: `clean`, `fp_only`, `fn_only`, `ind_flip_10`, `group_flip_10`, … |
| `rlvr.rewards` | `fractional` ∈ {0,⅓,⅔,1}, `binary` = all-tests-pass; GRPO within-group advantages. |
| `rlvr.metrics` | Clean–noisy advantage agreement / cosine, pairwise fidelity, top-1, margins, difficulty strata. |
| `rlvr.config` | Layered YAML loading plus validated model, reward, noise, seed, and condition selection. |
| `rlvr.run_logging` | Stable run IDs, resolved-config snapshots, metadata manifests, and JSONL events. |

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
validates names/rates and expands a deterministic condition order for the future
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
```

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
| `--config` | `configs/offline_diagnostic.yaml` | YAML with G, temp, noise list, etc. |
| `--limit` | all train | Cap number of train problems (smoke tests) |
| `--batch-size` | from config (`4`) | `num_return_sequences` chunk size; raise if VRAM allows (e.g. `8`) |
| `--dtype` | from config (`bfloat16`) | `bfloat16` / `float16` / `float32` |
| `--skip-generate` | off | Reuse existing `rollouts.jsonl` (no GPU) |
| `--out-dir` | `outputs/offline_diagnostic/<org>__<model>/` | Override output directory |

#### Config keys — `configs/offline_diagnostic.yaml`

| Key | Current | Meaning |
|---|---|---|
| `group_size` | `16` | Rollouts per problem (`G`) |
| `temperature` / `top_p` | `1.0` / `0.95` | Train-time sampling |
| `max_new_tokens` | `256` | Generation length |
| `batch_size` | `4` | Default chunk size (CLI overrides) |
| `noise_draws` | `3` | Independent noise resamples for offline metrics |
| `noise_base_seed` | `0` | Base seed for noise RNGs |
| `few_shot_task_ids` | `[2,3,4]` | Prompt examples (fallback: first 3 of prompt split) |
| `noise_specs` | clean, fp_only, fn_only, ind_flip_10, group_flip_10 | Schedules to apply offline |
| `aggregations` | fractional, binary | Reward maps for signal metrics |
| `dtype` | `bfloat16` | Model dtype |
| `split` | `train` | Which MBPP split to sample |

### 2) Grade + analyze (CPU)

Uses saved `rollouts.jsonl`. Each candidate code runs in an isolated subprocess (default `--mem-mb 64`). Resumes if `clean_grades.jsonl` already has some task_ids.

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

## Tests

```bash
uv run pytest tests/ -q
```

Covers fractional/binary maps, GRPO advantages, Prop.1 Monte Carlo sanity (FN hurts binary correct answers more than FP invents wrong ones), and signal buckets — no GPU.

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

Next (not implemented here yet): priority GRPO training block — Clean + FP-only + FN-only × binary/fractional (LoRA GRPO, group size 16).

---

## Notes index

| File | Contents |
|---|---|
| [`notes/testbed description.md`](notes/testbed%20description.md) | Shared vocabulary, Study A/B matrices, logging, run budget |
| [`notes/proposition 1.md`](notes/proposition%201.md) | Equal-$J$ FP vs FN under binary vs fractional |
| [`notes/pre_registration.md`](notes/pre_registration.md) | Primary metric, outcome cutoffs, offline signal buckets |
