# Seed 0 shutdown archive audit

Audit date: 2026-10-24

Local archive root:

`/Users/rail/Diplom/RLVR-GRPO-experiments`

## Verification result

- Server `outputs/` versus local `outputs/`: 0 checksum mismatches.
- Server Hugging Face cache versus local `hf_cache/`: 0 checksum mismatches.
- Server RLVR home directories and logs versus the local auxiliary archive: 0 checksum mismatches.
- Server `.git` versus the local server Git copy: 0 checksum mismatches.
- Local SHA-256 manifest: 3,581 files verified, 0 failures.
- Local seed analyzer: 6 conditions complete, 6 valid, 0 errors, 0 warnings.
- The GPU server was not stopped, deleted, or modified through the provider panel.

## Main seed 0 artifacts

Path: `outputs/online_matrix/priority_bridge/main/seed-0/`

- Six conditions, with 525 files in each condition directory.
- 66 LoRA adapter weight files: ten checkpoints and one final adapter per condition.
- 60 optimizer states, 60 scheduler states, and 60 RNG states.
- 2,244 completion parquet files: 374 per condition.
- Six complete rollout logs containing corruption masks, clean/noisy outcomes and rewards, GRPO advantages, and zero-variance indicators.
- Validation and test predictions, trainer states, resolved configs, run metadata, resource reports, events, and the matrix status.
- Main execution log: `rlvr-main-seed0.log`.

## Base model

Portable cache path: `hf_cache/`

- Model: `Qwen/Qwen3-1.7B-Base`.
- Revision: `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`.
- The snapshot symlinks resolve inside the copied cache.
- Transformers loaded the config with `local_files_only=True`.
- Safetensors opened successfully and exposed 310 tensors.

Use the cache locally by setting:

```bash
export HF_HOME=/Users/rail/Diplom/RLVR-GRPO-experiments/hf_cache
```

## Additional preserved server data

Path: `outputs/online_matrix/priority_bridge/server_auxiliary/`

- Difficulty-generation and incomplete-test logs.
- Offline difficulty backup and current train/test difficulty artifacts.
- Stop-sequence and group-size preflight runs.
- Progress-validation and smoke-validation outputs.
- Exact server `.git` copy, including both stash reflog entries.
- A verified Git bundle for the newest stash ref.

Preserved stash entries:

- `56b732c1832850d11288dae36640c96bd61b1eb8` — `seed0-report-pre-sync`.
- `93a67df64309d9de39005f2cd2d1379106069a5b` — `server-pre-sync-frozen-512`.

## Integrity check

From the repository root:

```bash
shasum -a 256 -c reports/priority_bridge/seed-0/local_archive_SHA256SUMS.txt
```

Raw outputs and the model cache are intentionally ignored by Git. The checksum manifest and this audit report are tracked in `feat/rlvr-grpo-pipeline`.
