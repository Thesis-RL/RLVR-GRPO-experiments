# Where to find the experiment artifacts

Paths below are relative to the original experiment repository. The raw data
is stored alongside the checkout and transferred separately. The
[analysis package](../analyses/priority_bridge_v2_2026-11-25/README.md) contains
the compact statistics and figures.

| Path | Contents |
| --- | --- |
| `outputs/online_matrix/priority_bridge_v2/main/seed-{0,1,2}/` | Complete raw v2 run directories, approximately 11.9 GB combined |
| `.../seed-N/<condition>/rollouts.jsonl` | Prompts and completions, clean/noisy test outcomes, corruption masks, both rewards and advantages, actual training advantage, difficulty and group identifiers |
| `.../seed-N/<condition>/events.jsonl` and `trainer_state.json` | Step metrics, optimiser diagnostics, checkpoint and evaluation events |
| `.../seed-N/<condition>/completions/` | 20,196 Parquet files across the three seeds |
| `.../seed-N/<condition>/evaluations/` | Saved clean validation/test predictions and summaries |
| `.../seed-N/<condition>/checkpoint-*` and `final_adapter/` | 180 checkpoints with adapter, optimiser, scheduler and RNG state, plus 18 final LoRA adapters |
| `outputs/online_matrix/priority_bridge_v2/common-base/` | Matched untrained test reference, 4,000 rows |
| `outputs/online_matrix/priority_bridge_v2/main/gradient_audit_v1_1/` | 54 target records, summary and audit manifests |
| `outputs/online_matrix/priority_bridge_v2/container-logs/` | Five console log exports and matching Docker inspect records |
| `outputs/online_matrix/priority_bridge_v2/server_releases/` | Eight archived release bundles |
| `outputs/offline_diagnostic/`, `data/mbpp/`, `hf_cache/` | Frozen difficulty data, task splits and pinned base model |
