# RLVR-GRPO experiments

- Work on the feature branch. Keep `main` unchanged.
- Read [README.md](README.md) and the [v2 analysis](analyses/priority_bridge_v2_2026-11-25/README.md).
- Treat the old 374-step `priority_bridge` and the 1,122-step `priority_bridge_v2`
  as different experiments. The latter completed seeds 0, 1 and 2. Full Study A
  and Study B remain separate.
- Keep archived inputs and run directories intact. Store derived results
  separately. Do not commit raw runs, model caches, or local deployment settings.
- Treat the endpoint contrast as uncertain with three training seeds. Describe the
  LoRA audit as a replay of 54 selected groups.
- GPU launches and new conditions require explicit user direction. On shared
  infrastructure, operate only this project's containers and processes.
