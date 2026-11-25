# Run provenance and supporting analyses

| Stage | Evidence |
| --- | --- |
| Verifier v2.1 | [Linux/Docker reference gate](verifier_v2_1_docker_gate.json) |
| Protocol selection | [Validation-only pilot results and decision](stage5_pilot_gate.json) |
| Stage 6 | [Original three-seed endpoint and training-signal analysis](stage6_multiseed_reference.json) |
| LoRA gradient audit | [Audit completeness check](gradient_audit_verification.md), [frozen audit protocol](../../../docs/GRADIENT_AUDIT_PROTOCOL.md) |

The training runs used archived Docker release bundles. These names resolve
under `outputs/online_matrix/priority_bridge_v2/server_releases/` in the raw
archive.

| Job | Release bundle |
| --- | --- |
| Verifier v2.1 gate | `20261116-verifier-v2.1-stage5-v2` |
| Clean pilot and protocol selection | `20261116-stage5-pilot-gate-v3` |
| v2 seed 0 | `20261117-stage6-seed0-v1` |
| v2 seeds 1 and 2 | `20261119-stage6-seeds12-v1` |
| LoRA audit | `20261123-gradient-audit-v1_1` |

Before preparing the branch, the current v2 config and training runner matched
both Stage 6 releases byte for byte. The LoRA audit script matched its executed
v1.1 release. The PR applies formatting to some files without changing the
frozen settings or analysis logic. Paths in the audit config and the reference
analysis now resolve relative to the checkout. The release directories retain
the exact historical sources and their hashes.

| File | Executed SHA-256 |
| --- | --- |
| `configs/priority_bridge_v2.yaml` | `fe4870247f350873f838802844cd8250f2b13c207bdf25fd1929b0cccdba4190` |
| `scripts/run_online_grpo.py` | `6596e28a4fd7766ae371f38db44a2afcf05a6bafb6e7e0f1696a9f2a44813895` |
| `scripts/audit_lora_gradients.py` | `35850279ced3011c3d7c30ddda4e562281c29f2bf2f6784e04ea08cbd28b7a59` |

The [v2 archive manifest](main_archive_trees.json) records the original
counts and hashes for the listed data trees. The [progress report](../PROGRESS_REPORT_EN.md)
summarizes the earlier one-pass evaluation that motivated the v2 design.
