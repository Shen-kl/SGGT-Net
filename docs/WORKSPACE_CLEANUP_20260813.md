# Workspace cleanup record — 2026-08-13

## Purpose

Remove checkpoints and logs from rejected, failed, superseded, smoke, dev,
diagnostic and probe runs so that future paper experiments cannot accidentally
reuse an obsolete artifact. Source code, tests, paper evidence, formal reports
and the current reproducible training/evaluation lineage are retained.

No Python training or evaluation process was active when cleanup began.

## Retained checkpoint lineages

- `formal_innovation_v1`
- `causal_innovation_formal_seed123`
- `causal_innovation_gate_formal_seed123`
- `v2_formal_seed123`
- `paper_a0_lifecycle_fixed_scratch_seed123`
- `paper_a1_no_sean_gasf_lifecycle_fixed_converged_seed123`
- `paper_a2_sean_gasf_lifecycle_fixed_scratch_seed123`
- `paper_v3_shared_base_seed123`
- `paper_v3_temporal_only_gasf_scratch_seed123`
- `paper_v3_temporal_gasf_fusion_aux_scratch_seed123`
- `formal_cv_fixed_expert_full_seed123`
- `formal_cv_dynamic_full_seed123`
- `formal_cv_physical_shared_init_seed123`

The first three entries are the active causal-innovation lineage used by the
complete validation/IID/OOD comparisons. The remaining entries are retained
historical paper baselines or formal mechanism controls, not candidates for
silently initializing a new main-model run.

## Removed artifact classes

- All other direct children of `checkpoint/` (155 directories before cleanup,
  approximately 1.86 GB).
- The complete contents of `tmp/` (approximately 136 MB), including the
  four-trajectory SSE audit files after the audit passed.
- Logs whose stem does not belong to a retained checkpoint lineage, plus stale
  PID/stdout/stderr launcher files.
- Superseded smoke/dev/probe result JSON files. Full formal protocol, oracle,
  calibration, current validation controls, complete IID/OOD reports and
  trajectory bootstrap results are retained.
- Python `__pycache__` directories and `.pyc` files.

## Canonical current artifacts

- Temporal source checkpoint:
  `checkpoint/formal_innovation_v1/seed_123/A_temporal/`
- Full correction checkpoint:
  `checkpoint/causal_innovation_formal_seed123/`
- Dynamic activation checkpoint:
  `checkpoint/causal_innovation_gate_formal_seed123/`
- Complete IID/OOD reports:
  `evaluation_results/causal_activation_formal_seed123/`
- Paired trajectory bootstrap and summary:
  `results/causal_activation_formal_seed123/`
- Reproduction runner:
  `scripts/run_causal_activation_iid_ood_bootstrap.ps1`
- Long-term handoff:
  `docs/SGGT_V2_HANDOFF.md`

Deletion was performed only on resolved paths verified to be direct children
of the repository's `checkpoint`, `log`, `tmp`, `results`, or
`evaluation_results` directories. The external V9 dataset was not modified.
