# Checkpoint index

The directory was curated on 2026-08-13. Do not initialize a new formal model
from an arbitrary historical checkpoint.

## Active causal-innovation lineage

- `formal_innovation_v1/seed_123/A_temporal/`: frozen Temporal source model.
- `causal_innovation_formal_seed123/`: full causal correction; current
  position-primary main checkpoint is `best_position.pth`.
- `causal_innovation_gate_formal_seed123/`: learned dynamic-activation
  mechanism ablation; current checkpoint is `best_position.pth`.

## Retained historical paper baselines

- `v2_formal_seed123/`
- `paper_a0_lifecycle_fixed_scratch_seed123/`
- `paper_a1_no_sean_gasf_lifecycle_fixed_converged_seed123/`
- `paper_a2_sean_gasf_lifecycle_fixed_scratch_seed123/`
- `paper_v3_shared_base_seed123/`
- `paper_v3_temporal_only_gasf_scratch_seed123/`
- `paper_v3_temporal_gasf_fusion_aux_scratch_seed123/`
- Other branches under `formal_innovation_v1/`

## Retained formal mechanism controls

- `formal_cv_fixed_expert_full_seed123/`
- `formal_cv_dynamic_full_seed123/`
- `formal_cv_physical_shared_init_seed123/`

## Retained external-comparison baselines

- `formal_mtp_go_v1/`: MTP-GO baseline trained without spectral or
  same-source innovation correction; use its native process-noise setting in
  the paper protocol.
- `mcst_v9_full_seed123/`: converged MCST single-target comparison checkpoint.

Checkpoint weights and generated histories are local artifacts and are not
stored in Git.  Their training and evaluation entry points are versioned under
`scripts/` and `evaluation/`.

See `docs/WORKSPACE_CLEANUP_20260813.md` for the deletion policy and
`results/causal_activation_formal_seed123/IID_OOD_SUMMARY.md` for current
formal results.
