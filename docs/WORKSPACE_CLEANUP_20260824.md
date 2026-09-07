# Workspace cleanup record - 2026-08-24

## Scope

The cleanup removed only artifacts that were explicitly identified as
development, diagnostic, smoke, cache, or temporary outputs. No external
dataset, formal checkpoint, paper audit result, historical comparison pickle,
source file, or final scene02 diagnosis artifact was removed.

## Removed

- 20 superseded direct checkpoint children matching the audited `dev_*` and
  `diag_*` list, including rejected causal-Q development checkpoints and
  association-CGIC probes.
- Matching `dev_*`, `diag_*`, and stale `q8_position_scan_5mc` log/PID files.
- Repository Python `__pycache__` directories.
- `.tmp/paper_preview`, the empty `tmp` directory, two V2 smoke reports, and
  four scene02 Q smoke reports.

The operation removed 65 exact targets and released 217.32 MB. All recursive
targets were resolved and verified to remain below
`E:\PycharmProjects\Neural Network\SGGT-Net` before deletion.

## Recoverability

The deleted development checkpoints and logs were untracked files and are not
recoverable through Git. Their validated conclusions are retained in:

- `debug_record.md`
- `docs/SGGT_V2_HANDOFF.md`
- `results/scene02_diagnosis_20260823/P4_q/CAUSAL_Q_VALIDATION_SUMMARY.md`

## Retained active lineage

- `checkpoint/formal_innovation_v1`
- `checkpoint/formal_association_cgic`
- the frozen historical paper checkpoints and comparison assets
- the complete paper-revision result audit

The new scene02 fixed-Q formal suite must write to a new dated result root and
must not reuse `results/paper_scene02_20260823`, whose measurement protocol is
different.

## Post-formal cleanup

After the fixed-Q suite passed its formal protocol audit, its superseded smoke
reports, smoke logs, ambiguous pre-mode robustness status file, and the
single-seed high-clutter diagnostic report were removed. The formal reports,
formal logs, final plots, structured curves, tables, status files, and artifact
hash manifest were retained under `results/paper_scene02_fixed_q_20260824`.

## Curve-storage revision (2026-08-26)

The previous suite's `CURVES.json`, `curves_long.csv`, and `curves.npz` were
deleted by explicit request. Its raw formal reports, tables, and PNG figures
were retained. A new aggregate-only suite under
`results/paper_scene02_mc_mse_20260826` stores per-frame Monte Carlo MSE curves
and global RMSE values without retaining per-run curve records.
