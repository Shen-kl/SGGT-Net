# Workspace cleanup record - 2026-09-06

## Scope

This cleanup prepares the current SGGT-Net implementation, training entry
points, evaluation tools, tests, and manuscript-support files for version
control before a new GASF experiment is introduced.  No Python training or
evaluation process was active when the cleanup started.

## Removed

- Repository-local `.tmp` contents and Python bytecode caches.
- LaTeX intermediate files and a Visio lock file; the TeX, bibliography,
  template, figures, diagram sources, and locally built PDF are retained.
- MCST smoke and 200/50 short-run checkpoints and their smoke logs.
- The explicitly incomplete seed-789 checkpoint archive.
- Result directories or files explicitly named `smoke`, including BP,
  physical-baseline, association-context, and old simulation-scenario smoke
  outputs.

These files were development artifacts superseded by formal runs and can be
reproduced from the retained scripts.  The deleted untracked files are not
recoverable through Git.

## Retained local artifacts

- All formal SGGT-Net, Temporal, CGIC, MTP-GO, and MCST checkpoints.
- Formal IID/OOD, event-stratified, simulation-scenario, robustness, OSPA, and
  cardinality reports and curve aggregates.
- `checkpoint/mcst_v9_full_seed123`, the converged MCST comparison model.
- `checkpoint/formal_mtp_go_v1`, the native-process-noise MTP-GO baseline.

Generated checkpoints, logs, and reports are intentionally excluded from Git
by `.gitignore`; their lineage is documented in `checkpoint/README.md`, the
experiment protocol documents, and `docs/SGGT_V2_HANDOFF.md`.

## Version-control boundary

The commit following this cleanup contains source code, launch/evaluation
scripts, tests, protocol and audit documents, and manuscript-support sources.
It does not embed model weights, raw logs, or regenerated report trees.

