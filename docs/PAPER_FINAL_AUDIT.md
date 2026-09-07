# SGGT-Net paper revision gate audit

Date: 2026-08-19 (Asia/Shanghai)

Overall status: **DRAFT READY / SUBMISSION HOLD**

The revised manuscript and response letter are ready for author review. They
are not yet submission-ready because the core learned-model comparison has one
formal training seed and the supplied paper assets cannot currently be compiled.

## Completed and verified

- The formal result builder reconstructed 22 tracking reports, 240 paired
  bootstrap rows, 80 NIS rows, 18 BP model-condition rows, and 8 complexity
  rows with zero audit errors.
- The BP uncertainty analysis uses hierarchical paired resampling over scenes
  and paired Monte Carlo trials. It does not treat repeated trials from one
  scene as independent trajectories.
- Six SVG figures and six LaTeX tables are generated from audited CSV/JSON
  inputs rather than copied manually.
- The manuscript static audit recursively checked 11 authoritative LaTeX
  inputs, balanced braces, 24 labels, 10 references, and 17 citation keys. It
  reports zero structural errors.
- The revised method presents BP as an adopted soft-association front end and
  CGIC as the proposed strictly causal innovation correction. It does not claim
  group-label supervision and does not present SEAN/GASF/ODE/EKF as independent
  stacked innovations.
- The authoritative response letter states the protocol changes, weak evidence,
  and negative calibration result rather than hiding them.

## Evidence boundary used by the draft

- On validation, P4 versus P2 improves prior/posterior position RMSE by
  1.360/0.724 m; the trajectory-paired 95% intervals are [1.148, 1.561] and
  [0.603, 0.846] m.
- Prior-position improvements increase from steady (1.033 m) to switch
  (1.512 m) and strong-maneuver frames (1.717 m), which is consistent with the
  proposed correction target.
- The same prior-position direction holds on IID and OOD tests. The current
  learned-model values are all from seed 123 and are labelled as such.
- Across the nine BP conditions, P4 improves matched velocity RMSE and OSPA in
  8/9 cells. The hierarchical macro intervals are [0.048, 0.266] m/s and
  [0.147, 1.750], respectively. The matched-position macro interval crosses
  zero, and 95% NIS coverage is not improved.
- P4 versus P5 is small and mixed. Association-context input is therefore a
  secondary ablation, not an established contribution.

## Submission blockers

1. Seed 456 has a P2 checkpoint but no P4 checkpoint; seed 789 lacks the core
   P2/P4 pair. The final contribution strength must not be frozen until the
   user resumes this minimal multi-seed sequence.
2. `ref.bib` was not supplied in either the repository revision directory or
   the paper directory, so the 17 citation keys cannot be checked.
3. `IEEEtaes.cls` and a usable `pdflatex` installation are unavailable. No PDF
   compile or visual page-layout inspection has been performed.
4. The extreme BP cell (`Pd=0.75`, clutter mean 30) remains a documented
   regression and the association evaluation has only three generated scenes.

## Authoritative artifacts

- Plan: `write_plan.md`
- Manuscript entry: `docs/paper_revision/GNN_IEEE_REVISED_SINGLE.tex`
- Response letter: `docs/paper_revision/RESPONSE_LETTER_20260819.md`
- Result audit: `results/paper_revision_20260819/RESULT_AUDIT.json`
- Manuscript audit: `docs/PAPER_MANUSCRIPT_AUDIT_20260826.md`
- Multi-seed decision: `docs/PAPER_MULTISEED_DECISION_20260819.md`

The original paper in `SGGT-Net_paper` has not been overwritten. A synchronized
working copy is stored in its `revision_20260819` subdirectory.
