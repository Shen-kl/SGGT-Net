# Frozen Paper Experiment Matrix

Status update (2026-08-19): the seed-123 formal predictor, calibration,
bootstrap, complexity, and full BP matrices are complete. The originally
pre-registered three-seed matrix remains incomplete: seed 456 has only its
Temporal checkpoint and seed 789 has no completed Temporal checkpoint. Values
must not be filled from older protocols.

Supplementary update (2026-08-26): the designed `scene02` fixed-Q suite is
complete for P2--P5 and for the requested observation-mismatch conditions.
Each cell contains 20 paired Monte Carlo realizations and 100 frames. The
retained curve assets are aggregate per-frame Monte Carlo MSE and final global
RMSE only; per-realization error curves were deliberately removed.

## Core predictor matrix

| ID | Prior / model | Peer channel | Association context | Seeds | Status |
|---|---|---:|---:|---|---|
| P0 | CV | no | no | deterministic | validation-tuned val/IID/OOD complete |
| P1 | two-mode IMM | no | no | deterministic | validation-tuned val/IID/OOD complete |
| P2 | Temporal | no | no | 123 complete; 456 checkpoint; 789 missing | seed-123 val/IID/OOD/association/NIS/BP complete |
| P3 | CGIC own-only | no | yes | 123 | val/IID/OOD/association and bootstrap complete |
| P4 | CGIC full | yes | yes | 123 | val/IID/OOD/association/NIS/BP/scaling complete |
| P5 | CGIC w/o association context | yes | no | 123 | val/IID/OOD/association and bootstrap complete |

Single-seed diagnostics: dynamic activation, fixed alpha 0.5, and ODE-Jacobian
covariance propagation. These do not replace the core ablations.

## Required evaluation axes

- Splits: validation, IID test, OOD test.
- Events: steady, maneuver switch, strong maneuver.
- Association conditions: clean and fixed augmented validation.
- State metrics: prediction/update position RMSE and velocity RMSE.
- Probabilistic metrics: Gaussian NLL, NIS coverage/calibration, association
  detection mass, entropy, and candidate spread.
- Mechanism metrics: correction RMS/max/saturation, history eligibility, and
  own-only versus own+peer paired gain.
- Efficiency: trainable/total parameters, peak GPU memory, real-V9 online
  rollout time, and a separate timing-only 5/10/20/40-target scaling test.
  Because V9 contains only 5--14 simultaneous targets, the 20/40-target rows
  replicate a real five-target causal window into disconnected graph blocks;
  they must never be described as accuracy evidence.
- Statistics: mean and standard deviation over seeds plus trajectory-paired
  bootstrap 95% confidence intervals.

## Full BP tracker matrix

The same frozen model checkpoint is used in every cell. The tracker keeps
nominal `Pd=0.85` and clutter mean `20`; the actual data-generating conditions
are crossed below.

| Actual Pd / clutter mean | 10 | 20 | 30 |
|---|---:|---:|---:|
| 0.75 | P2/P4 complete | P2/P4 complete | P2/P4 complete |
| 0.85 | P2/P4 complete | P2/P4 complete | P2/P4 complete |
| 0.95 | P2/P4 complete | P2/P4 complete | P2/P4 complete |

Report OSPA with localization/cardinality components, state RMSE on valid
matched targets, missed and false tracks, fragmentation, NLL/NIS, and runtime.
Matched-parameter oracle runs are appendix-only.

Implementation note: the complete BP evaluation preserves the network motion
prior for every legacy group-partition hypothesis instead of collapsing that
hypothesis axis. Actual detection/clutter generation is independent of the
tracker's fixed nominal `Pd=0.85` and clutter mean `20`. A full 60-frame smoke
test passes. Its high cardinality OSPA is caused by missed/unconfirmed
later-born targets in the adopted legacy track manager, so localization,
missed/false-track, fragmentation, and identity-switch metrics must always be
shown separately rather than attributing total OSPA to the learned prior.

## Designed scene02 supplementary matrix

The formal scene02 operating point uses the frozen P4 model, fixed CV
covariance propagation, fixed process-noise scale 6.874306, actual/nominal
`Pd=0.99`, actual/nominal clutter mean `2`, range-noise standard deviation
`3 m`, and azimuth-noise standard deviation `0.01 deg`.

| Branch | Actual conditions | Status |
|---|---|---|
| Ablation | P2, P3, P4, P5 at `Pd=0.99`, clutter `2` | complete |
| Detection robustness | clutter `3`; `Pd=0.99, 0.90, 0.80` | complete |
| Clutter robustness | `Pd=0.99`; clutter `2, 10, 15` | complete |

Legacy MCST--BP, MTP--GO, and BP--R--RGTT curve assets were re-aggregated with
the same Monte Carlo MSE formula for descriptive continuity only. They retain
the original-paper observation protocol and fewer Monte Carlo runs, so they
must not be presented as a paired or protocol-matched ranking against P4.

## Physical-baseline selection rule

P0 and P1 use the same known-identity V9 predictor protocol as the learned
models: one online prediction and measurement update at every valid frame,
posterior feedback, birth-frame initialization without scoring, and no scoring
or update when `real_mask=0`. CV acceleration noise and the two-mode IMM's
low/high acceleration noise plus persistence are selected once on validation
prediction+update position RMSE, recorded in
`physical_baseline_tuning.json`, then frozen before IID/OOD evaluation. No test
split participates in this selection.

## Evidence gate

The 200/50 development run must show all of the following before P3--P5 are
launched formally:

1. finite losses and positive-semidefinite covariances;
2. downward fixed-association validation prediction RMSE;
3. no material clean-validation regression;
4. non-zero, non-saturated correction magnitude;
5. a practically meaningful effect, not merely a numerically resolved change.
