# Association-Aware CGIC Protocol

Status date: 2026-08-13  
Branch: `feature/sggt-v2-causal-innovation`  
Base commit recorded at handoff: `e192aa6` plus the current working tree

This document freezes the development protocol before any paper-scale run. It
does not contain or imply final paper conclusions.

## Method boundary

The learned contribution is a label-free, strictly causal group-innovation
motion-prior corrector (CGIC). Belief-propagation data association is an adopted
tracking backend and is not claimed as a contribution.

At prediction time, CGIC consumes only past information:

- the target's physical and whitened innovations;
- leave-one-out peer physical and whitened innovations from the historical
  graph;
- NIS and innovation covariance;
- optionally, soft-association detection mass, entropy, and candidate spread.

No current-frame measurement, ground-truth group label, or ground-truth
association identity is a neural input. Ground truth is used only by tracking
losses during supervised training.

The native main model uses `causal_innovation_alpha=1`. Learned dynamic
activation is an ablation, not the default main model.

## Partial-history rule

The legacy rule required all eight own and peer innovations to be valid. The
association-aware model is eligible when:

- at least 4 of the last 8 paired own/peer innovation frames are valid; and
- at least 1 of the most recent 3 frames is valid.

If the rule is not met, the correction is exactly zero and the model reduces to
the Temporal prior.

## Training augmentation

The development augmentation is causal and identity-consistent:

- sample target detection probability from `Uniform(0.75, 0.98)`;
- a missed detection strictly retains the prior and writes an invalid
  innovation frame;
- when the target is detected, sample a small number of clutter candidates near
  the generated target measurement;
- compute soft candidate probabilities from the detached current prior, its
  covariance, measurement covariance, and clutter model;
- use probability-data-association moment matching for the posterior update;
- feed detection mass, normalized association entropy, and candidate spread to
  CGIC when the association-context flag is enabled.

Global uniform clutter with mean `10--30` and cross-target one-to-one exclusion
are reserved for the complete BP tracker. They are not used in predictor
training because the training rollout has neither track-existence management
nor re-acquisition logic. Injecting them directly caused an artificial
miss--clutter positive-feedback loop and is rejected by the diagnostics below.

## Verified invariants

The following must remain true:

1. One candidate with probability one matches the original EKF state and
   covariance update.
2. A certain missed detection preserves the prior exactly.
3. Ambiguous candidates increase posterior covariance through mixture spread.
4. No-perturbation augmentation (`Pd=1`, no clutter) matches the original
   closed-loop training path.
5. Packed scenes do not exchange candidates or BP messages.
6. BP association input is a soft marginal, not a hard maximum-association
   measurement.
7. BP history stores `measurement - current prior`, not
   `measurement - posterior`.
8. The BP graph builder slides its state history once per frame.

Relevant tests:

- `tests/test_association_uncertainty.py`
- `tests/test_bp_sggt_adapter.py`
- `tests/test_causal_innovation_corrector.py`
- `tests/smoke_v3_pipeline.py`

## Development diagnostics

These are protocol diagnostics, not paper results. All values below use eight
training trajectories, zero learning rate, and the same Temporal checkpoint.

| Observation perturbation | Prediction RMSE (m) | Update RMSE (m) | Interpretation |
|---|---:|---:|---|
| None | 17.63 | 13.93 | Soft update degenerates correctly to EKF |
| Miss only (`Pd=0.85`) | 19.46 | 15.65 | Finite degradation from skipped updates |
| Local clutter only | 25.72 | 22.87 | Ambiguous local candidates are the main shift |
| Miss + local clutter | 21.93 | 17.85 | Final identity-consistent training augmentation |

Rejected variants:

- independent target likelihoods over other targets' detections: approximately
  300--370 m closed-loop prediction error;
- scene-level exclusion BP over other targets' detections under fixed-ID
  supervised loss: approximately 87.5 m;
- global clutter after missed detections without track management: above
  200 m.

These variants are not evidence against BP. They demonstrate that a fixed-ID
predictor-training rollout cannot substitute for a full tracker with existence,
birth/death, exclusion, pruning, and re-acquisition logic.

## Fixed association validation

When `--association_validation` is enabled, validation additionally runs the
same augmentation realization every epoch using
`--association_validation_seed 20260813`. Clean validation remains unchanged.
This deterministic curve, not stochastic training RMSE, determines whether the
new branch learns robustness.

## Experiment gate

Proceed from 32/16 to 200/50 only if:

- there are no NaNs or non-positive covariance failures;
- the fixed association-validation prediction RMSE trends downward;
- clean validation does not materially regress;
- CGIC correction RMS becomes non-zero without saturating its bound.

Proceed from 200/50 to formal three-seed training only if the same findings hold
on the larger development split.

### 32/16 gate result

The 10-epoch gate run completed at
`checkpoint/dev_assoc_cgic_fixedval_32_16` using seed 123.

| Metric | Epoch -1 | Epoch 9 | Change |
|---|---:|---:|---:|
| Fixed-association prediction RMSE (m) | 31.2089 | 31.1950 | -0.0139 |
| Fixed-association update RMSE (m) | 28.7367 | 28.7294 | -0.0073 |
| Clean prediction RMSE (m) | 18.8549 | 18.8318 | -0.0231 |
| Clean update RMSE (m) | 14.9313 | 14.9186 | -0.0127 |
| Training correction position RMS (m) | 0 | 0.0365 | +0.0365 |

The run is numerically stable and passes the development gate, but its effect
size is too small to support a paper claim. The next required step is the
pre-registered 200/50 development run.

### 200/50 gate result

The pre-registered run completed all 10 epochs without non-finite values. The
fixed-association checkpoint criterion selected epoch 5, not the clean metric's
epoch 9. Relative to immutable epoch -1, epoch 5 improved fixed-association
prediction/update RMSE by 0.4231/0.2312 m and clean prediction/update RMSE by
0.5804/0.3369 m. Correction position RMS/max was 1.0739/3.7165 m against a
50 m bound. Every machine check in
`results/association_cgic_200_50_gate.json` passed, so formal training is
approved. These remain development-gate values, not final paper results.

## Formal matrix after the gate

Core predictor experiments, three seeds (`123`, `456`, `789` unless the frozen
manifest says otherwise):

1. CV prior.
2. IMM prior.
3. Temporal learned prior.
4. CGIC own-innovation only.
5. CGIC own + peer innovation (main model).
6. Main model without association-uncertainty context.
7. Dynamic causal activation (seed 123 diagnostic).
8. Fixed activation `0.5` (seed 123 diagnostic).
9. ODE-Jacobian covariance propagation (seed 123 diagnostic); CV covariance is
   the default speed-controlled setting.

For each learned core model report clean IID, OOD, fixed association validation,
event-stratified errors, NLL/NIS calibration, parameters, and runtime. Use
trajectory-level paired bootstrap.

Full tracker experiments use frozen learned checkpoints and measurement Monte
Carlo seeds. Actual `Pd` is `{0.75, 0.85, 0.95}` and global clutter mean is
`{10, 20, 30}` while nominal tracker parameters remain fixed at `0.85/20`.
Matched-oracle tracker parameters belong only in the appendix. Report OSPA and
its localization/cardinality components, fragmentation, false/missed tracks,
and runtime. Complexity uses target counts `{5, 10, 20, 40}`.

### Implemented evaluation safeguards

- A neural motion prior is replicated over every active legacy group-partition
  hypothesis. The BP likelihood/message computation retains the hypotheses;
  only their motion prior is shared.
- The data-generating detection probability and clutter mean are independent
  of the tracker's fixed nominal assumptions. Adaptive Beta-Pd inference is
  disabled in the mismatched-parameter matrix.
- Full-tracker output separates OSPA localization/cardinality, matched state
  position/velocity RMSE, state NIS/NLL, missed and false tracks,
  fragmentation, identity switches, and runtime.
- CV and two-mode IMM evaluators use the identical V9 lifecycle population and
  online predict/update feedback. Hyperparameters are selected only on the
  validation set and frozen for IID/OOD tests.

A 60-frame BP smoke rollout completes without the former group-partition
index error. On that diagnostic trajectory, the adopted legacy tracker kept
only the first three confirmed tracks and missed later births. This is a track
management limitation, not evidence about CGIC motion-prior accuracy, and the
formal paper must disclose it instead of interpreting total OSPA alone.
