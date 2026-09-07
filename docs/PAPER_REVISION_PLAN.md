# New-Manuscript Revision Plan

This is the working skeleton for a new manuscript. Do not overwrite the
original `GNN - IEEE.tex`, and do not write comparative conclusions until the
frozen experiments are complete.

## Proposed positioning

Working title:

> Label-Free Causal Group-Innovation Learning for Maneuvering Group-Target
> Tracking

The title deliberately foregrounds the new motion-prior problem rather than the
adopted BP association backend. The abstract and system diagram must still say
explicitly that the prior is embedded in a BP tracker.

## Required manuscript structure

1. Introduction
   - Define the new problem: learning a causal group-motion correction without
     group labels and under uncertain associations.
   - Explain why module composition is not the claim.
   - State falsifiable contributions and their matching experiments.
2. Tracking problem and adopted BP backend
   - Define states, measurements, clutter, missed detections, lifecycle mask,
     and association variables before use.
   - Give a concise BP factorization and update outline.
   - Clearly mark BP as adapted background; enumerate only the integration
     changes needed to expose soft innovation statistics.
3. Causal group-innovation prior
   - Define strict prior innovation and whitening/NIS.
   - Define own/leave-one-out peer decomposition.
   - Define the partial-history mask and exact Temporal fallback.
   - Define bounded CGIC correction and alpha=1 native operation.
4. Association-uncertainty interface
   - Define detection mass, association entropy, and candidate spread from soft
     BP marginals.
   - Explain why hard maximum association is not used.
   - Separate training augmentation from full BP evaluation.
5. Training and inference
   - Full-time causal supervision, fixed data splits, random seeds, optimizer,
     learning-rate schedule, checkpoint criterion, and early stopping.
   - State that no group or association truth enters the model.
6. Experiments
   - Predictor IID/OOD and maneuver-event analysis.
   - Own-only versus own+peer counterfactual ablation.
   - Association-context ablation and NIS calibration.
   - Unified BP tracker results, robustness, and complexity.
   - Three-seed statistics and trajectory bootstrap.
   - Tune deterministic CV/IMM process noise on validation only, freeze it,
     then evaluate the untouched IID/OOD sets.
   - Decompose full-tracker OSPA and separately report missed/false tracks,
     fragmentation and identity switches; do not infer predictor quality from
     birth/confirmation failures.
7. Limitations
   - Simulation-only evidence.
   - Fixed/adopted association backend.
   - No claim of real-radar deployment until such data exist.

## Reviewer mapping

| Reviewer concern | Required response/evidence |
|---|---|
| Incremental module stacking | Reframe around a new causal, label-free innovation problem; own/peer counterfactual and oracle evidence |
| Simulated data only | Strong IID/OOD, mismatched `Pd`/clutter robustness, Monte Carlo uncertainty, explicit limitation |
| Weak baselines | CV/IMM, Temporal, own-only, full CGIC, uncertainty ablation, strong learned priors under the same BP backend |
| Unfair group labels | No group-structure supervision or group truth input; truth only in state loss |
| BP originality unclear | A dedicated concise background section and an explicit adopted-versus-modified table |
| Hard association in update | Soft BP marginal moments, association entropy, and spread; no max association in CGIC history |
| CV/ODE covariance mismatch | CV covariance as main speed-controlled protocol; ODE Jacobian as a named diagnostic |
| Missing reproducibility | Frozen manifests, checkpoint hashes, seeds, splits, commands, parameter/runtime accounting |
| Robustness used oracle parameters | Main tracker held at nominal `Pd=0.85`, clutter mean 20; matched oracle only appendix |
| Complexity/storage unclear | Real-V9 rollout timing, clearly labeled controlled 5/10/20/40 timing-only scaling, and explicit buffer/storage complexity |

## BP exposition decision

The association method should be explained, but compactly. A reader needs the
factor graph, association variable, likelihood factors, message-passing output,
and how those marginals enter the state/innovation interface. The manuscript
must not present standard BP derivations as a contribution. This is analogous
to RFS-based tracking papers that summarize the adopted RFS recursion before
describing the new component.

## Result-writing rule

Until the formal matrix is complete, tables should contain configuration IDs
and blank result cells only. Do not reuse old OSPA, robustness, or runtime
numbers because the current CGIC-to-BP integration and BP history semantics have
changed.
