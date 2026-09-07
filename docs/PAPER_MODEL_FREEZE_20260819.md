# Paper Model and Protocol Freeze — 2026-08-19

This document is the authoritative mapping between the current implementation,
formal experiment IDs, checkpoints, and manuscript terminology. It supersedes
older development descriptions when they conflict with the artifacts below.

## Provenance and evidence boundary

- Branch: `feature/sggt-v2-causal-innovation`
- Git HEAD: `e192aa680069673ec93bd47d153f989fcc6bf930`
- Formal protocol:
  `results/formal_association_cgic_protocol_v1.json`
- Formal protocol SHA-256:
  `63b0ee5c33d4a8df6ad4c8802293def1e61ce737cd8ba319f12be95a6f1b917e`
- Dataset: `GroupTargetsDataset_dataRate_2_v9_pack`
- Train/validation/IID/OOD trajectories: 1000/200/300/400
- Formal protocol declared seeds: 123, 456, 789
- Completed CGIC formal seeds at this freeze: **123 only**
- Seed 456 has a completed Temporal checkpoint but no P3/P4/P5 checkpoints.
- Seed 789 has no completed formal Temporal checkpoint.

The working tree intentionally contains the implementation and evaluation
repairs recorded by the formal protocol manifests. The two post-freeze,
evaluation-only amendments are:

1. `results/formal_association_cgic/protocol_amendment_nis_fix_20260817.json`
2. `results/formal_association_cgic/protocol_amendment_bp_partition_fix_20260818.json`

Neither amendment changes training, checkpoints, dataset splits, or the model
definition.

## Immutable experiment IDs

| ID | Definition | Learned peer channel | Association context | Status |
|---|---|---:|---:|---|
| P0 | validation-tuned constant-velocity Kalman model | no | no | deterministic val/IID/OOD complete |
| P1 | validation-tuned two-mode IMM | no | no | deterministic val/IID/OOD complete |
| P2 | frozen Temporal SGGT-V2 predictor | no | no | seed 123 formal complete; seed 456 checkpoint only |
| P3 | P2 plus causal own-innovation corrector | no | yes | seed 123 formal complete |
| P4 | P2 plus causal own and leave-one-out peer innovation corrector | yes | yes | seed 123 formal complete; main model |
| P5 | P4 without association-uncertainty context | yes | no | seed 123 formal complete |

P0 uses acceleration standard deviation 1.0. P1 uses low/high acceleration
standard deviations 0.25/3.0 and diagonal transition probability 0.98. These
values were selected once on the 200-trajectory validation set using prediction
plus update position RMSE. No IID/OOD test sample participated in selection.

## Frozen seed-123 checkpoints

| ID | Selected checkpoint | Epoch | SHA-256 |
|---|---|---:|---|
| P2 | `checkpoint/formal_innovation_v1/seed_123/A_temporal/GNN_groupTargetTracking/best_prediction.pth` | 48 | `c2c0dfdde7ccef1a7d892459c1e4a1a5d5c96cbc539f771ab09e862f9d359022` |
| P3 | `checkpoint/formal_association_cgic/seed_123/P3_own_only/GNN_groupTargetTracking/best_association_position.pth` | 0 | `18c471b6b45648f4d0b63641fb9f5341c61deb030ccaa79fb29441f00cc27da4` |
| P4 | `checkpoint/formal_association_cgic/seed_123/P4_full/GNN_groupTargetTracking/best_association_position.pth` | 13 | `6177ab5834c0388c74d8d75e5bdbfb90c56f41bc6ce4baf1ad07258cb0385581` |
| P5 | `checkpoint/formal_association_cgic/seed_123/P5_no_association_context/GNN_groupTargetTracking/best_association_position.pth` | 8 | `ca6498fb5e9c302fdd42bfed6696030aba49fa489b78252ba756532b3cbeb395` |

The P3 epoch-0 selection is a result, not a labeling error: its validation
checkpoint criterion found no later checkpoint better than epoch 0. It must be
reported transparently because it supports the conclusion that own innovation
alone provides negligible incremental value under this protocol.

## P2 Temporal backbone

The manuscript definition of **Temporal** is exactly:

- model variant `sggt_v2`;
- eight-frame causal input and one-step prediction output;
- streaming encoder execution after the initial history window;
- one decoder call per predicted frame;
- GASF enabled;
- fixed `a,b,c` filter mode;
- `temporal_only` GASF mode, with no graph-frequency propagation;
- feature-level concatenation fusion;
- SEAN disabled;
- old innovation decomposition disabled;
- CV covariance transition;
- legacy adaptive-Q path retained;
- no causal innovation corrector.

P2 contains 317,540 model parameters according to the frozen scaling report.

## P4 main model

P4 keeps the entire P2 Temporal predictor frozen and trains only the causal
group-innovation corrector. The corrector adds 17,669 trainable model
parameters, giving 335,209 total model parameters.

At prediction step `t`, P4 uses only histories formed before the current
measurement:

- eight frames of the target's own raw innovation;
- eight frames of leave-one-out peer raw innovation formed on the historical
  geometric graph;
- own and peer whitened innovations;
- NIS, innovation covariance diagonal/correlation, and validity masks;
- soft-association detection mass, normalized entropy, and candidate-spread
  eigenvalue summaries.

Own and peer raw motion channels have independent encoders. Uncertainty
features condition their fusion but do not mix the two raw channels before
encoding. No current-frame measurement, ground-truth group label, or
ground-truth association identity is a neural input.

Correction eligibility requires:

- at least four valid paired own/peer innovation frames in the last eight; and
- at least one valid paired frame in the most recent three.

Otherwise the correction is exactly zero and P4 reduces to P2. The native
formal model uses fixed `alpha=1`; learned dynamic activation is not used. The
normalized correction is bounded to the equivalent of 50 m in position and
5 m/s in velocity, and is added at state-feature level to the Temporal prior
before the EKF measurement update.

## P3 and P5 counterfactuals

- P3 has the association-context inputs but zeros the peer-innovation channel.
  It tests whether the improvement can be obtained from the target's own past
  innovation alone.
- P5 retains own and peer innovation but removes detection mass, entropy, and
  candidate spread. It tests the incremental value of association context.

The primary mechanism contrast is P4 versus P3 (aligned peer information). P4
versus P5 is a smaller, secondary association-context contrast and must not be
overstated if its paired confidence interval includes zero.

## Training and inference protocol

- Full-time supervision over all legal causal target steps.
- `real_mask=1` means the target exists at the scored frame; `real_mask=0`
  frames do not contribute to loss, RMSE, or measurement-update evaluation.
- Batch size 4; streaming encoder; trajectory-batch optimizer steps.
- AdamW, maximum learning rate `1e-4`, minimum `3e-6`, one warmup epoch,
  cosine decay, and weight decay `1e-4`.
- Gradient clipping norm 5.0.
- Maximum 30 epochs; early-stopping patience 6 and minimum position improvement
  0.05 m.
- P3/P4/P5 initialize from the same-seed frozen P2 checkpoint.
- Training stage `causal_innovation_expert` freezes the Temporal backbone and
  trains the corrector.
- Checkpoint selection uses deterministic fixed-association validation
  prediction plus update position performance.
- Training augmentation samples detection probability in `[0.75, 0.98]` and
  uses identity-consistent local clutter only. Global clutter is reserved for
  the complete BP tracker.

## Adopted BP tracker boundary

Belief-propagation association is an adopted backend, not a claimed algorithmic
contribution. It supplies soft association marginals, missed-detection mass,
entropy, candidate spread, and the strict prior innovation used by CGIC.

Formal BP robustness keeps tracker parameters fixed at nominal `Pd=0.85` and
clutter mean 20 while the data-generating values cross actual
`Pd={0.75,0.85,0.95}` and clutter `{10,20,30}`. Seed-123 P2 and P4 each have
all nine conditions, three trajectories per condition, and 20 measurement
Monte Carlo runs per trajectory.

The BP adapter repair replicates the shared neural state prior across every
active Gaussian-mixture component and current group-partition hypothesis. This
is an evaluation integrity repair, not a model contribution.

## Explicitly excluded from the final main-method claim

The following appeared in earlier development but are not part of the frozen
P4 main model:

- supervised group-structure labels or structure loss (`lambda_struct=0`);
- SEAN graph prediction;
- graph-frequency propagation inside GASF;
- old common/individual innovation decomposition;
- learned confidence/dynamic-activation gate;
- fixed alpha sweeps as a main-model selection procedure;
- ODE-Jacobian covariance propagation;
- GASF-to-ODE FiLM gating;
- direct acceleration-output correction;
- ODE-derived process-noise/covariance claims;
- matched-oracle Pd/clutter tracker parameters in the main robustness table.

These may be mentioned only as rejected diagnostics or appendix ablations when
they help explain the final design. They must not be listed as simultaneous
novel components of P4.

## Manuscript terminology

| Code/artifact term | Manuscript term |
|---|---|
| `A_temporal`, P2 | Temporal prior / Temporal SGGT-V2 backbone |
| `CausalGroupInnovationCorrector` | causal group-innovation corrector (CGIC) |
| `raw` | target-own strict prior innovation |
| `peer_raw` | leave-one-out peer-consensus innovation |
| `association_detection_mass` | soft detection/association mass |
| `association_entropy` | normalized association entropy |
| `association_spread` | soft-candidate spatial spread |
| P4 | full association-aware CGIC model / proposed model |
| P3 | own-only counterfactual |
| P5 | no-association-context counterfactual |

## Result-writing restrictions at this freeze

1. State clearly that the current complete CGIC evidence uses one training
   seed, even though trajectory-paired bootstrap is available.
2. Do not convert trajectory bootstrap into a claim of between-training-seed
   robustness.
3. Do not claim improved NIS calibration: the current BP macro coverage is
   slightly lower for P4 than P2.
4. Report the BP result as a modest robustness improvement (P4 wins 8/9 cells),
   including the `Pd=0.75`, clutter-30 regression.
5. Treat the association-context contribution as secondary because P4 and P5
   are nearly tied on several clean/IID/OOD metrics.
6. Treat aligned peer innovation as the principal empirically supported
   incremental mechanism because P4 substantially outperforms P3.

