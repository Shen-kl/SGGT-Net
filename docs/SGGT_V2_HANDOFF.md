# SGGT-V2 handoff

## Dynamic-SEAN experiment (2026-07-22)

- Branch: `fix/sggt-v2-gpu-training`
- Seed: 123
- Dataset split: repository train/test split, unchanged
- Training: random 8-step rollout, 50% event windows, batch size 4
- Validation: fixed window, batch size 1
- Covariance transition: Neural ODE Jacobian
- Early stopping: patience 10, maximum 100 epochs

The revised SEAN retains supervised symmetric structure prediction but acts only
as a task-weight and activation controller over edges already present in the
encoder communication graph. It no longer injects a second graph aggregation
directly into the ODE base feature or process-noise path. Its confidence-gated
GASF/FiLM residual is initialized near zero, so the full model starts close to
the GNN-GRU baseline.

Sequential runner: `scripts/run_dynamic_sean_ablation.ps1`

Experiments:

1. `v2_dynamic_sean_full_seed123`
2. `v2_dynamic_sean_no_sean_seed123`

After both runs finish, evaluate each best checkpoint with
`evaluation/run_experiment.py` in fixed-window and full-rollout modes. Do not
draw paper conclusions until those results and additional seeds are available.

## AdamW experiment (2026-07-22)

The optimizer is now AdamW with `weight_decay=1e-4`. The default schedule uses
five epochs of linear warmup from `1e-4` to `1e-3`, followed by cosine annealing
to `1e-5`. Both the full and no-SEAN models must use this optimizer for a fair
comparison. Sequential runner: `scripts/run_dynamic_sean_adamw_ablation.ps1`.

AdamW seed-123 best epochs were 48 (full) and 18 (no SEAN). Full-rollout global
RMSE (full vs no SEAN) was: position prediction 33.017 vs 37.644, velocity
prediction 7.431 vs 10.604, position update 22.187 vs 23.448, and velocity
update 6.750 vs 9.892. Thus the fixed-window position-update reversal did not
persist over complete trajectories.

Paired fixed-window EKF diagnostics show that the full model has a smaller
position gain (0.000698 vs 0.000772), smaller position-velocity cross covariance
(0.000240 vs 0.000473), and smaller posterior covariance trace (0.000471 vs
0.001071). Its position-update-worsened rate is 17.92% vs 16.35%. NIS is
heavy-tailed and poorly calibrated in both models. This indicates covariance/
gain calibration, not RK4 integration, explains the small fixed-window position
update disadvantage. Diagnostic JSON files are under `evaluation/results/`.

## Remaining core ablations (started 2026-07-23)

Runner: `scripts/run_remaining_adamw_ablations.ps1`. It uses the same seed,
split, eight-step window policy, AdamW/warmup/cosine schedule, early stopping,
and ODE Jacobian as the completed full and no-SEAN runs. It sequentially trains:

1. `v2_dynamic_sean_adamw_no_gasf_seed123` (SEAN on, GASF off)
2. `v2_dynamic_sean_adamw_no_sean_no_gasf_seed123` (SEAN off, GASF off)

Results are pending. Select checkpoints by minimum logged validation score rather
than by the final periodic checkpoint, then run fixed and full unified evaluation.

Training completed on 2026-07-24. Fixed-window best results:

| Variant | Best epoch | Composite | Pos pred | Vel pred | Pos update | Vel update | Struct F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full | 48 | 75.883 | 37.175 | 7.697 | 24.095 | 6.915 | 0.942 |
| No SEAN | 18 | 85.012 | 39.606 | 11.176 | 23.658 | 10.573 | N/A |
| No GASF | 62 | 80.868 | 38.739 | 9.060 | 24.132 | 8.937 | 0.815 |
| No SEAN, no GASF | 44 | 79.328 | 38.078 | 8.820 | 23.875 | 8.555 | N/A |

Best new checkpoints are `2026_07_24_06_26_.pth` (no GASF) and
`2026_07_24_15_29_.pth` (neither module). Full-rollout evaluation was started
after training; update this section from the unified JSON outputs when complete.

## Parametric GASF revision (2026-07-24)

The learned per-frequency graph-polynomial mixture was replaced by a compact
`a,b,c` response with constrained `a<=0` and `b<=0`, guaranteeing a smooth
monotone low-pass response. One real gain multiplies each complex FFT
coefficient, preserving phase. Filter parameters are conditioned on the
low/high energy ratio and structure entropy. Historical residual input is now a
CV-prior measurement innovation proxy instead of measurement minus truth or
posterior state. GASF remains a confidence-gated FiLM latent modulation; direct
acceleration correction was deliberately not used because prior experiments
showed that branch collapsing to zero.

Smoke training and all seven tests passed. Formal runner:
`scripts/run_parametric_gasf_full.ps1`. Compare its best checkpoint against the
existing AdamW full and no-GASF checkpoints using the unified evaluator.

Formal training stopped at epoch 27; the best checkpoint is epoch 17
(`2026_07_25_02_50_.pth`). Its fixed-window composite score was 84.629
(position prediction 39.585, velocity prediction 11.374, position update
23.603, velocity update 10.067, structure F1 0.562). This is worse than both
the previous full model (75.883) and no-GASF model (80.868). Do not adopt this
combined revision. Full-rollout evaluation was started to confirm the result;
then isolate filter shape, innovation definition, and confidence fusion in
separate one-factor experiments.

### ABC-filter-only experiment

`scripts/run_abc_filter_only.ps1` restores the original measurement-minus-state
residual construction, original internal residual scaling, and original
multiplicative SEAN/GASF confidence gate. The sole experimental change relative
to the previous AdamW full model is the constrained phase-preserving `a,b,c`
low-pass filter. Tests and a CPU real-data smoke optimization passed. Formal
output name: `v2_abc_filter_only_seed123`.

Training completed at epoch 43 (early stopping); best epoch 33 checkpoint is
`2026_07_25_18_16_.pth`. Fixed-window results: composite 75.505, position
prediction 36.813, velocity prediction 7.757, position update 24.080, velocity
update 6.855, structure F1 0.982. This slightly improves the previous full
model composite (75.883) and clearly beats no GASF (80.868). The failed combined
revision therefore cannot be attributed to the `a,b,c` filter alone. Full
rollout evaluation is in progress.

Full rollout completed. ABC-only vs previous full vs no GASF:

| Variant | Pos pred | Vel pred | Pos update | Vel update | NLL |
|---|---:|---:|---:|---:|---:|
| ABC only | 33.119 | 7.786 | 22.138 | 6.771 | -15.297 |
| Previous full | 33.017 | 7.431 | 22.187 | 6.750 | -15.397 |
| No GASF | 35.553 | 8.670 | 22.808 | 8.030 | -15.119 |

ABC-only clearly retains a useful GASF contribution relative to no GASF, but
does not globally outperform the previous learned filter. At structure changes
it improves position prediction/update slightly, but global velocity metrics
are worse. Its structure F1 is 0.987, while specificity/balanced accuracy/MCC
drop to 0.504/0.749/0.610 versus 0.820/0.900/0.702 for the previous full model,
indicating an overly positive structure predictor. Do not replace the previous
filter based on seed 123 alone.

Mechanism diagnostics now report ABC response, FiLM gate, and auxiliary
velocity-direction metrics. A one-batch smoke diagnostic found Nyquist gain
0.855, mean FiLM gate 0.854, and auxiliary velocity cosine 0.047, suggesting a
weak low-pass response and near-always-on modulation; confirm on the full fixed
validation set. `scripts/run_filter_multiseed.ps1` runs ABC, learned-filter, and
no-GASF controls for seeds 456 and 789 using the same protocol.

On 2026-07-26, ABC seed 456 completed: best epoch 55, composite 73.588
(position prediction 36.383, velocity prediction 6.883, position update 24.068,
velocity update 6.254). Learned seed 456 was externally interrupted during
epoch 60 without traceback/NaN/OOM; its current best is epoch 55, composite
74.601. Resume support was added and `scripts/resume_filter_multiseed.ps1`
continues it from epoch 56 before running no-GASF seed 456 and all seed-789
variants.

Full fixed-set mechanism diagnostics for ABC seed 123 found mean Nyquist gain
0.884, mean spectral gain 0.954, mean FiLM gate 0.834, and auxiliary velocity
cosine -0.199. Thus the trained filter is weakly low-pass, modulation is nearly
always strongly active, and the auxiliary head does not align with the true
velocity increment. These facts limit the current mechanistic claim even though
trajectory RMSE improves.

## Counterfactual on-demand GASF training (2026-07-26)

The GASF correction and activation policy are now identifiable components.
`gasf_film` produces the bounded FiLM correction, while a separate 65-parameter
`gasf_gate` predicts when to activate it. Counterfactual training evaluates the
same decoded window with gate overrides 0 and 1. Stage `expert` freezes the
backbone/gate and trains the forced-on expert; stage `gate` freezes everything
except `gasf_gate` and uses the relative forced-off versus forced-on error as
the soft activation target. Validation counterfactual metrics use `real_mask`;
an initial unmasked diagnostic run is invalid and must not be reported.

Seven unit tests and a real-data GPU forward/backward smoke test pass. The smoke
run used about 4.8 GB process RAM and did not reproduce prior host-memory
pressure.

Stage-1 diagnostic protocol: seed 123, initialized from the no-GASF checkpoint,
16 training trajectories, 8 fixed validation trajectories, four-step windows,
AdamW with two-epoch warmup and cosine decay. The best expert checkpoint is
epoch 8:
`checkpoint/counterfactual_expert_stage1_masked_seed123/GNN_groupTargetTracking/2026_07_26_22_53_.pth`.
Forced-off validation MSE is 4.8820e-4 and forced-on expert MSE is 4.6802e-4
(4.13% relative improvement), while the expert wins on 46.4% of valid nodes.

Stage-2 trained only the 65 gate parameters for 10 epochs from that expert.
Best diagnostic checkpoint:
`checkpoint/counterfactual_gate_stage2_seed123/GNN_groupTargetTracking/2026_07_26_22_58_.pth`
(epoch 8). The deployed four-RMSE composite changed from about 72.337 early in
gate training to 72.319. This is a small positive diagnostic, not yet a formal
result. Before joint fine-tuning or paper claims, repeat expert/gate training on
a larger fixed subset and log gate calibration/activation by maneuver regime.

The first enlarged 64/24-trajectory run used an eight-step rollout but was
stopped by user decision before completion; it is not a valid experiment.
Starting 2026-07-26, four-step rollout is the default for subsequent training.
Eight-step rollout is reserved for later long-horizon fine-tuning only if the
four-step model does not reach adequate fixed-window/full-trajectory accuracy.
The replacement enlarged expert run writes to
`checkpoint/counterfactual_expert_stage1_large_r4_seed123`.

That replacement run was also stopped before completion when the rollout
execution policy changed. The trainer now caches the differentiable encoder
sequence, mixture coefficients, and decoder hidden state across the four
sliding prediction/update steps. Therefore each four-step rollout performs one
historical Encoder pass followed by four stateful Decoder passes; the four
losses are accumulated and backpropagated once. Measurements, residuals,
future graph edges, ODE propagation, and EKF updates still advance at every
step. A two-trajectory real-data GPU smoke run completed forward/backward and
validation without graph-reuse errors; seven unit tests pass. The valid enlarged
replacement output is
`checkpoint/counterfactual_expert_encoder_once_r4_large_seed123`.

The enlarged encoder-once expert run completed with early stopping at epoch 18.
Its best checkpoint is epoch 12:
`checkpoint/counterfactual_expert_encoder_once_r4_large_seed123/GNN_groupTargetTracking/2026_07_26_23_57_.pth`.
On the fixed 24-trajectory validation subset, forced-off MSE is 6.4090e-4 and
forced-on expert MSE is 6.0754e-4 (5.21% relative improvement); the expert wins
on 63.8% of valid nodes. The enlarged 65-parameter gate stage was then started
with the same 64/24 split and four-step encoder-once rollout, writing to
`checkpoint/counterfactual_gate_encoder_once_r4_large_seed123`.

The enlarged gate stage completed all 15 epochs. Its best checkpoint by the
four-RMSE validation composite is epoch 14:
`checkpoint/counterfactual_gate_encoder_once_r4_large_seed123/GNN_groupTargetTracking/2026_07_27_01_02_.pth`.
On the unified 24-trajectory fixed evaluation, it improves the expert
checkpoint's deployed composite from 104.168 to 103.954 (0.205%). Position
prediction, velocity prediction, position update, and velocity update RMSE
change from 50.302/11.389/29.293/13.183 to
50.218/11.312/29.283/13.142; NLL changes from -8.060 to -8.092. However mean
effective FiLM gate only increases from 0.0075 to 0.0590, despite a 63.8%
counterfactual expert win rate, and validation gate BCE remains 1.642. The
current multiplication of the learned policy by `GASF_confidence` therefore
appears to cap/suppress activation and makes the counterfactual target
unreachable. Do not start joint fine-tuning before separating or recalibrating
the confidence factor and adding gate-target calibration metrics.

The gate-target reachability fix was then tested: BCE supervises the raw policy
probability, while raw policy, GASF confidence, and their effective product are
reported separately. The 15-epoch run completed at
`checkpoint/counterfactual_policy_gate_encoder_once_r4_large_seed123/GNN_groupTargetTracking/2026_07_27_01_30_.pth`.
Validation BCE fell from 1.197 to 0.702, but the unified fixed composite is
104.059, between the untrained expert checkpoint (104.168) and the previous
confidence-coupled gate (103.954). Raw policy mean is 0.473, confidence mean is
0.0632, and effective gate mean is only 0.0299. Thus target reachability was
fixed, but the policy learned an almost constant ambiguous probability rather
than a discriminative on-demand decision; confidence remains the dominant
amplitude limiter. Do not use this checkpoint for joint fine-tuning. Next
diagnostics should measure policy-target AUC/calibration and add causal
state/innovation features (residual energy, NIS or base/expert disagreement)
to the gate rather than relying only on the GASF latent.

## Direct GASF/ODE fusion (2026-07-27)

By user decision, the new primary path removes both learned activation policy
and confidence scaling. The GASF spectral filter is the only residual-selection
mechanism. ODE base features and GASF latent features are concatenated, passed
through a lightweight projection, and added residually to the ODE hidden
feature. Legacy gated FiLM remains available only as an ablation via
`--gasf_fusion_mode film`; the new default is `concat`.

Seven unit tests and a two-trajectory GPU smoke run pass, with finite gradients
and stable covariance. An enlarged 64/24, four-step, encoder-once direct-fusion
expert run was started from the prior best trained GASF expert checkpoint and
writes to `checkpoint/gasf_concat_encoder_once_r4_large_seed123`. At epoch 0,
forced-off/forced-on validation MSE is 6.36e-4/6.30e-4 and the direct-fusion
expert wins on 57.2% of valid nodes.

The enlarged direct-fusion run completed with early stopping at epoch 17
(21m53s); best checkpoint is epoch 11:
`checkpoint/gasf_concat_encoder_once_r4_large_seed123/GNN_groupTargetTracking/2026_07_27_09_38_.pth`.
Forced-off/forced-on validation MSE is 6.1986e-4/6.0501e-4 (2.40% mean
improvement), although node-wise win rate is 49.3%, indicating fewer
higher-magnitude wins. Unified fixed evaluation gives
50.534/10.204/29.988/12.165 for position prediction, velocity prediction,
position update, and velocity update (composite 102.891). This beats the prior
best gated FiLM composite 103.954 by 1.02%, driven by velocity improvements,
but position prediction/update and NLL (-7.889 versus -8.092) are worse.
Therefore direct fusion is promising and matches the desired simpler mechanism,
but it still over-corrects velocity in a way that degrades position/calibration.
Do not claim an across-the-board improvement until full rollout and a
position/velocity-balanced fusion-loss check are completed.

Full-trajectory rollout was completed on the same first 24 test trajectories
using one Encoder pass per trajectory and recurrent Decoder updates (91 rollout
steps). Direct concat gives 32.788/8.463/20.642/6.503 and NLL -13.996; the
legacy gated FiLM control gives 30.183/7.655/19.387/5.988 and NLL -14.859.
Thus concat is worse on every full-rollout metric despite its better four-step
composite. Direct-fusion counterfactual correction averages 1.799 position
units and 1.442 velocity units per step, with an abnormal step-47 spike to
6.611/4.629; errors rise substantially after roughly step 31. The current EWTA
loss sums equally weighted normalized coordinates. Because S_POS=1500 and
S_VEL=300, near the Huber quadratic region a unit of physical velocity error is
weighted about 25 times as strongly as a unit of physical position error. This
matches the observed short-window velocity gain/position loss and is not an
appropriate basis for joint fine-tuning. Do not jointly fine-tune yet. First
bound the direct concat residual (without adding a gate) and test an explicit
position/velocity-balanced loss in a controlled fusion-only experiment.

## Current-protocol full-data three-way ablation (2026-07-27)

All three variants were retrained with all 400 training trajectories and all
100 validation trajectories, seed 123, batch size 4, one Encoder plus four
stateful Decoder steps, AdamW/warmup/cosine, and ODE Jacobian propagation.
Direct concat is used whenever GASF is enabled. Best fixed validation results:

| Variant | Epoch | Pos pred | Vel pred | Pos update | Vel update | Composite | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SEAN + GASF | 0 | 59.463 | 11.974 | 35.573 | 12.293 | 119.303 | 0.932 |
| no SEAN + GASF | 6 | 60.833 | 11.143 | 36.871 | 11.970 | 120.817 | N/A |
| no SEAN, no GASF | 3 | 60.096 | 11.894 | 36.221 | 11.728 | 119.940 | N/A |

SEAN+GASF has the best composite, 1.25% better than no-SEAN+GASF and
0.53% better than neither module. GASF without SEAN is 0.73% worse than neither,
again indicating that GASF needs the inferred graph. However the full model
selected epoch 0 and never improved afterward, and the variants inherit
different historical checkpoints. Treat this as transfer/fine-tuning evidence,
not a final from-scratch paper ablation. Identical full-trajectory evaluation
is still required.

## Epoch-0-best diagnosis (2026-07-27)

The full-model run's epoch-0 selection is a training-configuration failure,
not evidence that the transferred model is already optimal. During the
original run the fixed composite worsened from 119.303 at epoch 0 to 122.697,
125.002, 128.236 and 130.465 while training loss became more negative. The
warmup simultaneously raised the effective learning rate from 1e-4 toward
1e-3, and the pre-clip gradient norm stayed around 15--19.

Loss-component instrumentation showed the cause of the contradictory loss:
on a controlled validation batch, prediction/update losses were
0.000722/0.000370, while Gaussian NLL was -13.543. With
`lambda_nll=0.05`, NLL contributed about -0.677, hundreds of times more than
either trajectory term. The covariance log determinant can therefore improve
the scalar loss while physical RMSE worsens.

Two 64/24-trajectory, four-epoch controls used a fixed 1e-4 learning rate and
`lambda_nll=0`. With all parameters trainable, the validation composite was
110.928, 106.315, 102.656 and 103.072 (best epoch 2), with gradient norms only
0.142--0.210. With only GASF, concat fusion and the auxiliary head trainable,
the corresponding RMSE sums were 112.368, 107.860, 105.142 and 103.754. Thus
both training modes learn beyond epoch 0; adapting the newly initialized
concat layer matters, while cautious whole-network fine-tuning is slightly
better.

There is also a checkpoint-selection bug in the diagnostic `expert` stage:
`_validation_score` always selects `counterfactual_expert_mse`, which is zero
when counterfactual training is disabled. Its logged score is therefore zero
even though the physical RMSE changes. Fix this before reusing the staged
trainer. Before further full-data runs, evaluate and save the untrained
initialization as epoch -1, rebalance/normalize NLL, use transfer-specific
learning rates, and make the new concat residual identity-safe at
initialization.

## Staged fine-tuning fix (2026-07-27)

The five requested training changes are implemented. Gaussian NLL is now the
complete NLL including the Gaussian constant, averaged per state dimension and
clamped to [-5, 25]; its default weight is reduced from 0.05 to 1e-4. The
maximum learning rate defaults to 6e-4. For joint concat-GASF fine-tuning, only
GASF, concat fusion and the auxiliary velocity head train for two epochs at
6e-4; the whole model then unfreezes at 1e-4 and follows cosine decay toward
1e-5.

Validation now runs before any optimizer update and is recorded as epoch -1,
including a separately named checkpoint. Checkpoint filenames include the
epoch, periodic saves no longer replace the best-checkpoint pointer, and logs
report the learning rate actually used by each parameter group. Expert-stage
selection uses counterfactual expert MSE only when counterfactual training is
enabled; otherwise it falls back to the four-term physical RMSE composite.

Eight unit tests pass. A GPU smoke run confirmed epoch -1 recording, fusion-only
epoch 0 at 6e-4, full-model unfreezing at epoch 1 with 1e-4, finite gradients,
and symmetric PSD covariance. In that deliberately tiny 4/4-sample run,
epoch -1 remained best and the reported best path correctly stayed at
`checkpoint/staged_nll_epochminus1_smoke_v2/GNN_groupTargetTracking/2026_07_27_14_52_epoch_-1.pth`.
This tiny run validates control flow rather than accuracy. The full-data
ablation launcher now records the new settings explicitly.

## Full-data staged-v2 run (started 2026-07-27 16:01 CST)

The three experiments are running sequentially from
`scripts/run_current_protocol_full400_staged_v2.ps1`: SEAN+GASF,
no-SEAN+GASF, then no-SEAN/no-GASF. They use all 400/100 trajectories, seed
123, one Encoder plus four Decoder steps, batch size 4, normalized NLL weight
1e-4, two fusion-adaptation epochs at 6e-4 where applicable, then full-model
fine-tuning from 1e-4 toward 1e-5. Maximum length is 40 epochs with validation
early-stopping patience 8. Outputs use the
`checkpoint/current_full400_staged_v2_*_seed123` and
`log/current_full400_staged_v2_*_seed123.log` paths. The full model entered
the epoch -1 initialization evaluation at 16:01; Python PID was 28500 and GPU
memory usage was about 3.0 GB at the first status check.

## Evaluation-protocol correction and paper-v3 reset (2026-07-28)

The staged-v2 ablation was stopped because its variants inherited different
historical checkpoints and the old FiLM checkpoints lacked all eight concat
fusion parameters. The apparent 60/40 position error was traced to evaluation
rather than a sudden model collapse. A midpoint four-step window cold-started
the decoder and covariance every time: the staged full model's horizon
position prediction/update errors were 95.34/66.56 at step 1 but 36.58/22.05
at step 4. With four unscored burn-in steps, the same 100-trajectory
evaluation is 32.31/21.46. Full 100-trajectory rollout gives 32.97/22.60,
confirming that burn-in-4 is a useful low-cost proxy for steady tracking.

The unified evaluator now exposes `--encoder_mode`,
`--fixed_starts`, and `--burn_in_steps`. Encoder-once and
encoder-every-step were indistinguishable over burn-in-4 plus four scored
steps (composites 67.5795 and 67.5792), so encoder-once is retained based on
evidence. On full 100-trajectory rollout, the prior concat checkpoint scores
34.185/8.111/22.918/6.240 (composite 71.455), while staged epoch 0 scores
32.966/7.470/22.600/6.004 (composite 69.039).

A measurement covariance alignment bug was fixed. R had been computed from
`start_bias` although the first target is at `history + start_bias`, then held
constant for an entire rollout. Training, validation and unified evaluation
now use the target-time R and recompute it each step. The numerical effect on
the four-step checkpoint test was small, but the correction is required
because R depends on range and azimuth.

Training validation now performs four unscored recurrent burn-in steps and
scores the following four steps. Cold-start metrics remain a separate
diagnostic. A velocity-coordinate weight was added, but the proposed physical
weight 0.04 was rejected by a controlled 64/24 test: its best trained
composite was 74.787, versus 71.669 for weight 1.0. The formal default
therefore remains 1.0. SEAN is now included in the module-adaptation phase
when enabled. Best checkpoints are also copied to a stable `best.pth` alias.

The final paper protocol will no longer use three unrelated transfer
checkpoints. Phase 1 trains a no-SEAN/no-GASF shared base from scratch on all
400 trajectories. All three variants will then start from the exact same
`best.pth`; unused SEAN/GASF parameters in that checkpoint remain at their
common seeded initialization. The shared-base launcher is
`scripts/run_paper_v3_shared_base.ps1` (seed 123, rollout 4, validation
burn-in 4, normalized NLL weight 1e-4, maximum LR 6e-4, up to 60 epochs).
Ten unit tests and a from-scratch GPU smoke test pass.

The shared no-SEAN/no-GASF base completed in 19,018 seconds (about 5 h 17 m).
It reached its best burn-in-4/four-step validation composite at epoch 49 and
stopped at epoch 59 after ten epochs without improvement. Best metrics are
37.409 position prediction, 11.516 velocity prediction, 22.484 position
update and 8.737 velocity update (composite 80.146). The final epoch composite
is 80.521, only 0.47% above the best, supporting genuine convergence rather
than an unstable isolated minimum. The normalized NLL at the best point is
-2.621 and the training gradient norm is 0.181; covariance remained finite
and symmetric during training. F1 is intentionally zero because this shared
base disables SEAN. The stable shared initialization is:
`checkpoint/paper_v3_shared_base_seed123/GNN_groupTargetTracking/best.pth`.
## 2026-07-31: temporal-only GASF trained from scratch

- Control: SEAN off, GASF on, `gasf_filter_mode=abc`,
  `gasf_graph_mode=temporal_only`, concat fusion.  No graph propagation occurs
  inside GASF and no initialization checkpoint is loaded.
- Protocol: seed 123, full 400/100 split, encoder once + decoder rollout 4,
  validation burn-in 4, AdamW, peak LR `6e-4`, warmup 5, 60 epochs.
- Run time: 23,922.8 s (about 6 h 38 min); best checkpoint was epoch 54.
- Best fixed-window validation: position prediction 37.2609, velocity prediction
  11.0695, position update 22.8365, velocity update 8.0471; four-RMSE
  composite 79.2139.
- Same-seed scratch no-GASF base: 37.4094 / 11.5158 / 22.4840 / 8.7369;
  composite 80.1462. Temporal-only GASF improves the composite by 0.9323,
  mainly through velocity, while position update worsens by 0.3524.
- Checkpoint:
  `checkpoint/paper_v3_temporal_only_gasf_scratch_seed123/GNN_groupTargetTracking/best.pth`.
- Full-trajectory rollout remains to be run before using these numbers in the
  paper.

## 2026-08-09: label-free spectral-SEAN refactor status

Work continues on branch `feature/sggt-v2-spectral-sean` from commit
`e192aa6`. SEAN is embedded in GASF as an interaction selector after temporal
ABC filtering. It predicts a soft graph over the encoder candidate graph and
scales only the graph-interaction increment. Group-structure labels and their
losses are disabled. The ODE receives direct additive/concat features; no
separate confidence gate is used.

The graph expert is explicitly trained against a temporal-only counterfactual,
with auxiliary velocity and rank terms. The selector observes four strictly
causal residual statistics in addition to temporal/structural latent features.
Counterfactual utility labels were almost symmetric and unpredictable, so a
physical maneuver-demand target was added as an optional selector supervisor.

Two correctness bugs were fixed before interpreting that experiment:

1. `interaction_gate_lr_multiplier` was not connected to the optimizer. Main
   now creates and asserts named backbone/selector AdamW groups.
2. The supposed maneuver label used the learned previous posterior velocity,
   causing its class ratio to change across epochs. It now uses consecutive
   ground-truth velocities and is immutable. Class balancing uses a fixed
   trajectory-batch prior and pooled TP/TN statistics report balanced accuracy.

Validation status: 25 unit tests and a real V9 forward/backward smoke pass. A
full-time 8/8 pilot (three epochs, seed 123, additive fusion, backbone LR
1e-4, selector LR 1e-3, demand weight 0.1) improved position prediction/update
from 28.1209/18.9154 m to 27.4118/18.4214 m without NaNs. However the final
selector activation was 0.49114 with std 0.000097; positive/negative activation
means differed by only 0.000238 and balanced accuracy was 0.5. Thus training is
numerically valid, but on-demand activation is not yet empirically valid. Do
not launch formal 200/50 or full-data ablations from this checkpoint, and do
not claim dynamic interaction in the paper yet.

Relevant artifacts:

- checkpoint: `checkpoint/dev8_spectral_sean_fixed_demand_seed123/`
- training log: `log/dev8_spectral_sean_fixed_demand_seed123.log`
- audit evaluation: `log/eval_dev8_fixed_demand_best.log`

### Paired graph-expert validation (2026-08-09)

An evaluation-only intervention was added to compare three closed-loop paths
from the exact same epoch-2 checkpoint: Temporal-only (alpha=0), learned SEAN,
and full graph expert (alpha=1). The protocol used the first eight fixed V9
validation trajectories, streaming encoder, complete online prediction/update,
and 6,148 valid lifecycle-masked points.

| path | position prediction | velocity prediction | position update | velocity update |
|---|---:|---:|---:|---:|
| Temporal-only | 27.3059 | 6.5670 | 18.3410 | 4.8592 |
| Learned SEAN | 27.4118 | 6.5763 | 18.4214 | 4.8693 |
| Full expert | 27.5352 | 6.5884 | 18.5142 | 4.8816 |

Strong-window position prediction/update was 42.8151/26.7333 for Temporal-only
and 43.0276/26.9106 for learned SEAN. The paired full expert won 43.77% of all
valid nodes and 40.04% of strong-window nodes. Learned effective interaction
had norm 11.51% of the temporal feature, so this is not a numerical zero-path;
increasing it to the full 23.41% expert made every global metric worse.

This result does **not** validate SEAN superiority. It localizes the current
failure to the graph expert increment: performance degrades monotonically as
its strength increases. Do not scale or report this SEAN design. The next gate
for development is explicit: full-expert alpha=1 must beat temporal alpha=0 on
paired strong-window prediction before selector training resumes.

Evaluator: `evaluation/compare_sean_temporal_v3.py`  
Result: `evaluation_results/sean_vs_temporal_dev8_epoch2.json`

## 2026-08-09: graph-expert repair status

The graph expert was retrained under a fixed temporal-only online history with
task-scale physical counterfactual losses and an `expert_strong` checkpoint
criterion.  SEAN edge weights now preserve the causal candidate graph and
learn only a bounded residual reweighting.  Offline V9 analysis (32 train
scenes, 38,550 strong candidate pairs) found median neighbour maneuver cosine
0.9996, confirming that the dataset contains a strong graph-consensus signal.

The best current checkpoint is epoch 8:
`checkpoint/graph_expert_prior_strong_refine2_train32_val8_seed123/GNN_groupTargetTracking/best_expert_strong.pth`.
It was trained on 32 scenes and selected on the first 8 fixed validation
scenes.  Same-state validation strong advantage changed from -0.000821 to
+0.000013 and direction loss fell to 0.2815 without moving the temporal
baseline.

Strict alpha=1 closed-loop rollout improves global position prediction/update
from 27.3059/18.3410 to 27.2562/18.3098 m and improves steady and switch
windows.  Strong position prediction/update are 42.8425/26.7596 m versus the
temporal 42.8151/26.7333 m, still worse by 0.0273/0.0263 m; strong velocity
prediction improves marginally.  The graph expert is substantially repaired
but has not passed the final always-on strong-position gate.  Do not start a
formal large ablation or claim SEAN superiority from this checkpoint yet.

Diagnostics/results:
`evaluation_results/graph_prior_stats_train32.json` and
`evaluation_results/graph_expert_strong_refine2_epoch8_rollout.json`.

### On-demand activation run (started 2026-08-09)

The selector stage now freezes every component except the 4,353-parameter SEAN
activation head and learns from same-state alpha=0/alpha=1 radial-position
counterfactual labels.  It uses a 0.0001 m dead zone, class-balanced BCE,
steady/switch/strong weights of 1.0/1.5/2.0, and mixed-path regret protection.
Validation explicitly retains the paired branches, and frozen dropout remains
deterministic.  A real V9 gradient smoke verified that only the four activation
head tensors changed and all values remained finite; 30 unit tests pass.

A 32-train/8-val full-trajectory run is active at
`checkpoint/activation_gate_position_train32_val8_seed123`, with log
`log/activation_gate_position_train32_val8_seed123.log`.  Epoch -1 is
27.1788/18.2691 m position prediction/update, 0.500 gate balanced accuracy and
0.4911 mean activation.  After convergence, compare both `best_gate.pth` and
`best_position.pth` against strict alpha=0 and alpha=1 rollouts before making
any SEAN claim.

The run finished at epoch 8 by six-epoch early stopping.  It was numerically
stable, but on-demand activation did not emerge: epoch-8 validation balanced
accuracy is 0.5171 and activation mean/std are 0.4997/0.00013.  Complete-rollout
activation means are 0.49978/0.49970/0.49943 for steady/switch/strong.  Under
the correct `concat` protocol, temporal/learned/expert position prediction are
27.180320/27.178817/27.177257 m, and strong position prediction is
43.228883/43.225723/43.221289 m.  Learned alpha is only a near-constant
half-strength interpolation and does not beat alpha=1.

Important protocol correction: the older graph-expert JSON used evaluator
default `add`, although formal training uses `concat`.  Its 27.3059 m temporal
number is obsolete.  The old checkpoint re-evaluated with `concat` reproduces
27.180320 m exactly.  Current result files are
`evaluation_results/activation_gate_train32_best_gate_rollout.json` and
`evaluation_results/activation_gate_train32_best_position_rollout.json`.

The subsequent fixed-alpha/oracle feasibility test confirms that dynamic
selection is not currently worth further training. Under `concat`, position
prediction for alpha 0/0.25/0.5/0.75/1 is
27.180320/27.179720/27.178809/27.177887/27.177255 m. Thus alpha=1 is the best
fixed choice, but gains only 0.00307 m globally and 0.00760 m on strong windows;
switch prediction worsens by 0.00323 m. A non-causal closed-loop greedy oracle
reaches 27.171617 m, still only 0.00870 m ahead of temporal. The selector is
not hiding a meaningful accuracy opportunity. Diagnose graph-increment
strength and concat/ODE sensitivity next, including evaluation-only gains above
one, before redesigning or retraining activation. Full result:
`evaluation_results/activation_alpha_sweep_oracle_val8.json`.
## 2026-07-31: actual-fusion-residual auxiliary supervision pilot

- Added `--gasf_aux_source {gasf,fusion}`.  `fusion` applies the existing
  auxiliary delta-v head to the exact concat residual added to the ODE
  conditioning; inference fusion itself is unchanged.  Eight SGGT-V2 unit
  tests pass.
- Controlled pilot: temporal-only abc GASF, no SEAN, scratch seed 123, 64/24
  scenarios, 20 epochs maximum. Both variants selected epoch 12 and stopped at
  epoch 18.
- Full rollout on the same 24 validation trajectories:
  - raw GASF auxiliary source: 92.7596 / 37.4178 / 41.7476 / 19.6730;
    aux cosine 0.2776 (maneuver 0.2425).
  - actual fusion residual source: 92.2898 / 36.4251 / 42.7575 / 19.1176;
    aux cosine 0.3237 (maneuver 0.2765).
- Fusion-residual supervision improves prediction RMSE, velocity-update RMSE,
  and auxiliary direction agreement, but worsens position-update RMSE and NLL.
  The pilot is a weak positive signal, not yet sufficient evidence of a robust
  overall improvement; do not claim it in the paper or launch a large matrix
  without confirming on the full split.

## 2026-08-10: physical SEAN interaction path

The latest diagnosis overturns the earlier conclusion that SEAN has only a
millimetric opportunity.  That conclusion was specific to the latent graph
expert.  V9 does contain a useful causal graph signal: SEAN-weighted neighbour
latest innovation has validation cosine 0.31--0.34 with the ideal correction,
and reaches 0.475 on strong nodes in one independent block.  The latent
GASF/ODE path reduced this to roughly 0.017 cosine and 0.024 m correction.

The current formal development design therefore separates the two roles:

1. Temporal GASF uses the fixed `(a,b,c)` spectral filter to extract temporal
   residual features.
2. SEAN constructs a soft causal interaction graph from state/relative-motion
   evidence.
3. The graph-weighted latest innovation enters the ODE acceleration derivative
   through a bounded physical gain, rather than an unconstrained latent graph
   increment.  Maximum gain is 0.075 and initialization is 0.05.
4. An eight-feature post-graph selector provides dynamic activation.  A formal
   floor of 0.75 prevents low-confidence false negatives from removing a
   broadly useful correction.  Exact alpha=0 and alpha=1 evaluation overrides
   bypass the floor and remain valid ablations.

Offline selector fitting uses cached causal features from 32 train trajectories
and val48--63.  It obtains validation balanced accuracy 0.6228 and positive
minus negative score separation 0.1227.  The selected checkpoint is:

`checkpoint/physical_context_gate_floor075_offline_train32_val48_63_seed123/best_model.pth`

Independent complete online rollout on val80--95 with streaming encoder,
batch 1, lifecycle-correct masks and `concat` fusion gives:

| path | pos pred | pos update | vel pred | vel update |
|---|---:|---:|---:|---:|
| temporal alpha=0 | 29.370698 | 20.682845 | 6.080806 | 4.665422 |
| fixed expert alpha=1 | 26.500070 | 18.984977 | 5.502640 | 4.212708 |
| dynamic floor=0.75 | **26.469845** | **18.952816** | **5.493294** | **4.197807** |

Thus the physical interaction path improves position prediction by 9.8767%
and position update by 8.3646% versus the temporal path.  Dynamic activation
is 0.0302 m better globally than fixed expert, but strong-event prediction is
0.246 m worse than expert; report it as a calibrated compromise, not an oracle.

Important protocol repair: checkpoint payloads now persist
`interaction_activation_floor`, `gasf_fusion_mode`, `gasf_graph_mode`, and
SEAN/GASF enable flags.  V3 evaluators read the fusion mode from the checkpoint
and write it into result protocol metadata.  This prevents the previous silent
`add` versus `concat` mismatch.  Gate-loss reporting also excludes sentinel
zeros when a time step lacks both target classes.

Key files/results:

- implementation: `model/sggt_v2_modules.py`,
  `model/motion_models_base.py`, `model/gru_gnn_residual.py`
- selector fitting: `evaluation/pretrain_context_activation_v3.py`
- formal independent result:
  `evaluation_results/physical_context_gate_floor075_formal_concat_val80_95.json`
- baseline/expert result:
  `evaluation_results/physical_context_gate_baselines_val80_95.json`
- detailed reasoning and rejected paths: `debug_record.md`

Status: mechanism validation is positive, but the source backbone checkpoint
was only a physical-gain smoke run.  It is not paper-ready evidence.  Next run
from scratch under one fixed full-data split and seed: (A) temporal GASF,
(B) SEAN + fixed physical expert, and (C) SEAN + calibrated dynamic physical
interaction.  Use complete online rollout for final evaluation and retain
steady/switch/strong metrics.  Do not modify paper conclusions until those
runs converge and are repeated across seeds.

## 2026-08-10: formal CV physical-model run

The formal run now reuses the converged full-V9 Temporal backbone instead of
retraining it.  The auditable shared checkpoint is
`checkpoint/formal_cv_physical_shared_init_seed123/shared_init.pth`: 151
compatible backbone/Temporal tensors come from the epoch-44 A1 model, while 31
new SEAN/physical tensors come from the validated physical mechanism.  It
loads with no missing or unexpected keys and uses CV covariance propagation.

Do not reuse the former gain 0.05 with this backbone.  It worsens the first
eight validation trajectories.  Two disjoint gain scans established a robust
0.015--0.02 interval; gain 0.02 improves position prediction/update by
0.196/0.108 m on val0--7 and by 0.354/0.153 m on val8--15 versus gain 0.
The formal bounded gain initializes at 0.02 and retains maximum 0.075.

The sequential formal job began at 10:39 CST using
`scripts/run_formal_cv_physical_models.ps1`.  Fixed Expert training uses all
1000 train and 200 validation trajectories, batch 4, all legal causal steps,
streaming encoder, alpha=1, CV, seed 123 and position-first early stopping
(minimum improvement 0.05 m, patience 4, maximum 12 epochs).  Exactly one
physical-gain parameter trains.  On completion, the job automatically collects
full causal selector features once, fits the dynamic activation head offline,
sets floor 0.75 and runs the full 200-trajectory temporal/dynamic/expert
rollout.  Current log:
`log/formal_cv_fixed_expert_full_seed123.log`.

## 2026-08-10: formal CV physical-model run completed

The sequential job completed successfully at 15:00 CST; no training Python
process remains. Fixed-Expert optimization ran epoch -1 plus epochs 0--3 and
stopped under the predeclared 0.05 m / patience-4 rule. Validation position
prediction decreased monotonically from 21.280397 m at epoch -1 to
21.268532 m at epoch 3; position update decreased from 15.995119 m to
15.986453 m. The selected fixed-Expert checkpoint is epoch 3.

The offline selector used 606389 informative training points and 120946
validation points. Its validation balanced accuracy is 0.585102 and positive
minus negative score separation is 0.052148. Full causal online rollout on
all 200 validation trajectories gives:

| path | position prediction | position update | velocity prediction | velocity update |
|---|---:|---:|---:|---:|
| Temporal, alpha=0 | 21.621178 | 16.189333 | 4.273592 | 3.644459 |
| Dynamic, learned alpha | **21.261715** | 15.989510 | **4.239050** | **3.623252** |
| Fixed Expert, alpha=1 | 21.268532 | **15.986454** | 4.262993 | 3.638198 |

Dynamic improves position prediction/update over Temporal by 0.359463 m
(1.6625%) and 0.199823 m (1.2343%). It is only 0.006817 m better than fixed
Expert in prediction and 0.003057 m worse in update. Mean activation is
0.872900 with standard deviation 0.023338; steady/switch/strong means are
0.869995/0.870880/0.876730. Thus the graph interaction is useful, but the
learned selector remains close to always-on and has not established a strong
event-selective advantage. On strong events, fixed Expert is better than
Dynamic by 0.029350/0.022978 m in position prediction/update.

Artifacts:

- fixed Expert: `checkpoint/formal_cv_fixed_expert_full_seed123/GNN_groupTargetTracking/best_position.pth`
- dynamic model: `checkpoint/formal_cv_dynamic_full_seed123/best_model.pth`
- selector report: `evaluation_results/formal_cv_dynamic_pretrain_train1000_val200_seed123.json`
- three-way rollout: `evaluation_results/formal_cv_physical_models_full_val200_seed123.json`

Important evidence boundary: the 200-trajectory validation set participated
in selector fitting and model selection. These results validate the training
mechanism but are not unbiased paper test results. Evaluate the frozen models
once on untouched IID-test 300 and OOD-test 400 splits before making paper
claims.

## 2026-08-10: whitened-innovation SEAN development branch

Branch: `feature/sggt-v2-whitened-innovation`.

The optional `--innovation_decomposition` path uses the EKF innovation
covariance to whiten residuals and compute NIS, then decomposes physical
innovation into information-weighted SEAN group-common and individual terms.
The two standardized histories and uncertainty statistics enter GASF through
a zero-initialized feature residual. There is no confidence gate and no direct
acceleration correction. The legacy Temporal path remains the exact alpha=0
comparator, and lifecycle validity is `current_real & previous_real`.

Tests: 30 CPU unit tests and real V9 CPU/GPU two-step causal smoke pass. A
32-train/16-validation full-time development run completed 10 expert epochs
without NaN. Forced-alpha closed-loop validation at epoch 9 improves position
prediction/update from 31.403008/19.986940 m to 31.297400/19.960078 m; strong
event improvements are 0.071559/0.019204 m. This subset result is positive but
not paper-ready. See `debug_record.md` and
`results/innovation_decomp_selfinfo_train32_val16_epoch9_compare.json`.

## 2026-08-10: innovation diagnostics frozen and formal multi-seed run started

The NIS/statistics evaluator now reports current-step chi-square(2)
calibration and strictly causal t-1 group-common/individual innovations by
steady, switch and strong event.  A complete 200-validation-trajectory pass
on the development Expert gives overall NIS mean 2.280806, PIT-KS 0.039540
and p95 coverage 0.925169.  Strong maneuvers are the least calibrated (NIS
2.545784, PIT-KS 0.085790, p95 coverage 0.904016), indicating mild
overconfidence during maneuvers.  Common/individual history NIS means are
8.433788/0.882905 overall and 9.744200/0.922578 on strong events; common
energy rises from 0.484796 overall to 0.499374 on strong events.  The
decomposition reconstructs the original innovation within 1.57284e-5 m.

The current SEAN development checkpoint is not selective: a 16-trajectory
pass gives mean edge weight 0.684156 and 8.495328 effective neighbors out of
8.508704 candidates.  The formal no-SEAN fixed-graph model is consequently a
mechanistic control, not an optional ablation.

Frozen protocol:

- specification: `docs/INNOVATION_FORMAL_PROTOCOL.md`
- manifest: `results/formal_innovation_protocol_v1.json`
- protocol SHA-256:
  `2b99127d5d53bd3362045ff20937223a072b9ffa7f5d77c2800538d550e5e381`
- groups: A Temporal-only from scratch, B no-SEAN fixed graph from same-seed
  A, C full SEAN whitened common/individual decomposition from same-seed A;
- seeds 123/456/789, all 1000 train and 200 validation trajectories, batch 4,
  all-time causal supervision, streaming Encoder, CV covariance;
- IID 300 and OOD 400 remain untouched until all checkpoints are selected.

The independent sequential runner started at 2026-08-10 19:36 CST (PID
26900) and is currently in seed-123 A Temporal.  Monitor
`log/formal_innovation_v1_seed123_A_temporal.log` and
`checkpoint/formal_innovation_v1`.  It automatically proceeds A -> B -> C,
runs full validation and NIS evaluation, then advances to the next seed.  Two
earlier hosted-launch attempts produced no checkpoint and are preserved as
`checkpoint/_aborted_formal_innovation_v1_20260810_182225` for audit only.

## 2026-08-12: group-common maneuver oracle pivot

The user stopped the formal multi-seed run to test the group-information upper
bound.  Runner PID 26900 and active seed-456 child PID 33984 were stopped; no
formal training remains.  Seed123 A/B/C is complete, while seed456 A is only a
partial audit artifact and must not be treated as complete by a future
launcher.  The evaluator now honors `args.gasf_graph_mode`, which also means
the former frozen protocol source hash must be re-audited/re-frozen before
formal training resumes.

`evaluation/evaluate_group_maneuver_oracle_v3.py` evaluated the seed123
epoch-48 Temporal checkpoint on all 200 validation trajectories.  Scalar/2x2
maps were calibrated on trajectories 0--99 (84,934 valid node-steps) and
reported only on held-out trajectories 100--199 (87,776 steps).  Topology is
from t-1, target future truth is excluded, and only other targets' truth at t
is exposed for this deliberately non-deployable upper bound.  The first pass
found the IEEE `0 * NaN` lifecycle aggregation issue; invalid sources are now
zeroed before aggregation.  Three regression tests pass and the final JSON
contains no non-finite value.

Held-out Temporal position/velocity prediction is 17.6636 m / 3.1614 m/s.
True-group mean delta-v is physically coherent (96.22% linear explained
fraction overall; 97.73% and cosine 0.9563 on strong events), but its calibrated
state injection improves global position by only 0.0186 m and worsens strong
position by 0.0100 m.  In contrast, the true-group shared-prediction-error
oracle improves position by 1.8311 m overall (10.37%) and 3.4414 m on strong
events (17.05%).  Geometry alone gives 1.4734/2.5113 m; perfect true-group
topology adds 0.3577/0.9302 m beyond geometry.

Conclusion: do not abandon group information, but do not continue direct
group delta-v/ODE correction.  The missing target is common prediction error
or common whitened innovation.  Before another long training run, measure a
strictly causal lagged peer-innovation oracle.  If it retains meaningful gain,
build a robust common-innovation predictor; if it collapses, limit the group
claim to measurement update/uncertainty improvement.  This oracle uses future
peer truth and is diagnostic only, not a paper result.

Artifact: `results/group_maneuver_oracle_val200_seed123.json`.

## 2026-08-12: strictly causal group-innovation oracle confirms the direction

The seed123 epoch-48 Temporal checkpoint was rolled out on all 200 validation
trajectories.  Prediction features at t end strictly at t-1 after the prior
measurement update; no current measurement/truth is used.  Trajectories 0--99
calibrate small no-intercept ridge maps and 100--199 are held out.  Main
metrics use the same 87,776-step population for every history length by
leaving insufficient-history targets at the Temporal prediction.

Eight raw-innovation frames are the best position-first setting.  True-group
own+leave-one-out-peer histories reduce held-out position RMSE from 17.6636 to
16.0390 m, a 1.6246 m / 9.20% upper-bound gain; trajectory bootstrap 95% CI is
[1.2979, 1.9763] m.  On strong events, 20.1817 becomes 17.6611 m, a 2.5206 m
/ 12.49% gain, CI [1.8035, 3.2784].  Own history alone gains only 0.0051 m
globally and loses 0.0919 m on strong events.  Event/validity-stratified
shuffled peers gain only 0.0038/-0.0923 m, proving the aligned peer signal is
not an event-marginal or fitting artifact.

Geometry alone gives 1.0728 m global and 1.4012 m strong gains.  Perfect
true-group topology adds 0.5518 m globally (CI [0.2648, 0.9018]) and 1.1194 m
strong (CI [0.4121, 2.0026]) beyond geometry.  Causal eight-frame information
retains 88.7%/73.2% of the prior contemporaneous global/strong peer-error
ceiling.  Group structure is therefore useful for prediction, not merely the
measurement update.

Paired bootstrap selects eight over four raw frames by 0.1170 m globally (CI
[0.0210, 0.2040]) and 0.1352 m strong ([0.0041, 0.2506]).  Sixteen frames do
not help and significantly worsen strong-event performance.  Eight-frame raw
innovation is also substantially better for position than whitening alone
(1.6246 versus 1.0138 m global gain), although uncertainty context should be
retained for calibration.

Next implementation should not reuse the current `common = own + peers`,
`individual = own - common` white-only decomposition.  Freeze Temporal and
feed separate eight-frame raw-own and leave-one-out peer-consensus channels,
with whitened/NIS/log-S context, into a zero-initialized bounded normalized
prior-state correction.  Start with fixed geometry plus exact alpha=0
counterfactual protection on 32/16; test SEAN only after the fixed graph shows
closed-loop global and strong position gains.  Do not resume the formal
multi-seed runner before this mechanism passes development validation and the
protocol is re-frozen.

Artifacts: `results/causal_group_innovation_oracle_val200_seed123.json`,
`results/causal_group_innovation_oracle_val200_seed123_bootstrap.json`, and
the reusable tensor cache
`results/causal_group_innovation_oracle_val200_seed123_tensors.pt`.

## 2026-08-12: causal innovation corrector passes development gate; formal run active

Work is on branch `feature/sggt-v2-causal-innovation`, based on `e192aa6`.
The implemented corrector stores eight strictly causal frames of separate raw
own innovation and same-frame leave-one-out geometric-peer innovation. It uses
whitened innovation, NIS and S covariance only as uncertainty context, then
applies a bounded normalized state correction after the frozen Temporal prior
and before EKF measurement update. The output is zero-initialized; alpha 0 is
an exact Temporal counterfactual. Covariance propagation remains unchanged.

Nineteen targeted tests passed. A real V9 CUDA smoke completed without NaN,
and all 167 shared non-corrector tensors remained bitwise unchanged. Only
12,996 corrector parameters are trainable.

On 32 train / 16 checkpoint-selection trajectories, the position-prioritized
version reduced prediction/update RMSE from 18.8564/14.9320 to
18.1651/14.5370 m. Same-checkpoint alpha 0/1 on untouched validation
trajectories 16--47 reduced prediction/update from 18.0622/14.3450 to
17.2609/13.8565 m. Strong/switch position gains were 0.8911/1.0266 m, but
velocity regressed by 0.0319/0.0400 m/s. Alpha 0.25 retained 0.2923/0.1787 m
position gains with only 0.0011/0.0028 m/s velocity regression.

A direct-velocity removal and a local one-step velocity guard were rejected.
Equal physical position/velocity training was more stable: untouched
trajectories retained 0.7598/0.4624 m position gains with smaller, though still
nonzero, 0.0269/0.0359 m/s velocity regressions. The formal protocol therefore
uses equal weights and reports this Pareto rather than hiding it.

Formal seed123 training is running in managed cell `1698` with Python PID
38056. Dataset sizes are train/val/IID/OOD = 1000/200/300/400. Configuration:
all-time supervision, batch 4, stream encoder, CV covariance, Temporal epoch-48
checkpoint frozen, corrector trained from zero, AdamW LR 1e-4, one warmup epoch,
cosine decay, maximum 20 epochs, internal min-delta 0.005 m and patience 4,
checkpoint metric `position`. Full 200-trajectory epoch -1 prediction/update
baseline is 18.378831/14.503282 m, matching the old Temporal evaluation within
1.2e-5 m. The run is currently in epoch 0.

Formal output: `checkpoint/causal_innovation_formal_seed123`; log:
`log/causal_innovation_formal_seed123.log`; launcher:
`scripts/run_causal_innovation_formal.ps1`. After convergence, evaluate
`best_position.pth` with alpha 0 and 1 on the complete fixed validation set and
report global plus steady/switch/strong metrics before starting another seed or
adding SEAN.

## 2026-08-12: expert converged; causal activation training active

The formal expert stopped normally at epoch 17. On all 200 validation
trajectories its best-position checkpoint improved Temporal position
prediction/update from 18.3788/14.5033 to 17.0437/13.7934 m and slightly
improved both velocity endpoints. Fixed alpha 0/1 checks on 32 validation,
IID and OOD trajectories all showed more than 1.1 m prediction gain.

Do not use the instantaneous branch-winner label for activation. A controlled
32/16 probe left BCE at 0.693 and balanced accuracy at 0.503, demonstrating
that the next-step winner is dominated by unpredictable current noise. The
implemented selector instead learns a strictly causal continuous target: a
0.5 floor plus past-window NIS excess and positive own/leave-one-out-peer
whitened-innovation agreement. Its calibrated chi-square(2) NIS anchors are
1.386 (median) and 5.991 (p95). The development run lowered target loss from
0.0308 to 0.0231 and beat fixed alpha 0.5 in all four tracking endpoints.

Formal selector training runs in managed cell `1771`, PID 30652. Output/log:
`checkpoint/causal_innovation_gate_formal_seed123` and
`log/causal_innovation_gate_formal_seed123.log`. Only 3,137 selector parameters
are trainable; all-time supervision, batch 4, streaming encoder, CV covariance,
LR 1e-3, maximum 20 epochs, position min-delta 0.05 m and patience 4 are used.
Epoch -1 full-validation prediction/update is 17.3680/13.9439 m. After early
stop, compare dynamic activation against the same expert checkpoint at fixed
alpha 0, 0.5 and 1 on full validation/IID/OOD, then report event-stratified
steady/switch/strong metrics and causal-target MAE/correlation.

## 2026-08-12: formal causal activation result

The selector run finished normally after epoch 4 (about 80 minutes) because
the 0.05 m position plateau criterion accumulated four non-significant epochs.
There was no NaN or process failure. Exact position selection chose epoch 1:
prediction/update RMSE 17.3078/13.9123 m and velocity prediction/update RMSE
3.1458/3.0057 m/s. Gate selection chose epoch 4; its tracking result was within
0.001 m of epoch 1 while causal-target MAE/correlation improved to
0.0261/0.7307. Thus tracking plateaued before the auxiliary selector target.

On the same complete 200-trajectory validation rollout, fixed alpha 0.5 gave
17.3692/13.9445 m. Dynamic activation therefore adds only 0.0614/0.0322 m over
the equal-strength fixed control. The improvement is event ordered: prediction
gain over alpha 0.5 is 0.0419 m steady, 0.0632 m switch and 0.0871 m strong.
Among nodes with all eight own/peer history frames, activation mean is 0.5417,
0.5455 and 0.5490 for those event classes. This is consistent with demand
modulation, but the absolute modulation and incremental tracking contribution
are modest.

The event evaluator was corrected to read `causal_innovation_activation`
instead of the unrelated legacy GASF confidence and to exclude nodes without
a complete causal history from activation statistics. RMSE values were not
affected. Use
`evaluation_results/causal_gate_formal_seed123_val200_dynamic_best_position_v3.json`
for event activation reporting; v1/v2 event-activation fields are obsolete.

For the position-primary paper endpoint, the fixed alpha-1 expert remains best
at 17.0437/13.7934 m. The dynamic selector improves the position/velocity
Pareto and is a useful mechanism ablation, but it must not be presented as a
large independent gain until full IID/OOD paired evaluation and multi-seed or
trajectory-bootstrap uncertainty confirm the 0.03--0.09 m incremental effect.

## 2026-08-13: complete IID/OOD activation controls and trajectory bootstrap

The formal seed123 expert and dynamic-selector checkpoints were evaluated by
closed-loop online rollout on all 300 IID and 400 OOD test trajectories. Four
operating modes were compared: same-expert alpha 0 (exact Temporal
counterfactual), fixed alpha 0.5, fixed alpha 1, and learned dynamic
activation. The dynamic checkpoint differs from the expert only by four
selector tensors; all 190 shared state-dict tensors are bitwise identical.

IID prediction/update position RMSE is 17.9658/13.9820 m for Temporal,
16.8207/13.3524 m for alpha 0.5, 16.7348/13.3072 m for dynamic activation,
and 16.3719/13.1329 m for alpha 1. OOD results in the same order are
22.9874/16.8820, 21.6544/16.1545, 21.5032/16.0768, and 20.9332/15.7855 m.

Trajectory-paired 10,000-sample bootstrap resamples pooled SSE/counts, not the
unweighted mean of per-trajectory RMSE. Dynamic versus fixed alpha 0.5 gains
0.0859 m IID and 0.1512 m OOD prediction RMSE; 95% CIs are
[0.0651, 0.1066] and [0.1244, 0.1802]. Strong-event gains are larger at
0.1446 [0.1032, 0.1887] and 0.2039 [0.1665, 0.2402] m. Thus the dynamic
effect is statistically resolved but modest. Alpha 1 remains significantly
better than dynamic by 0.3629 [0.2493, 0.4740] m IID and
0.5700 [0.4441, 0.6891] m OOD prediction RMSE.

Eligible activation means follow the intended event ordering: IID
steady/switch/strong = 0.5421/0.5450/0.5492; OOD =
0.5469/0.5536/0.5707. The selector therefore responds to maneuver demand and
distribution shift, but its narrow range leaves useful position correction
unused. For the position-primary main model, alpha 1 is supported as the
native learned-correction scale (no post-hoc attenuation), while dynamic
activation should currently be reported as a robustness/mechanism ablation.

An evaluation bug was prevented before the formal run: PyG automatically
increments graph attributes whose names contain `index`, so batched
`trajectory_index` values were not valid IDs. The V3 loader now emits
`trajectory_id`, and a four-trajectory audit verified that summed trajectory
SSE reproduces every global RMSE to within 2e-7. Thirteen related tests pass;
all eight IID/OOD reports contain no non-finite values.

Detailed tables and artifact paths are in
`results/causal_activation_formal_seed123/IID_OOD_SUMMARY.md`. The resumable
runner is `scripts/run_causal_activation_iid_ood_bootstrap.ps1`.

## 2026-08-13: association-aware CGIC implementation and paper gate

The causal corrector now accepts soft-association reliability context:
detection mass, normalized entropy, and candidate-spread eigenvalues. It uses a
partial-history rule (at least four valid paired frames in eight, including one
of the latest three) and supports an exact own-only peer ablation. Training can
simulate identity-consistent missed detections and local ambiguous candidates;
global clutter and cross-target exclusion remain reserved for the complete BP
tracker, whose birth/death/reacquisition logic is required for a meaningful
test.

The adopted BP tracker now exposes its soft marginals to SGGT. The adapter uses
Gaussian-mixture moment matching for the soft innovation and uncertainty
statistics; hard maximum association is no longer used for the CGIC history.
The graph builder was also corrected to slide state history once per frame and
to store measurement-minus-current-prior innovations. The original BP
algorithm remains an adopted backend and is not claimed as a contribution.

Seventy-eight tests pass. Real V9 CPU/GPU smoke, backward, and optimizer checks
are finite. Diagnostic no-perturbation augmentation exactly recovers the EKF;
miss-only, local-clutter-only and their combined paths remain finite. The 32/16
fixed-association development run improved prediction/update RMSE by only
0.0139/0.0073 m, so it passed numerical stability but did not yet establish a
paper-scale effect.

The pre-registered 200/50 run uses clean and fixed-augmentation validation,
seed 20260813 for perturbations, and selects
`best_association_position.pth`. Its manifest is built after subsetting and
must contain exactly 50 validation trajectories. Epoch -1 is permanently
stored in `initial.pth` and `epoch_minus_one_metrics.json`; it cannot be
overwritten by epoch 0. On completion,
`evaluation/assess_association_cgic_gate.py` writes
`results/association_cgic_200_50_gate.json`. Formal training cannot start
unless that file approves at least 0.05 m fixed-association prediction gain,
non-worse association update, clean regression below 0.05 m, finite metrics,
and a nonzero nonsaturated correction.

The guarded formal runner is
`scripts/run_formal_association_cgic_multiseed.ps1`. It compares own-only,
own+peer (main), and own+peer without association context for seeds
123/456/789, but also requires each seed's frozen Temporal checkpoint. At this
handoff seed 789 Temporal is not available, so do not bypass the check; train
that seed under the frozen Temporal protocol first if the development gate
passes.

Paper working files are under `docs/paper_revision/`. They deliberately leave
numerical tables blank. The proposed exposition gives BP a concise adopted
backend section, then explicitly separates standard BP from the new soft
association-to-causal-innovation interface. Do not overwrite the rejected
submission or reuse its old robustness/OSPA numbers.

## 2026-08-13: physical baselines and complete-BP evaluation implemented

The common-protocol CV and two-mode IMM evaluators are now implemented in
`evaluation/evaluate_physical_baseline_v3.py`. Both run an online
predict--measurement-update--posterior-feedback recursion, use the corrected
`real_mask` lifecycle semantics, and exclude birth frames from forecast
metrics. `evaluation/tune_physical_baselines_v3.py` selects process-noise
settings using validation prediction+update position RMSE only; IID/OOD test
sets never participate. A four-trajectory smoke run produced finite,
positive-semidefinite CV and IMM rollouts, and the corrected IMM NIS uses the
pre-measurement fused prior rather than its posterior.

The complete BP adapter was fixed so a neural motion prior is represented for
every active legacy group-partition hypothesis. The actual detection/clutter
generator is independent of the tracker's fixed nominal assumptions, and the
mismatch matrix disables adaptive Beta-Pd learning. A complete 60-frame smoke
run now passes. It exposed a legacy track-management limitation: only the
initial confirmed tracks were maintained and later births were mostly missed.
Consequently formal output now separates OSPA localization/cardinality,
matched position/velocity RMSE, NIS/NLL, false/missed tracks, fragmentation,
identity switches and runtime. The robustness runner covers all three
generated scenes and pools state metrics from sufficient statistics rather
than averaging per-scene RMSE.

The pre-registered 200/50 association-CGIC development run completed all ten
epochs. Fixed-association checkpoint selection chose epoch 5: its
prediction/update RMSE improved from the immutable epoch -1 values
29.0497/26.3433 m to 28.6266/26.1121 m. On clean validation the same checkpoint
improved 18.4112/14.6296 m to 17.8308/14.2927 m. Correction RMS/max was
1.0739/3.7165 m against the 50 m bound. All machine checks passed in
`results/association_cgic_200_50_gate.json`. These remain development-gate
results and must not be copied into the paper's formal result tables.

## 2026-08-13: formal pipeline implementation and protocol audit

The missing seed-789 Temporal run initially stopped during epoch -1 because
its hidden launcher remained attached to the command-execution job. That
partial run contained no trained checkpoint. It also exposed a metadata
regression in `main_v3.py`: the protocol writer had changed from the seed
123/456 version-2 schema to version 1, replacing the frozen CV-covariance field
with a redundant weighting label. The incomplete directory and logs were
initially moved to `checkpoint/incomplete_archives/seed789_protocol_v1_20260813_1750`.
They were later deleted during the 2026-09-06 workspace cleanup after the
repaired protocol and the absence of a trained checkpoint had been verified.

The protocol writer now reproduces the seed-123/456 version-2 manifest and
hash `36e5157d...b91346f` exactly, including full-time supervision, streaming
encoder execution and CV covariance. A strict verifier checks architecture,
state-dict shape, finite checkpoint metrics and identical validation protocol
across seeds before formal work begins. Seeds 123 and 456 pass this audit; seed
789 must be trained again under the repaired protocol.

`scripts/run_paper_experiment_pipeline.ps1` is the resumable formal entry. It
first completes missing Temporal seeds, then verifies all three initializers,
freezes the code/data manifest, tunes CV/IMM only on validation, trains
P3/P4/P5, evaluates clean and fixed-association val/IID/OOD, runs trajectory
bootstrap, and finally compares representative-seed Temporal and full CGIC in
the full BP Pd/clutter matrix. BP reports use model-specific directories and
reject stale checkpoint hashes, so the two models cannot overwrite each
other. Ninety-two unit tests pass after this audit.

The final pre-launch audit increased the passing count to 93 by adding the
controlled scaling benchmark test. The immutable formal manifest was created
at `results/formal_association_cgic_protocol_v1.json`: protocol SHA-256
`63b0ee5c33d4a8df6ad4c8802293def1e61ce737cd8ba319f12be95a6f1b917e`,
code-manifest SHA-256
`f03d17d6e926dbf644b2add96d2a6e43568cedf862bf7ebfab490baa5c7f00c9`,
with train/val/IID/OOD counts 1000/200/300/400. Do not modify a manifest-listed
source file during the formal run; verification will intentionally fail.

The detached Windows task `SGGT_Formal_Paper_Experiments_20260813` started at
2026-08-13 20:47 Asia/Shanghai. At launch, seed-789 Python PID 16608 was alive,
responsive, using about 4.7 GiB host RAM and 2.3 GiB GPU memory. Its newly
written validation protocol is version 2 with hash
`36e5157d5c49cc99c29f7b650e32a7f3c0347770c19d7839142a03a81b91346f`,
identical to seeds 123/456. The pipeline transcript is
`log/formal_association_cgic_pipeline.log`; the current trainer log is
`log/formal_innovation_v1_seed789_A_temporal.log`. The first stage is a full
200-trajectory epoch -1 validation, so absence of a best checkpoint during
that stage is expected.

## 2026-08-19: paper-revision evidence freeze and first complete draft

The formal seed-123 evidence was audited into
`results/paper_revision_20260819`. The audit covers 22 clean/association
tracking reports, 240 trajectory-bootstrap rows, 80 NIS rows, all 18 BP
model-condition summaries, and 8 complexity rows; `RESULT_AUDIT.json` reports
`pass` with no errors. Hierarchical BP bootstrap preserves the
scene-to-Monte-Carlo nesting instead of treating repeated trials as independent
trajectories.

The manuscript is now organized under `docs/paper_revision`, with the fully
inlined `GNN_IEEE_REVISED_SINGLE.tex` as its sole TeX source. The revised narrative treats
factor-graph BP as an adopted soft-association front end and presents the new
mechanism as strictly causal own/leave-one-out group-common innovation
correction (CGIC), not as a list of SEAN/GASF/ODE/EKF modules. It explicitly
removes group-label supervision claims and reports the weak association-context
increment and NIS/coverage limitation. `RESPONSE_LETTER_20260819.md` is the
matching reviewer-response draft.

The result and manuscript static audits pass. A compiled preview PDF is available
because the supplied paper directory has no `ref.bib`, the IEEE class is not
stored locally, and `pdflatex` is unavailable on this machine. The current
learned-model evidence is seed 123 only. Per the user's instruction to use one
seed for now, no additional long training was launched. Before final submission,
train/evaluate P4 from the existing seed-456 P2 initializer, train P2/P4 for
seed 789, freeze the cross-seed claim, restore the bibliography/template, and
compile plus visually inspect the paper.

## 2026-08-23: causal adaptive-Q mechanism rejected after matched-mean tests

A new `causal_nis` process-noise adapter was implemented with separate
acceleration-multiplier and direct-position-Q heads, using only strictly lagged
innovation/NIS history. Its parameterization, gradients, real-V9 CUDA path, and
25 SGGT-V2 unit tests pass.

The decisive validation is not dynamic Q versus the old Q setting, but dynamic
Q versus a fixed Q with exactly the same marginal mean. On independent
`val[50:200]`, both a positively supervised Q/NIS response and a tracking-only
response were worse than their matched-mean fixed-Q counterfactual. Removing
the positive surprise loss did substantially improve tracking by calibrating
the average operating point to approximately acceleration multiplier 6.87 and
per-axis position Q 1.32 m^2, but its residual time variation did not help.

Do not claim that the present model causally enlarges Q during maneuvers. Treat
the result as fixed covariance calibration unless a future per-step
counterfactual Q-regret target demonstrates a dynamic advantage. Detailed
tables, bootstrap intervals, artifacts, and code decisions are in
`results/scene02_diagnosis_20260823/P4_q/CAUSAL_Q_VALIDATION_SUMMARY.md`.

## 2026-08-26: scene02 aggregate-only Monte Carlo MSE suite

The fixed calibrated-Q scene02 protocol was rerun/aggregated with actual
Pd/clutter conditions `(0.99,3)`, `(0.90,3)`, `(0.80,3)`, `(0.99,2)`,
`(0.99,10)`, and `(0.99,15)`. The standard model and tracker nominal point
remain Pd=0.99, clutter=2, with range noise 3 m and azimuth noise 0.01 degrees.
Each condition uses 20 Monte Carlo runs over all 100 frames.

Results are under `results/paper_scene02_mc_mse_20260826`. Only per-frame
Monte Carlo MSE curves and their global RMSE are retained; the NPZ contains
only `mc_mse_curve` and `final_rmse` arrays, and temporary new-condition
per-run reports were deleted after aggregation. The previous 2026-08-24
CURVES/CSV/NPZ artifacts were also deleted by explicit request.

P4's standard OSPA/position/velocity RMSE is 12.8540/5.6549/3.6374. Detection
degradation at clutter 3 is monotone: OSPA RMSE is 13.0992, 17.7027, and
23.4880 at Pd 0.99, 0.90, and 0.80. At Pd 0.99, clutter 2/10/15 gives OSPA
RMSE 12.8540/18.2894/24.5469; the degradation is mainly cardinality/false
tracks, while matched position RMSE remains 5.6549/5.9715/5.9609.

## 2026-09-04: P1 MTP-GO-backbone training

P1 is now defined as the current graph-temporal encoder and Neural-ODE/EKF
backbone with both proposed residual mechanisms removed.  Its frozen switches
are `model_variant=sggt_v2`, `decoder_use_GASF=false`,
`decoder_use_SEAN=false`, `innovation_decomposition=false`, and
`causal_innovation_correction=false`.  It retains the legacy hidden-state Q
path and uses CV covariance propagation, matching the current efficient formal
protocol.  IMM must remain a separate classical reference and must not be
called P1 in the learned-model ablation table.

The seed-123 200/50 gate used six epochs, batch size 4, all legal causal steps,
the streaming encoder, AdamW with peak LR `6e-4`, one warm-up epoch, and a
prediction-RMSE checkpoint criterion.  It ran from 13:05 to 13:56.  Validation
prediction/update RMSE improved from 30.2350/21.4149 m at epoch -1 to the best
25.1614/17.7893 m at epoch 3; velocity prediction/update RMSE at that point was
5.5082/4.1294 m/s.  Epoch 5 ended at 25.4472/17.9708 m.  The strict gate found
no nonfinite state or metric, verified exactly 50 validation trajectories, and
confirmed that no GASF or causal-corrector parameter was present.  The report
is `results/formal_mtp_go_v1/seed_123/P1_mtp_go_smoke_200_50_gate.json`.

After that gate passed, a separate from-scratch full-data P1 run started at
13:56 on train/validation counts 1000/200.  Its validation protocol hash is
`36e5157d5c49cc99c29f7b650e32a7f3c0347770c19d7839142a03a81b91346f`, the
same frozen full validation protocol used by the earlier formal experiments.
The run uses at most 60 epochs, five warm-up epochs, patience 8, and a 0.05 m
minimum improvement.  At handoff it is running as Python PID 22136 under the
scheduled task `SGGT_P1_MTP_GO_20260904`.

- Entry: `scripts/run_p1_mtp_go_training.ps1`
- Integrity gate: `scripts/validate_p1_mtp_go_run.py`
- Full checkpoint root: `checkpoint/formal_mtp_go_v1/seed_123/P1_mtp_go_formal`
- Full trainer log: `log/formal_mtp_go_v1/seed_123/P1_mtp_go_formal.log`
- Pipeline transcript: `log/formal_mtp_go_v1/seed_123/pipeline_transcript.log`
- Branch/commit at launch: `feature/sggt-v2-causal-innovation`,
  `e192aa680069673ec93bd47d153f989fcc6bf930` (dirty worktree; preserve all
  unrelated user changes).
