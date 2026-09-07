# Previous-ABC GASF recurrence validation (2026-09-07)

## Scope

This experiment tests only recurrence of the preceding per-scale low-pass
parameters. It does not retain or consume the preceding spectral latent.
For scale $L$, the implementation is

\[
\widehat{\boldsymbol\theta}_{k,L}=\phi_L(E_{k,L},H_k),\qquad
\alpha_{k,L}=\sigma\!\left(\psi_L(E_{k,L},H_k,
\boldsymbol\theta_{k-1,L},v_{k,L})\right),
\]

\[
\boldsymbol\theta_{k,L}=(1-\alpha_{k,L})
\boldsymbol\theta_{k-1,L}+\alpha_{k,L}
\widehat{\boldsymbol\theta}_{k,L},\qquad
\boldsymbol\theta=[a,b,c].
\]

The state is reset when a target is invalid. A newly born target bypasses the
recurrence and uses the current candidate. Recurrence is disabled by default
until experimental evidence supports it.

Branch: `feature/gasf-recurrent-abc-param-only`

Shared source checkpoint:
`checkpoint/formal_innovation_v1/seed_123/A_temporal/GNN_groupTargetTracking/best_prediction.pth`
(epoch 48, SHA-256
`c2c0dfdde7ccef1a7d892459c1e4a1a5d5c96cbc539f771ab09e862f9d359022`).

## Implementation and resource checks

- The recurrence head contains 514 parameters for two scales.
- Runtime state contains only three FP32 values and one Boolean validity value
  per target and scale. With two scales this is $26N$ bytes, or 1,664 bytes
  for 64 targets.
- The abandoned latent-plus-parameter prototype required 34,432 bytes for 64
  targets; parameter-only state reduces this by 95.2%.
- Unit tests cover exact interpolation, gradient flow, invalid/reborn reset,
  state transport, state-independent disabled recurrence, and exact recovery
  of the feed-forward filter at \(\alpha=1\).
- Real-data 8/4 smoke training completed without NaN/Inf. Covariance and
  backward checks passed.

## Protocol diagnostics

An initial 32/16 experiment fine-tuned the complete network. Both arms became
worse than epoch -1, so it was rejected as evidence about recurrence. Resetting
teacher forcing to zero did not remove the degradation. The cause was updating
the complete converged model on only 32 training trajectories. The final
mechanism test therefore froze the Temporal GASF, Encoder, ODE, fusion, Q, and
EKF paths and optimized only the 514 recurrence-head parameters.

The training entry point now records configurable teacher-forcing start,
minimum, and decay values so checkpoint adaptation cannot silently reuse the
from-scratch schedule.

## Recurrence-head-only training (32 train / 16 validation)

Configuration: seed 123, batch 4, full-time causal supervision, streaming
Encoder, CV covariance transition, closed-loop training, AdamW, maximum
learning rate $6\times10^{-4}$, and early-stopping minimum improvement
0.05 m.

| Model | Best epoch | Position prediction RMSE (m) | Position update RMSE (m) | Velocity prediction RMSE (m/s) |
|---|---:|---:|---:|---:|
| R0 feed-forward ABC | frozen | 18.85492 | 14.93125 | 3.41113 |
| R1 previous-ABC recurrence | 4 | 18.87796 | 14.94080 | 3.41762 |

Relative to its own epoch -1 value, R1 improved position prediction by only
0.00005 m. Relative to R0 it was worse by 0.02304 m. Its valid-transition
coefficient was 0.10087 on average. The strong/steady coefficient ratio was
1.00061, so the learned coefficient did not distinguish maneuver demand.
The frequency-response step fell from 0.04493 to 0.00783 (82.6% reduction),
showing that the mechanism smooths the filter but not that the smoothing helps
tracking.

Artifacts:

- `results/dev_gasf_recurrent_abc/seed_123/filter_only_32_16/comparison.json`
- `checkpoint/dev_gasf_recurrent_abc/seed_123/filter_only_32_16/`

## Fixed-alpha causal scan (50 validation trajectories)

This scan uses the same frozen checkpoint and 50 fixed validation trajectories
(41,827 valid target-time points). No parameter is trained. Alpha 1 exactly
recovers the feed-forward ABC filter.

| Alpha | Position prediction (m) | Position update (m) | Velocity prediction (m/s) | Steady position prediction (m) | Strong position prediction (m) |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 18.42704 | 14.63645 | 3.10137 | 17.81412 | 20.59298 |
| 0.25 | 18.42151 | 14.63377 | 3.10042 | 17.81161 | 20.58159 |
| 0.50 | 18.41784 | 14.63250 | 3.09993 | 17.80695 | 20.57922 |
| 0.75 | 18.41518 | 14.63160 | 3.09955 | 17.80105 | 20.58126 |
| 1.00 | **18.41123** | **14.62963** | **3.09879** | **17.79195** | 20.58447 |

The overall position metrics improve monotonically as memory is removed. Alpha
0.5 improves strong-maneuver prediction by only 0.00525 m relative to alpha 1,
while worsening steady prediction by 0.01500 m. This trade-off is too small and
has negative overall value. The measured inference time was 1.240 ms per valid
node-step without recurrence and 1.258 ms at recurrent alpha 1 in these runs;
the approximately 1.5% difference is descriptive rather than a precision
benchmark.

Artifacts:

- `results/dev_gasf_recurrent_abc/seed_123/alpha_sweep_val50/summary.json`
- `results/dev_gasf_recurrent_abc/seed_123/alpha_sweep_val50/feedforward_baseline.json`

## Decision

Previous-ABC recurrence is mechanically valid and inexpensive, but it is not
supported as the main GASF design under the current candidate filter and
tracking objective. It reduces parameter and response variation without a
meaningful accuracy gain; the held-out optimum is the no-memory limit
\(\alpha=1\). Do not start 200/50 or full-data training for this variant, and do
not add it to the paper model. Keep the implementation as an off-by-default
diagnostic ablation.
