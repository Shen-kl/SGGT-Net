# SGGT-V2 Whitened-Innovation Formal Protocol v1

Date frozen: 2026-08-10 (Asia/Shanghai)

This protocol tests whether SEAN-conditioned, whitened group-common and
individual innovations improve tracking beyond temporal filtering and a fixed
geometric graph. It does not use group-structure labels, confidence gates,
direct acceleration corrections, ODE Jacobians or BP features.

## Data and seeds

- Dataset: `E:\Dataset\GroupTargetsDataset_dataRate_2_v9_pack`
- Train/validation/IID/OOD trajectories: 1000/200/300/400.
- Seeds: 123, 456 and 789.
- Every group uses all legal causal steps and the same frozen file order.
- `real_mask=0` means unborn/dead and is excluded from prediction, update,
  loss and metrics. A valid innovation additionally requires previous-frame
  existence.
- IID and OOD splits remain untouched until every seed/model/checkpoint has
  been selected using training and validation only.

The generated manifest `results/formal_innovation_protocol_v1.json` records
all trajectory paths/sizes, split hashes, source-file hashes and the complete
protocol SHA-256. The runner aborts if verification changes.

## Shared training settings

- batch size 4 for training and validation;
- full-time supervision and streaming Encoder;
- one-step Decoder and complete online prediction/update rollout;
- CV covariance propagation;
- AdamW, weight decay `1e-4`, gradient clipping 5;
- `(a,b,c)` monotone low-pass filter and concat feature fusion;
- no structure supervision and no temporal adjacency loss;
- Gaussian NLL weight `1e-4`;
- primary paper endpoint: position prediction RMSE;
- also report position update, velocity prediction/update, NLL, NIS,
  steady/switch/strong metrics, parameters and online inference time.

## Formal model groups

### A — Temporal

- GASF temporal FFT/low-pass filtering;
- no graph propagation and no SEAN;
- trained from scratch independently for each seed;
- maximum 60 epochs, peak LR `6e-4`;
- select by validation position-prediction RMSE;
- early-stop minimum improvement 0.05 m, patience 8.

### B — No SEAN fixed graph

- initialized from the same-seed A checkpoint;
- GASF temporal filtering followed by the fixed causal geometric graph;
- no SEAN and no innovation decomposition;
- Encoder/ODE are frozen; adapt GASF, concat fusion and auxiliary head;
- maximum 30 epochs, peak LR `1e-4`;
- select by validation position-prediction RMSE;
- early-stop minimum improvement 0.05 m, patience 6.

### C — Full innovation decomposition

- initialized from the same-seed A checkpoint;
- label-free SEAN soft graph;
- EKF-whitened innovation, NIS and information-weighted group-common /
  individual innovation streams;
- no activation gate: deployment uses the complete expert (`alpha=1`);
- zero-initialized feature residual; no direct acceleration correction;
- train only SEAN and the innovation expert while the online training rollout
  remains the exact Temporal comparator (`alpha=0`);
- maximum 30 epochs, peak LR `1e-4`;
- select by validation counterfactual expert loss, minimum improvement
  `1e-4`, patience 6.

## Statistical reporting

For current-step EKF NIS, use the two-dimensional chi-square reference:

- expected mean 2;
- 50/90/95/99% thresholds 1.3863/4.6052/5.9915/9.2103;
- report observed coverage, PIT mean, PIT-KS and coverage MAE.

For the strictly causal history through `t-1`, report group/individual NIS,
common energy fraction, individual-to-own norm ratio, SEAN edge weights and
effective neighbour count, edge-wise common-innovation smoothness, isolation
rate and reconstruction error. Group NIS is a maneuver score rather than a
chi-square calibration target because neighbour innovations share correlated
common motion.

Final comparisons use mean, standard deviation and per-seed values. With only
three seeds, report paired seed differences and bootstrap confidence intervals
over complete trajectories; do not rely on an asymptotic normality claim.

Runner: `scripts/run_formal_innovation_multiseed.ps1`  
Detached launcher: `scripts/launch_formal_innovation_multiseed.py`
