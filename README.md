# SGGT-Net

SGGT-Net is a graph-based neural state estimator for group target tracking. The
current `main` branch contains the revised SGGT-V2 architecture used by the
latest experiments. It replaces the early SEAN/GASF design described by the
old README.

## Current model

The formal model follows a causal online tracking pipeline:

1. A streaming GNN-GRU encoder represents the observed target history.
2. The Multi-scale Parametric Spectral Innovation Filter (MPSIF) processes
   causal innovation histories with short and long windows of 8 and 16 frames.
3. Each window uses a 64-point zero-padded FFT and an adaptive parametric
   low-pass response

   \[
   g(f)=\sigma(af^2+bf+c), \qquad a<0,\; b<0.
   \]

4. The filtered spectral feature is fused with the decoder representation and
   used by the Neural ODE acceleration model. RK4 performs state extrapolation.
5. The Causal Individual-and-Peer Innovation Corrector (CIPIC) can provide a
   bounded residual state correction from causal individual and peer raw
   innovation histories.
6. Process noise is predicted from the fused decoder/MPSIF representation. The
   former sigmoid scalar amplitude gate has been removed; two acceleration
   standard deviations and their correlation directly define a
   positive-semidefinite acceleration covariance.
7. An EKF propagates covariance and performs the measurement update.

The current formal configuration disables SEAN and group-structure
supervision. MPSIF operates in temporal-only mode. Data association is handled
outside the neural predictor; the BP implementation is used by complete
tracking evaluations and is not an internal layer of SGGT-Net.

## Repository layout

```text
SGGT-Net/
├── model/
│   ├── sggt_net.py              # End-to-end SGGT-V2 and EKF integration
│   ├── gru_gnn_residual.py      # Streaming encoder and revised decoder
│   ├── sggt_v2_modules.py       # MPSIF and optional graph components
│   ├── causal_innovation.py     # CIPIC implementation
│   ├── motion_models_base.py    # Neural ODE feature fusion and correction
│   └── kalman_utils.py          # EKF covariance and measurement updates
├── data/
│   ├── load_data.py             # Legacy data loader
│   └── load_data_v3.py          # Causal full-trajectory data pipeline
├── train/
│   ├── trainer_v3.py            # Full-time causal training loop
│   └── training_utils_v3.py     # Streaming and masking utilities
├── evaluation/                  # IID/OOD, event and tracking evaluation
├── config.py                    # Command-line configuration
├── main_v3.py                   # Recommended training entry point
└── main.py                      # Legacy-compatible entry point
```

## Environment

The latest experiments were run with:

- Python 3.10
- PyTorch 2.8 with CUDA 12.6
- PyTorch Geometric
- NumPy, SciPy, pandas, loguru and tqdm

Install a PyTorch build appropriate for the local CUDA driver, followed by
PyTorch Geometric and the remaining Python dependencies. Dataset files and
trained checkpoints are intentionally not stored in this repository.

## Dataset format

The current loader expects a packed trajectory dataset with separate splits:

```text
<data-root>/
├── train/
│   ├── input/
│   └── target/
├── val/
│   ├── input/
│   └── target/
└── test/
    ├── input/
    └── target/
```

`real_mask = 1` marks a target that exists at a frame. Frames before target
birth or after target disappearance use `real_mask = 0` and are excluded from
losses, filtering updates and reported RMSE values.

## Training the current architecture

The following PowerShell command captures the essential settings of the
current formal model. Replace the data and output paths as needed.

```powershell
python main_v3.py `
  --data_dir "<data-root>" `
  --output_dir "checkpoint/formal_sggt_v2" `
  --log_dir "log/formal_sggt_v2.log" `
  --device cuda `
  --seed 123 `
  --model_variant sggt_v2 `
  --train_batch_size 4 `
  --eval_batch_size 4 `
  --train_epochs 60 `
  --lr 0.0003 `
  --min_lr 0.000003 `
  --warmup_epochs 5 `
  --weight_decay 0.0001 `
  --rollout_steps 4 `
  --full_time_supervision `
  --encoder_execution_mode stream `
  --optimizer_step_mode trajectory_batch `
  --covariance_transition cv `
  --decoder_use_GASF `
  --no-decoder_use_SEAN `
  --gasf_filter_mode abc `
  --gasf_graph_mode temporal_only `
  --gasf_fusion_mode concat `
  --gasf_recurrent_filter `
  --gasf_fft_points 64 `
  --causal_innovation_correction `
  --causal_innovation_output_mode state_offset `
  --causal_innovation_state_channels full `
  --causal_innovation_minimum_valid_history 4 `
  --causal_innovation_recent_valid_window 3 `
  --no-causal_innovation_association_context `
  --causal_innovation_use_peer `
  --adaptive_q_mode legacy `
  --training_stage joint
```

Training starts with an explicit epoch `-1` evaluation of the randomly
initialized model. Checkpoint selection therefore does not confuse the initial
model with epoch 0. The recommended metric is prediction-position RMSE with
trajectory-level causal validation.

## Outputs and checkpoints

For a run named `<run-name>`, checkpoints are stored under:

```text
checkpoint/<run-name>/GNN_groupTargetTracking/
```

The principal files are:

- `initial.pth`: state before optimization;
- `best_prediction.pth`: best validation prediction-position checkpoint;
- `best_position.pth`: best validation posterior-position checkpoint;
- `last.pth`: last completed epoch.

Checkpoint metadata records the model variant, MPSIF mode, FFT length,
covariance transition and CIPIC configuration. Use those saved values when
reconstructing a model for evaluation.

## Evaluation

The repository provides evaluation utilities for:

- in-distribution and out-of-distribution trajectory prediction;
- position and velocity RMSE;
- steady, transition and strong-maneuver subsets;
- innovation and covariance diagnostics;
- complete BP-based tracking with OSPA and cardinality metrics.

Use the scripts in `evaluation/` with the selected checkpoint and the same
dataset split/protocol used during training. Do not compare checkpoints using
different masks, rollout definitions or association conditions.

## Reproducibility notes

- Train from scratch after changing the architecture.
- Record the random seed, data split, causal mask, FFT length and covariance
  transition with every result.
- Checkpoints, datasets, logs and local experiment outputs are kept out of Git.
- The current model source corresponds to commit `68b9447` or later on `main`.

## License and citation

No license or archival citation has yet been declared in this repository.
Please contact the authors before redistributing the code or using unpublished
experimental results.
