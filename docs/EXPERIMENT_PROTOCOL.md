# SGGT-V2 Unified Experiment Protocol

## Frozen evaluation rules

- Dataset split: the existing `train` / `test` directories (400 / 100 trajectories).
- Evaluation uses teacher forcing probability 0.
- Primary reporting mode: full closed-loop trajectory rollout.
- Fixed-window diagnostics: 8-step windows at the beginning, middle, and end.
- Structure threshold: 0.5. Threshold scans are diagnostic only and must not tune on the test split.
- Maneuver interval: within 4 frames of `param_change_time`.
- Checkpoint selection: composite validation trajectory RMSE, not total Gaussian NLL.
- Report seed and use the same split, window policy, and evaluation command for every ablation.

## Unified evaluation command

```powershell
& 'H:\Anaconda3\envs\py310_torch\python.exe' evaluation\run_experiment.py `
  --checkpoint checkpoint\v2_formal_seed123\GNN_groupTargetTracking\2026_07_19_02_52_.pth `
  --experiment_name v2_full_seed123 `
  --output evaluation\results\v2_full_seed123.json `
  --mode full `
  --batch_size 1 `
  --seed 123
```

Use `--mode both` to additionally evaluate the three fixed windows. Use
`--max_batches 1` only for smoke tests. For publishable inference timing, use
`--batch_size 1`; larger batches are suitable for faster diagnostic runs.

Checkpoint metadata automatically selects:

- `sggt_v1` or `sggt_v2`;
- SEAN enabled/disabled;
- GASF enabled/disabled;
- `ode_jacobian` or `cv` covariance propagation.

## Result schema

Each JSON file contains:

- `metadata`: checkpoint, epoch, model switches, seed, environment, and parameter counts;
- `config`: exact evaluation CLI configuration;
- `results.<mode>.global`: position/velocity prediction and update RMSE, measurement RMSE, and Gaussian NLL;
- `results.<mode>.structure`: class balance, confusion metrics, F1, specificity, balanced accuracy, MCC, and threshold scans;
- `results.<mode>.covariance`: eigenvalue, symmetry, trace, log-determinant, and non-positive counts;
- `results.<mode>.scenarios`: maneuver/non-maneuver and structure-change/stable metrics;
- `results.<mode>.timing`: forward latency;
- `resources`: wall time and peak CPU/CUDA memory.

## Core ablation names

Use stable experiment names:

- `v2_full_seed123`
- `v2_no_sean_seed123`
- `v2_no_gasf_seed123`
- `v2_no_sean_no_gasf_seed123`
- `v2_cv_covariance_seed123`

After seed 123 is complete and verified, repeat with seeds 456 and 789.
