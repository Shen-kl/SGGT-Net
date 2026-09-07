param(
    [int[]]$Seeds = @(789),
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$sggtPython = "H:\Anaconda3\envs\py310_torch\python.exe"
$dataRoot = "E:\Dataset\GroupTargetsDataset_dataRate_2_v9_pack"

$common = @(
    "main_v3.py",
    "--data_dir", $dataRoot,
    "--device", "cuda",
    "--train_batch_size", "4",
    "--eval_batch_size", "4",
    "--loader_workers", "0",
    "--skip_final_tests",
    "--train_epochs", "60",
    "--warmup_epochs", "5",
    "--warmup_start_factor", "0.1",
    "--early_stopping_patience", "8",
    "--early_stopping_min_delta", "0.05",
    "--checkpoint_metric", "prediction",
    "--lr", "0.0006",
    "--min_lr", "0.00001",
    "--weight_decay", "0.0001",
    "--full_time_supervision",
    "--encoder_execution_mode", "stream",
    "--optimizer_step_mode", "trajectory_batch",
    "--train_metric_interval", "16",
    "--finite_check_interval", "0",
    "--grad_clip_norm", "5",
    "--covariance_transition", "cv",
    "--decoder_use_GASF",
    "--no-decoder_use_SEAN",
    "--gasf_filter_mode", "abc",
    "--gasf_graph_mode", "temporal_only",
    "--gasf_fusion_mode", "concat",
    "--gasf_aux_source", "fusion",
    "--no-innovation_decomposition",
    "--no-causal_innovation_correction",
    "--training_stage", "joint",
    "--lambda_struct", "0",
    "--lambda_temporal", "0",
    "--lambda_nll", "0.0001",
    "--adaptive_q_mode", "legacy"
)

foreach ($seed in $Seeds) {
    $root = "checkpoint\formal_innovation_v1\seed_$seed\A_temporal"
    $best = "$root\GNN_groupTargetTracking\best_prediction.pth"
    if (Test-Path -LiteralPath $best) {
        Write-Output "SKIP seed $seed Temporal; checkpoint exists"
        continue
    }
    $arguments = $common + @(
        "--output_dir", $root,
        "--log_dir", "log\formal_innovation_v1_seed${seed}_A_temporal.log",
        "--seed", "$seed"
    )
    Write-Output "START seed $seed Temporal at $(Get-Date -Format o)"
    if ($DryRun) {
        Write-Output ((@($sggtPython) + $arguments) -join " ")
        continue
    }
    & $sggtPython @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "seed $seed Temporal failed with exit code $LASTEXITCODE"
    }
    if (-not (Test-Path -LiteralPath $best)) {
        throw "seed $seed Temporal did not create $best"
    }
    Write-Output "FINISH seed $seed Temporal at $(Get-Date -Format o)"
}
