param(
    [int]$TrainSamples = 200,
    [int]$ValSamples = 50,
    [int]$Epochs = 10,
    [int]$Seed = 123,
    [ValidateSet("full", "position_only", "velocity_only")]
    [string]$StateChannels = "full",
    [double]$MaxPositionM = 50.0,
    [double]$MaxVelocityMps = 5.0,
    [string]$TemporalCheckpoint = (
        "checkpoint\formal_innovation_v1\seed_123\A_temporal\" +
        "GNN_groupTargetTracking\best_prediction.pth"
    ),
    [string]$RunName = "dev_cipic_direct_state_own_peer_seed123"
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))
$sggtPython = "H:\Anaconda3\envs\py310_torch\python.exe"
$dataRoot = "E:\Dataset\GroupTargetsDataset_dataRate_2_v9_pack"

if (-not (Test-Path -LiteralPath $TemporalCheckpoint)) {
    throw "Temporal checkpoint not found: $TemporalCheckpoint"
}

$arguments = @(
    "main_v3.py",
    "--data_dir", $dataRoot,
    "--output_dir", "checkpoint\$RunName",
    "--log_dir", "log\$RunName.log",
    "--init_checkpoint", $TemporalCheckpoint,
    "--seed", "$Seed",
    "--device", "cuda",
    "--train_batch_size", "4",
    "--eval_batch_size", "4",
    "--loader_workers", "0",
    "--debug_train_samples", "$TrainSamples",
    "--debug_val_samples", "$ValSamples",
    "--skip_final_tests",
    "--train_epochs", "$Epochs",
    "--warmup_epochs", "1",
    "--warmup_start_factor", "0.1",
    "--early_stopping_patience", "4",
    "--early_stopping_min_delta", "0.05",
    "--checkpoint_metric", "position",
    "--lr", "0.0001",
    "--min_lr", "0.000003",
    "--weight_decay", "0.0001",
    "--full_time_supervision",
    "--encoder_execution_mode", "stream",
    "--optimizer_step_mode", "trajectory_batch",
    "--train_metric_interval", "32",
    "--finite_check_interval", "16",
    "--grad_clip_norm", "5",
    "--covariance_transition", "cv",
    "--decoder_use_GASF",
    "--no-decoder_use_SEAN",
    "--gasf_filter_mode", "abc",
    "--gasf_graph_mode", "temporal_only",
    "--gasf_fusion_mode", "concat",
    "--gasf_aux_source", "fusion",
    "--no-innovation_decomposition",
    "--causal_innovation_correction",
    "--causal_innovation_history", "8",
    "--causal_innovation_hidden_size", "32",
    "--causal_innovation_output_mode", "state_offset",
    "--causal_innovation_state_channels", $StateChannels,
    "--causal_innovation_max_acceleration_mps2", "5",
    "--causal_innovation_max_position_m", "$MaxPositionM",
    "--causal_innovation_max_velocity_mps", "$MaxVelocityMps",
    "--causal_innovation_minimum_valid_history", "4",
    "--causal_innovation_recent_valid_window", "3",
    "--no-causal_innovation_association_context",
    "--causal_innovation_use_peer",
    "--no-causal_innovation_dynamic_activation",
    "--training_stage", "causal_innovation_expert",
    "--lambda_struct", "0",
    "--lambda_temporal", "0",
    "--lambda_nll", "0.0001",
    "--adaptive_q_mode", "legacy",
    "--lambda_interaction_expert", "1",
    "--lambda_interaction_base", "0",
    "--lambda_interaction_regret", "0",
    "--lambda_interaction_gate", "0",
    "--lambda_interaction_aux_expert", "0",
    "--lambda_interaction_aux_rank", "0",
    "--lambda_interaction_demand", "0",
    "--lambda_interaction_state_alignment", "0",
    "--lambda_interaction_state_improvement", "0.25",
    "--lambda_causal_velocity_guard", "0"
)

Write-Output (
    "START $RunName mode=state_offset/$StateChannels own_peer train=$TrainSamples " +
    "val=$ValSamples epochs=$Epochs seed=$Seed position_bound=$MaxPositionM " +
    "velocity_bound=$MaxVelocityMps at $(Get-Date -Format o)"
)
& $sggtPython @arguments
if ($LASTEXITCODE -ne 0) {
    throw "$RunName failed with exit code $LASTEXITCODE"
}
Write-Output "FINISH $RunName at $(Get-Date -Format o)"
