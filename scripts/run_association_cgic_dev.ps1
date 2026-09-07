param(
    [int]$TrainSamples = 200,
    [int]$ValSamples = 50,
    [int]$Epochs = 10,
    [int]$Seed = 123,
    [string]$RunName = "dev_assoc_cgic_fixedval_200_50"
)

$ErrorActionPreference = "Stop"
$sggtPython = "H:\Anaconda3\envs\py310_torch\python.exe"
$dataRoot = "E:\Dataset\GroupTargetsDataset_dataRate_2_v9_pack"
$temporalCheckpoint = (
    "checkpoint\formal_innovation_v1\seed_123\A_temporal\" +
    "GNN_groupTargetTracking\best_prediction.pth"
)

$arguments = @(
    "main_v3.py",
    "--data_dir", $dataRoot,
    "--output_dir", "checkpoint\$RunName",
    "--log_dir", "log\$RunName.log",
    "--init_checkpoint", $temporalCheckpoint,
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
    "--early_stopping_patience", "0",
    "--checkpoint_metric", "association_position",
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
    "--causal_innovation_minimum_valid_history", "4",
    "--causal_innovation_recent_valid_window", "3",
    "--causal_innovation_association_context",
    "--causal_innovation_use_peer",
    "--association_augmentation",
    "--association_validation",
    "--association_validation_seed", "20260813",
    "--association_pd_min", "0.75",
    "--association_pd_max", "0.98",
    "--association_clutter_mu_min", "0",
    "--association_clutter_mu_max", "0",
    "--association_max_clutter", "0",
    "--association_local_clutter_rate", "0.25",
    "--association_max_local_clutter", "2",
    "--association_local_clutter_scale", "2.5",
    "--no-association_include_target_competitors",
    "--association_clutter_requires_detection",
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

Write-Output "START $RunName at $(Get-Date -Format o)"
& $sggtPython @arguments
if ($LASTEXITCODE -ne 0) {
    throw "$RunName failed with exit code $LASTEXITCODE"
}
& $sggtPython evaluation\assess_association_cgic_gate.py `
    --run_dir "checkpoint\$RunName" `
    --output "results\association_cgic_200_50_gate.json"
if ($LASTEXITCODE -ne 0) {
    throw "$RunName gate assessment failed with exit code $LASTEXITCODE"
}
$gatePath = "results\association_cgic_200_50_gate.json"
if (-not (Test-Path -LiteralPath $gatePath)) {
    throw "$RunName gate assessment returned success but did not create $gatePath"
}
$gate = Get-Content -LiteralPath $gatePath -Raw | ConvertFrom-Json
if (-not $gate.approved_for_formal_training) {
    throw "$RunName completed but did not pass the formal-training gate"
}
Write-Output "FINISH $RunName at $(Get-Date -Format o)"
