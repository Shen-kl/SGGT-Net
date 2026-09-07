param(
    [int]$TrainSamples = 8,
    [int]$ValSamples = 4,
    [int]$Epochs = 2,
    [int]$Seed = 123,
    [string]$RunName = "dev_causal_q_8_4_seed123",
    [double]$LearningRate = 0.0003,
    [double]$MinLearningRate = 0.00003,
    [int]$TrainStart = 0,
    [int]$ValStart = 0,
    [double]$QSurpriseWeight = 0.05,
    [double]$QSmoothWeight = 0.01
)

$ErrorActionPreference = "Stop"
$sggtPython = "H:\Anaconda3\envs\py310_torch\python.exe"
$dataRoot = "E:\Dataset\GroupTargetsDataset_dataRate_2_v9_pack"
$p4Checkpoint = (
    "checkpoint\formal_association_cgic\seed_123\P4_full\" +
    "GNN_groupTargetTracking\best_association_position.pth"
)

$arguments = @(
    "main_v3.py",
    "--data_dir", $dataRoot,
    "--output_dir", "checkpoint\$RunName",
    "--log_dir", "log\$RunName.log",
    "--init_checkpoint", $p4Checkpoint,
    "--seed", "$Seed",
    "--device", "cuda",
    "--train_batch_size", "4",
    "--eval_batch_size", "4",
    "--loader_workers", "0",
    "--debug_train_samples", "$TrainSamples",
    "--debug_train_start", "$TrainStart",
    "--debug_val_samples", "$ValSamples",
    "--debug_val_start", "$ValStart",
    "--skip_final_tests",
    "--train_epochs", "$Epochs",
    "--warmup_epochs", "0",
    "--early_stopping_patience", "0",
    "--checkpoint_metric", "position",
    "--lr", "$LearningRate",
    "--min_lr", "$MinLearningRate",
    "--weight_decay", "0.0001",
    "--full_time_supervision",
    "--encoder_execution_mode", "stream",
    "--optimizer_step_mode", "trajectory_batch",
    "--train_metric_interval", "64",
    "--finite_check_interval", "32",
    "--grad_clip_norm", "5",
    "--position_loss_weight", "32",
    "--velocity_loss_weight", "1",
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
    "--training_stage", "q_adapt",
    "--lambda_struct", "0",
    "--lambda_temporal", "0",
    "--lambda_aux", "0",
    "--lambda_nll", "0.001",
    "--adaptive_q_mode", "causal_nis",
    "--adaptive_q_max_multiplier", "16",
    "--adaptive_q_max_position_variance", "16",
    "--adaptive_q_initial_multiplier", "8",
    "--adaptive_q_initial_position_variance", "1",
    "--lambda_q_rank", "0",
    "--lambda_q_surprise", "$QSurpriseWeight",
    "--q_surprise_margin", "0.1",
    "--q_surprise_min_gap", "0.1",
    "--lambda_q_smooth", "$QSmoothWeight",
    "--lambda_q_baseline", "0",
    "--lambda_interaction_expert", "0",
    "--lambda_interaction_base", "0",
    "--lambda_interaction_regret", "0",
    "--lambda_interaction_gate", "0",
    "--lambda_interaction_aux_expert", "0",
    "--lambda_interaction_aux_rank", "0",
    "--lambda_interaction_demand", "0",
    "--lambda_interaction_state_alignment", "0",
    "--lambda_interaction_state_improvement", "0",
    "--lambda_causal_velocity_guard", "0"
)

Write-Output "START $RunName at $(Get-Date -Format o)"
& $sggtPython @arguments
if ($LASTEXITCODE -ne 0) {
    throw "$RunName failed with exit code $LASTEXITCODE"
}
Write-Output "FINISH $RunName at $(Get-Date -Format o)"
