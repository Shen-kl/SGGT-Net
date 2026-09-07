param(
    [int]$Seed = 123
)

$ErrorActionPreference = "Stop"
$sggtPython = "H:\Anaconda3\envs\py310_torch\python.exe"
$dataRoot = "E:\Dataset\GroupTargetsDataset_dataRate_2_v9_pack"
$expertCheckpoint = (
    "checkpoint\causal_innovation_formal_seed123\" +
    "GNN_groupTargetTracking\best_position.pth"
)
$outputRoot = "checkpoint\causal_innovation_gate_formal_seed$Seed"
$logPath = "log\causal_innovation_gate_formal_seed$Seed.log"

$arguments = @(
    "main_v3.py",
    "--data_dir", $dataRoot,
    "--output_dir", $outputRoot,
    "--log_dir", $logPath,
    "--init_checkpoint", $expertCheckpoint,
    "--seed", "$Seed",
    "--device", "cuda",
    "--train_batch_size", "4",
    "--eval_batch_size", "4",
    "--loader_workers", "0",
    "--skip_final_tests",
    "--train_epochs", "20",
    "--warmup_epochs", "1",
    "--warmup_start_factor", "0.1",
    "--early_stopping_patience", "4",
    "--early_stopping_min_delta", "0.05",
    "--checkpoint_metric", "position",
    "--lr", "0.001",
    "--min_lr", "0.00003",
    "--weight_decay", "0.0001",
    "--full_time_supervision",
    "--encoder_execution_mode", "stream",
    "--optimizer_step_mode", "trajectory_batch",
    "--train_metric_interval", "32",
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
    "--causal_innovation_correction",
    "--causal_innovation_dynamic_activation",
    "--causal_innovation_history", "8",
    "--causal_innovation_hidden_size", "32",
    "--causal_innovation_max_position_m", "50",
    "--causal_innovation_max_velocity_mps", "5",
    "--training_stage", "causal_innovation_gate",
    "--lambda_struct", "0",
    "--lambda_temporal", "0",
    "--lambda_nll", "0.0001",
    "--adaptive_q_mode", "legacy",
    "--lambda_interaction_regret", "0",
    "--interaction_gate_position_margin_m", "0.05",
    "--interaction_gate_steady_weight", "1",
    "--interaction_gate_switch_weight", "1.5",
    "--interaction_gate_strong_weight", "2"
)

Write-Output "START formal causal innovation gate training at $(Get-Date -Format o)"
& $sggtPython @arguments
if ($LASTEXITCODE -ne 0) {
    throw "formal causal innovation gate training failed with exit code $LASTEXITCODE"
}
Write-Output "FINISH formal causal innovation gate training at $(Get-Date -Format o)"
