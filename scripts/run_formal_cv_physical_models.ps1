$ErrorActionPreference = "Stop"
$sggtPython = "H:\Anaconda3\envs\py310_torch\python.exe"
$dataRoot = "E:\Dataset\GroupTargetsDataset_dataRate_2_v9_pack"
$shared = "checkpoint\formal_cv_physical_shared_init_seed123\shared_init.pth"
$expertRoot = "checkpoint\formal_cv_fixed_expert_full_seed123"
$expertCheckpoint = "$expertRoot\GNN_groupTargetTracking\best_position.pth"
$dynamicRoot = "checkpoint\formal_cv_dynamic_full_seed123"
$featureCache = "evaluation_results\formal_cv_dynamic_features_train1000_val200_seed123.pt"

$expertArguments = @(
    "main.py",
    "--data_dir", $dataRoot,
    "--output_dir", $expertRoot,
    "--log_dir", "log\formal_cv_fixed_expert_full_seed123.log",
    "--init_checkpoint", $shared,
    "--seed", "123",
    "--train_epochs", "12",
    "--warmup_epochs", "2",
    "--warmup_start_factor", "0.1",
    "--early_stopping_patience", "4",
    "--early_stopping_min_delta", "0.05",
    "--checkpoint_metric", "position",
    "--train_batch_size", "4",
    "--eval_batch_size", "1",
    "--full_time_supervision",
    "--encoder_execution_mode", "stream",
    "--optimizer_step_mode", "trajectory_batch",
    "--train_metric_interval", "16",
    "--loader_workers", "0",
    "--lr", "0.0006",
    "--min_lr", "0.00001",
    "--weight_decay", "0.0001",
    "--covariance_transition", "cv",
    "--decoder_use_GASF",
    "--decoder_use_SEAN",
    "--gasf_filter_mode", "abc",
    "--gasf_graph_mode", "graph",
    "--gasf_fusion_mode", "concat",
    "--interaction_physical_residual",
    "--interaction_physical_max_gain", "0.075",
    "--interaction_physical_init_gain", "0.02",
    "--interaction_activation_floor", "0.0",
    "--training_stage", "physical_gain",
    "--counterfactual_training",
    "--lambda_interaction_expert", "1.0",
    "--lambda_interaction_state_alignment", "0.0",
    "--lambda_interaction_state_improvement", "0.0"
)

& $sggtPython @expertArguments
if ($LASTEXITCODE -ne 0) {
    throw "Fixed Expert training failed with exit code $LASTEXITCODE"
}
if (-not (Test-Path -LiteralPath $expertCheckpoint)) {
    throw "Fixed Expert checkpoint was not created: $expertCheckpoint"
}

$featureArguments = @(
    "evaluation\diagnose_activation_predictability_v3.py",
    "--checkpoint", $expertCheckpoint,
    "--data_dir", $dataRoot,
    "--output", "evaluation_results\formal_cv_dynamic_predictability_train1000_val200_seed123.json",
    "--feature_output", $featureCache,
    "--train_count", "1000",
    "--val_start", "0",
    "--val_count", "200",
    "--margin", "0.05",
    "--steps", "1",
    "--lr", "0.01",
    "--seed", "123",
    "--device", "cuda"
)

& $sggtPython @featureArguments
if ($LASTEXITCODE -ne 0) {
    throw "Dynamic feature collection failed with exit code $LASTEXITCODE"
}

$pretrainArguments = @(
    "evaluation\pretrain_context_activation_v3.py",
    "--features", $featureCache,
    "--checkpoint", $expertCheckpoint,
    "--output_checkpoint", "$dynamicRoot\best_model.pth",
    "--output_report", "evaluation_results\formal_cv_dynamic_pretrain_train1000_val200_seed123.json",
    "--hidden_dim", "32",
    "--steps", "2000",
    "--lr", "0.01",
    "--weight_decay", "0.0001",
    "--seed", "123",
    "--activation_floor", "0.75",
    "--gasf_fusion_mode", "concat",
    "--device", "cuda"
)

& $sggtPython @pretrainArguments
if ($LASTEXITCODE -ne 0) {
    throw "Dynamic selector pretraining failed with exit code $LASTEXITCODE"
}

$evaluationArguments = @(
    "evaluation\compare_sean_temporal_v3.py",
    "--checkpoint", "$dynamicRoot\best_model.pth",
    "--data_dir", $dataRoot,
    "--output", "evaluation_results\formal_cv_physical_models_full_val200_seed123.json",
    "--start_trajectory", "0",
    "--max_trajectories", "200",
    "--device", "cuda",
    "--modes", "temporal", "learned", "expert"
)

& $sggtPython @evaluationArguments
if ($LASTEXITCODE -ne 0) {
    throw "Final full-validation comparison failed with exit code $LASTEXITCODE"
}
