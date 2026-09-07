param(
    [ValidateSet("R0", "R1", "Both")]
    [string]$Arm = "Both",
    [int]$TrainSamples = 200,
    [int]$ValSamples = 50,
    [int]$Epochs = 12,
    [int]$Seed = 123,
    [string]$Tag = "adapt_200_50",
    [double]$LearningRate = 0.0001,
    [switch]$FilterOnly,
    [string]$InitialCheckpoint = (
        "checkpoint\formal_innovation_v1\seed_123\A_temporal\" +
        "GNN_groupTargetTracking\best_prediction.pth"
    ),
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot
$sggtPython = "H:\Anaconda3\envs\py310_torch\python.exe"
$dataRoot = "E:\Dataset\GroupTargetsDataset_dataRate_2_v9_pack"
$experimentRoot = "checkpoint\dev_gasf_recurrent_abc\seed_$Seed\$Tag"
$resultRoot = "results\dev_gasf_recurrent_abc\seed_$Seed\$Tag"
$logRoot = "log\dev_gasf_recurrent_abc\seed_$Seed\$Tag"

if (-not (Test-Path -LiteralPath $InitialCheckpoint)) {
    throw "Missing shared Temporal checkpoint: $InitialCheckpoint"
}
New-Item -ItemType Directory -Force -Path $resultRoot, $logRoot | Out-Null

$common = @(
    "main_v3.py",
    "--data_dir", $dataRoot,
    "--device", "cuda",
    "--seed", "$Seed",
    "--model_variant", "sggt_v2",
    "--init_checkpoint", $InitialCheckpoint,
    "--debug_train_samples", "$TrainSamples",
    "--debug_val_samples", "$ValSamples",
    "--train_batch_size", "4",
    "--eval_batch_size", "4",
    "--loader_workers", "0",
    "--pin_memory",
    "--skip_final_tests",
    "--train_epochs", "$Epochs",
    "--warmup_epochs", "1",
    "--warmup_start_factor", "0.1",
    "--early_stopping_patience", "5",
    "--early_stopping_min_delta", "0.05",
    "--checkpoint_metric", "prediction",
    "--lr", "$LearningRate",
    "--min_lr", "0.000003",
    "--weight_decay", "0.0001",
    "--rollout_steps", "4",
    # The shared source checkpoint is already a closed-loop model. Restarting
    # the scratch-training teacher-forcing curriculum would confound this
    # single-factor adaptation experiment.
    "--teacher_forcing_start", "0",
    "--teacher_forcing_min", "0",
    "--teacher_forcing_decay_epochs", "0",
    "--full_time_supervision",
    "--encoder_execution_mode", "stream",
    "--encoder_stream_reset_interval", "0",
    "--optimizer_step_mode", "trajectory_batch",
    "--train_metric_interval", "32",
    "--finite_check_interval", "32",
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
    "--no-association_augmentation",
    "--no-association_validation",
    "--lambda_struct", "0",
    "--lambda_temporal", "0",
    "--lambda_nll", "0.0001",
    "--adaptive_q_mode", "legacy"
)

function Invoke-Arm([string]$Name, [bool]$UseRecurrence) {
    $runDir = "$experimentRoot\$Name"
    $logFile = "$logRoot\$Name.log"
    $best = "$runDir\GNN_groupTargetTracking\best_prediction.pth"
    $report = "$resultRoot\${Name}_val.json"
    $recurrenceArgument = if ($UseRecurrence) {
        "--gasf_recurrent_filter"
    } else {
        "--no-gasf_recurrent_filter"
    }
    $trainingStage = if ($FilterOnly) { "filter_recurrence" } else { "joint" }
    $arguments = $common + @(
        "--output_dir", $runDir,
        "--log_dir", $logFile,
        "--training_stage", $trainingStage,
        $recurrenceArgument,
        "--gasf_recurrent_gate_initial_update", "0.1"
    )
    Write-Output "START $Name at $(Get-Date -Format o)"
    if ($DryRun) {
        if ($FilterOnly -and -not $UseRecurrence) {
            Write-Output "EVALUATE frozen R0 checkpoint: $InitialCheckpoint"
        } else {
            Write-Output ((@($sggtPython) + $arguments) -join " ")
        }
        return
    }
    if ($FilterOnly -and -not $UseRecurrence) {
        # The frozen feed-forward checkpoint is the exact counterfactual for
        # recurrence-only adaptation; retraining it adds no information.
        $best = $InitialCheckpoint
    } else {
        & $sggtPython @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "$Name training failed with exit code $LASTEXITCODE"
        }
        if (-not (Test-Path -LiteralPath $best)) {
            throw "$Name did not create $best"
        }
    }
    & $sggtPython evaluation\evaluate_checkpoint_events_v3.py `
        --checkpoint $best `
        --data_dir $dataRoot `
        --output $report `
        --split val `
        --max_trajectories $ValSamples `
        --batch_size 4 `
        --device cuda
    if ($LASTEXITCODE -ne 0) {
        throw "$Name evaluation failed with exit code $LASTEXITCODE"
    }
    Write-Output "FINISH $Name at $(Get-Date -Format o)"
}

if ($Arm -in @("R0", "Both")) {
    Invoke-Arm "R0_feedforward_abc" $false
}
if ($Arm -in @("R1", "Both")) {
    Invoke-Arm "R1_previous_abc" $true
}

if (-not $DryRun -and $Arm -eq "Both") {
    & $sggtPython evaluation\compare_gasf_recurrent_abc_results.py `
        --baseline "$resultRoot\R0_feedforward_abc_val.json" `
        --recurrent "$resultRoot\R1_previous_abc_val.json" `
        --output "$resultRoot\comparison.json"
    if ($LASTEXITCODE -ne 0) {
        throw "Result comparison failed with exit code $LASTEXITCODE"
    }
}
