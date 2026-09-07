param(
    [int[]]$Seeds = @(123, 456, 789),
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$sggtPython = "H:\Anaconda3\envs\py310_torch\python.exe"
$dataRoot = "E:\Dataset\GroupTargetsDataset_dataRate_2_v9_pack"
$protocol = "results\formal_innovation_protocol_v1.json"

if (-not (Test-Path -LiteralPath $protocol)) {
    throw "Frozen protocol does not exist: $protocol"
}
& $sggtPython evaluation\freeze_innovation_protocol_v3.py `
    --data_dir $dataRoot --output $protocol --verify
if ($LASTEXITCODE -ne 0) {
    throw "Formal protocol verification failed"
}

$common = @(
    "main.py",
    "--data_dir", $dataRoot,
    "--train_batch_size", "4",
    "--eval_batch_size", "4",
    "--loader_workers", "0",
    "--full_time_supervision",
    "--encoder_execution_mode", "stream",
    "--optimizer_step_mode", "trajectory_batch",
    "--train_metric_interval", "16",
    "--finite_check_interval", "0",
    "--grad_clip_norm", "5",
    "--fusion_adapt_epochs", "0",
    "--weight_decay", "0.0001",
    "--covariance_transition", "cv",
    "--decoder_use_GASF",
    "--gasf_filter_mode", "abc",
    "--gasf_fusion_mode", "concat",
    "--gasf_aux_source", "fusion",
    "--lambda_struct", "0",
    "--lambda_temporal", "0",
    "--lambda_nll", "0.0001",
    "--adaptive_q_mode", "legacy"
)

function Invoke-Stage {
    param(
        [string]$Name,
        [string[]]$Arguments,
        [string]$ExpectedCheckpoint
    )
    if (Test-Path -LiteralPath $ExpectedCheckpoint) {
        Write-Output "SKIP $Name; checkpoint exists: $ExpectedCheckpoint"
        return
    }
    Write-Output "START $Name at $(Get-Date -Format o)"
    if ($DryRun) {
        Write-Output (($sggtPython, $Arguments) -join " ")
        return
    }
    & $sggtPython @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
    if (-not (Test-Path -LiteralPath $ExpectedCheckpoint)) {
        throw "$Name did not create $ExpectedCheckpoint"
    }
    Write-Output "FINISH $Name at $(Get-Date -Format o)"
}

foreach ($seed in $Seeds) {
    $seedRoot = "checkpoint\formal_innovation_v1\seed_$seed"
    $temporalRoot = "$seedRoot\A_temporal"
    $fixedGraphRoot = "$seedRoot\B_no_sean_fixed_graph"
    $fullRoot = "$seedRoot\C_full_innovation"
    $temporalBest = "$temporalRoot\GNN_groupTargetTracking\best_prediction.pth"
    $fixedGraphBest = "$fixedGraphRoot\GNN_groupTargetTracking\best_prediction.pth"
    $fullBest = "$fullRoot\GNN_groupTargetTracking\best_expert.pth"

    $temporalArguments = $common + @(
        "--output_dir", $temporalRoot,
        "--log_dir", "log\formal_innovation_v1_seed${seed}_A_temporal.log",
        "--seed", "$seed",
        "--train_epochs", "60",
        "--warmup_epochs", "5",
        "--warmup_start_factor", "0.1",
        "--early_stopping_patience", "8",
        "--early_stopping_min_delta", "0.05",
        "--checkpoint_metric", "prediction",
        "--lr", "0.0006",
        "--min_lr", "0.00001",
        "--no-decoder_use_SEAN",
        "--gasf_graph_mode", "temporal_only",
        "--no-innovation_decomposition",
        "--training_stage", "joint",
        "--no-counterfactual_training"
    )
    Invoke-Stage "seed $seed A Temporal" $temporalArguments $temporalBest

    $fixedGraphArguments = $common + @(
        "--output_dir", $fixedGraphRoot,
        "--log_dir", "log\formal_innovation_v1_seed${seed}_B_no_sean.log",
        "--init_checkpoint", $temporalBest,
        "--seed", "$seed",
        "--train_epochs", "30",
        "--warmup_epochs", "2",
        "--warmup_start_factor", "0.1",
        "--early_stopping_patience", "6",
        "--early_stopping_min_delta", "0.05",
        "--checkpoint_metric", "prediction",
        "--lr", "0.0001",
        "--min_lr", "0.00001",
        "--no-decoder_use_SEAN",
        "--gasf_graph_mode", "graph",
        "--no-innovation_decomposition",
        "--training_stage", "expert",
        "--no-counterfactual_training"
    )
    Invoke-Stage "seed $seed B no-SEAN fixed graph" `
        $fixedGraphArguments $fixedGraphBest

    $fullArguments = $common + @(
        "--output_dir", $fullRoot,
        "--log_dir", "log\formal_innovation_v1_seed${seed}_C_full.log",
        "--init_checkpoint", $temporalBest,
        "--seed", "$seed",
        "--train_epochs", "30",
        "--warmup_epochs", "2",
        "--warmup_start_factor", "0.1",
        "--early_stopping_patience", "6",
        "--early_stopping_min_delta", "0.0001",
        "--checkpoint_metric", "expert",
        "--lr", "0.0001",
        "--min_lr", "0.00001",
        "--decoder_use_SEAN",
        "--gasf_graph_mode", "graph",
        "--innovation_decomposition",
        "--innovation_whiten_clip", "20",
        "--training_stage", "innovation_expert",
        "--counterfactual_training",
        "--lambda_interaction_expert", "1.0",
        "--lambda_interaction_base", "0",
        "--lambda_interaction_regret", "0",
        "--lambda_interaction_gate", "0",
        "--lambda_interaction_aux_expert", "0",
        "--lambda_interaction_aux_rank", "0",
        "--lambda_interaction_demand", "0",
        "--lambda_interaction_state_alignment", "0.1",
        "--lambda_interaction_state_improvement", "0.1"
    )
    Invoke-Stage "seed $seed C full innovation" $fullArguments $fullBest

    if (-not $DryRun) {
        foreach ($evaluation in @(
            @{ Name = "A_temporal"; Checkpoint = $temporalBest },
            @{ Name = "B_no_sean_fixed_graph"; Checkpoint = $fixedGraphBest },
            @{ Name = "C_full_innovation"; Checkpoint = $fullBest }
        )) {
            & $sggtPython evaluation\evaluate_checkpoint_events_v3.py `
                --checkpoint $evaluation.Checkpoint --data_dir $dataRoot `
                --output "results\formal_innovation_v1_seed${seed}_$($evaluation.Name)_val200.json" `
                --split val --start_trajectory 0 --max_trajectories 200 `
                --device cuda
            if ($LASTEXITCODE -ne 0) {
                throw "seed $seed $($evaluation.Name) validation failed"
            }
        }
        & $sggtPython evaluation\compare_sean_temporal_v3.py `
            --checkpoint $fullBest --data_dir $dataRoot `
            --output "results\formal_innovation_v1_seed${seed}_C_val200.json" `
            --start_trajectory 0 --max_trajectories 200 --device cuda `
            --modes temporal expert
        if ($LASTEXITCODE -ne 0) {
            throw "seed $seed full validation comparison failed"
        }
        & $sggtPython evaluation\evaluate_innovation_calibration_v3.py `
            --checkpoint $fullBest --data_dir $dataRoot `
            --output "results\formal_innovation_v1_seed${seed}_C_nis_val200.json" `
            --split val --max_trajectories 0 --device cuda --mode expert
        if ($LASTEXITCODE -ne 0) {
            throw "seed $seed innovation calibration failed"
        }
    }
}

Write-Output "ALL FORMAL INNOVATION TRAINING STAGES FINISHED at $(Get-Date -Format o)"
