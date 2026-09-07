param(
    [int[]]$Seeds = @(123, 456, 789),
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$sggtPython = "H:\Anaconda3\envs\py310_torch\python.exe"
$dataRoot = "E:\Dataset\GroupTargetsDataset_dataRate_2_v9_pack"
$gateFile = "results\association_cgic_200_50_gate.json"
$protocolFile = "results\formal_association_cgic_protocol_v1.json"

if (-not (Test-Path -LiteralPath $gateFile)) {
    throw "Formal training is gated: missing $gateFile"
}
$gate = Get-Content -LiteralPath $gateFile -Raw | ConvertFrom-Json
if (-not $gate.approved_for_formal_training) {
    throw "Formal training is gated: 200/50 run was not approved"
}
$common = @(
    "main_v3.py",
    "--data_dir", $dataRoot,
    "--device", "cuda",
    "--train_batch_size", "4",
    "--eval_batch_size", "4",
    "--loader_workers", "0",
    "--skip_final_tests",
    "--train_epochs", "30",
    "--warmup_epochs", "1",
    "--warmup_start_factor", "0.1",
    "--early_stopping_patience", "6",
    "--early_stopping_min_delta", "0.05",
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

function Get-Sha256 {
    param([string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Test-CheckpointReport {
    param([string]$Report, [string]$Checkpoint)
    if (-not (Test-Path -LiteralPath $Report)) { return $false }
    $payload = Get-Content -LiteralPath $Report -Raw | ConvertFrom-Json
    $expected = Get-Sha256 $Checkpoint
    $observed = [string]$payload.protocol.checkpoint_sha256
    if ($observed.ToLowerInvariant() -ne $expected) {
        throw "Stale report $Report uses checkpoint $observed; expected $expected"
    }
    return $true
}

function Test-BootstrapReport {
    param([string]$Report, [string]$Left, [string]$Right)
    if (-not (Test-Path -LiteralPath $Report)) { return $false }
    $payload = Get-Content -LiteralPath $Report -Raw | ConvertFrom-Json
    $expectedLeft = Get-Sha256 $Left
    $expectedRight = Get-Sha256 $Right
    $observedLeft = [string]$payload.protocol.reports.$($payload.protocol.left_name).sha256
    $observedRight = [string]$payload.protocol.reports.$($payload.protocol.right_name).sha256
    if (
        $observedLeft.ToLowerInvariant() -ne $expectedLeft -or
        $observedRight.ToLowerInvariant() -ne $expectedRight
    ) {
        throw "Stale bootstrap report $Report does not match its source reports"
    }
    return $true
}

# Fail before launching any expensive run if a seed's frozen Temporal baseline
# is unavailable.  Partial multi-seed matrices are easy to misinterpret.
foreach ($seed in $Seeds) {
    $preflightTemporal = (
        "checkpoint\formal_innovation_v1\seed_$seed\A_temporal\" +
        "GNN_groupTargetTracking\best_prediction.pth"
    )
    if (-not (Test-Path -LiteralPath $preflightTemporal)) {
        throw "Missing frozen Temporal checkpoint for seed ${seed}: $preflightTemporal"
    }
}

$temporalVerification = (
    "results\formal_association_cgic\temporal_checkpoint_verification.json"
)
& $sggtPython evaluation\verify_temporal_checkpoints_v3.py `
    --checkpoint_root checkpoint\formal_innovation_v1 `
    --seeds $Seeds --output $temporalVerification
if ($LASTEXITCODE -ne 0) {
    throw "Frozen Temporal checkpoints are not protocol-compatible"
}

# Freeze only after every required initialization checkpoint is present. This
# avoids hashing the complete dataset and then failing before the first run.
if (-not (Test-Path -LiteralPath $protocolFile)) {
    & $sggtPython evaluation\freeze_association_cgic_protocol_v3.py `
        --data_dir $dataRoot --output $protocolFile
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to freeze formal association-CGIC protocol"
    }
}
& $sggtPython evaluation\freeze_association_cgic_protocol_v3.py `
    --data_dir $dataRoot --output $protocolFile --verify
if ($LASTEXITCODE -ne 0) {
    throw "Formal association-CGIC protocol verification failed"
}

$physicalArtifacts = @(
    "results\formal_association_cgic\physical_baseline_tuning.json",
    "results\formal_association_cgic\P0_cv_val.json",
    "results\formal_association_cgic\P0_cv_iid_test.json",
    "results\formal_association_cgic\P0_cv_ood_test.json",
    "results\formal_association_cgic\P1_imm_val.json",
    "results\formal_association_cgic\P1_imm_iid_test.json",
    "results\formal_association_cgic\P1_imm_ood_test.json"
)
$missingPhysicalArtifact = $physicalArtifacts | Where-Object {
    -not (Test-Path -LiteralPath $_)
}
if ($missingPhysicalArtifact) {
    Write-Output "START/resume deterministic P0/P1 physical baselines"
    $physicalArguments = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
        "scripts\run_physical_baselines_v3.ps1"
    )
    if ($DryRun) { $physicalArguments += "-DryRun" }
    & powershell @physicalArguments
    if (-not $DryRun -and $LASTEXITCODE -ne 0) {
        throw "P0/P1 physical baseline protocol failed"
    }
}

foreach ($seed in $Seeds) {
    $temporalCheckpoint = (
        "checkpoint\formal_innovation_v1\seed_$seed\A_temporal\" +
        "GNN_groupTargetTracking\best_prediction.pth"
    )
    $variants = @(
        @{
            Id = "P3_own_only";
            Flags = @(
                "--causal_innovation_association_context",
                "--no-causal_innovation_use_peer"
            )
        },
        @{
            Id = "P4_full";
            Flags = @(
                "--causal_innovation_association_context",
                "--causal_innovation_use_peer"
            )
        },
        @{
            Id = "P5_no_association_context";
            Flags = @(
                "--no-causal_innovation_association_context",
                "--causal_innovation_use_peer"
            )
        }
    )

    $variantCheckpoints = @{}
    foreach ($variant in $variants) {
        $root = "checkpoint\formal_association_cgic\seed_$seed\$($variant.Id)"
        $best = "$root\GNN_groupTargetTracking\best_association_position.pth"
        $arguments = $common + @(
            "--output_dir", $root,
            "--log_dir", "log\formal_association_cgic_seed${seed}_$($variant.Id).log",
            "--init_checkpoint", $temporalCheckpoint,
            "--seed", "$seed"
        ) + $variant.Flags
        Invoke-Stage "seed $seed $($variant.Id)" $arguments $best
        $variantCheckpoints[$variant.Id] = $best
    }

    if (-not $DryRun) {
        $evaluationModels = @(
            @{ Id = "P2_temporal"; Checkpoint = $temporalCheckpoint },
            @{ Id = "P3_own_only"; Checkpoint = $variantCheckpoints["P3_own_only"] },
            @{ Id = "P4_full"; Checkpoint = $variantCheckpoints["P4_full"] },
            @{ Id = "P5_no_association_context"; Checkpoint = $variantCheckpoints["P5_no_association_context"] }
        )
        foreach ($evaluationModel in $evaluationModels) {
            foreach ($split in @("val", "iid_test", "ood_test")) {
                $output = (
                    "results\formal_association_cgic\seed_$seed\" +
                    "$($evaluationModel.Id)_${split}.json"
                )
                if (Test-CheckpointReport $output $evaluationModel.Checkpoint) {
                    Write-Output "SKIP evaluated $seed $($evaluationModel.Id) $split"
                    continue
                }
                & $sggtPython evaluation\evaluate_checkpoint_events_v3.py `
                    --checkpoint $evaluationModel.Checkpoint `
                    --data_dir $dataRoot --output $output --split $split `
                    --max_trajectories 0 --batch_size 4 --trajectory_sse `
                    --device cuda
                if ($LASTEXITCODE -ne 0) {
                    throw "seed $seed $($evaluationModel.Id) $split evaluation failed"
                }
            }
            $associationOutput = (
                "results\formal_association_cgic\seed_$seed\" +
                "$($evaluationModel.Id)_val_association.json"
            )
            if (Test-CheckpointReport $associationOutput $evaluationModel.Checkpoint) {
                Write-Output "SKIP evaluated $seed $($evaluationModel.Id) val association"
                continue
            }
            & $sggtPython evaluation\evaluate_checkpoint_events_v3.py `
                --checkpoint $evaluationModel.Checkpoint `
                --data_dir $dataRoot --output $associationOutput --split val `
                --max_trajectories 0 --batch_size 4 --trajectory_sse `
                --association_augmentation --association_seed 20260813 `
                --device cuda
            if ($LASTEXITCODE -ne 0) {
                throw "seed $seed $($evaluationModel.Id) association evaluation failed"
            }
        }

        foreach ($calibrationModel in $evaluationModels | Where-Object {
            $_.Id -in @("P2_temporal", "P4_full")
        }) {
            foreach ($split in @("val", "iid_test", "ood_test")) {
                $calibrationOutput = (
                    "results\formal_association_cgic\seed_$seed\" +
                    "$($calibrationModel.Id)_nis_${split}.json"
                )
                if (Test-CheckpointReport $calibrationOutput $calibrationModel.Checkpoint) {
                    Write-Output "SKIP calibrated $seed $($calibrationModel.Id) $split"
                    continue
                }
                & $sggtPython evaluation\evaluate_innovation_calibration_v3.py `
                    --checkpoint $calibrationModel.Checkpoint `
                    --data_dir $dataRoot --output $calibrationOutput `
                    --split $split --max_trajectories 0 --mode expert `
                    --device cuda
                if ($LASTEXITCODE -ne 0) {
                    throw "seed $seed $($calibrationModel.Id) NIS $split failed"
                }
            }
            $associationCalibrationOutput = (
                "results\formal_association_cgic\seed_$seed\" +
                "$($calibrationModel.Id)_nis_val_association.json"
            )
            if (Test-CheckpointReport $associationCalibrationOutput $calibrationModel.Checkpoint) {
                Write-Output "SKIP calibrated $seed $($calibrationModel.Id) val association"
                continue
            }
            & $sggtPython evaluation\evaluate_innovation_calibration_v3.py `
                --checkpoint $calibrationModel.Checkpoint `
                --data_dir $dataRoot --output $associationCalibrationOutput `
                --split val --max_trajectories 0 --mode expert `
                --association_augmentation --association_seed 20260813 `
                --device cuda
            if ($LASTEXITCODE -ne 0) {
                throw "seed $seed $($calibrationModel.Id) association NIS failed"
            }
        }

        foreach ($split in @("val", "iid_test", "ood_test", "val_association")) {
            $comparisons = @(
                @{ Left = "P2_temporal"; Right = "P4_full" },
                @{ Left = "P3_own_only"; Right = "P4_full" },
                @{ Left = "P5_no_association_context"; Right = "P4_full" }
            )
            foreach ($comparison in $comparisons) {
                $resultRoot = "results\formal_association_cgic\seed_$seed"
                $leftReport = "$resultRoot\$($comparison.Left)_${split}.json"
                $rightReport = "$resultRoot\$($comparison.Right)_${split}.json"
                $bootstrapOutput = "$resultRoot\bootstrap_$($comparison.Right)_over_$($comparison.Left)_${split}.json"
                if (Test-BootstrapReport $bootstrapOutput $leftReport $rightReport) {
                    Write-Output "SKIP bootstrap $seed $($comparison.Right) over $($comparison.Left) $split"
                    continue
                }
                & $sggtPython evaluation\bootstrap_paired_reports_v3.py `
                    --left_name $comparison.Left `
                    --left_report $leftReport `
                    --right_name $comparison.Right `
                    --right_report $rightReport `
                    --output $bootstrapOutput `
                    --samples 10000 --seed 20260813
                if ($LASTEXITCODE -ne 0) {
                    throw "seed $seed bootstrap $($comparison.Right) over $($comparison.Left) $split failed"
                }
            }
        }
    }
}

if (-not $DryRun) {
    & $sggtPython evaluation\summarize_formal_association_cgic.py `
        --result_root results\formal_association_cgic `
        --seeds $Seeds `
        --output results\formal_association_cgic\SUMMARY.json
    if ($LASTEXITCODE -ne 0) {
        throw "formal association-CGIC multi-seed summary failed"
    }

    # Full BP is much more expensive than known-identity predictor rollout.
    # Run it for the preregistered representative seed only, but compare the
    # exact same Temporal initialization and the full CGIC checkpoint under
    # every actual Pd/clutter cell. Model-specific roots prevent overwrites.
    $representativeSeed = 123
    if ($Seeds -contains $representativeSeed) {
        $scalingModels = @(
            @{
                Name = "seed123_P2_temporal"
                Checkpoint = (
                    "checkpoint\formal_innovation_v1\seed_123\A_temporal\" +
                    "GNN_groupTargetTracking\best_prediction.pth"
                )
            },
            @{
                Name = "seed123_P4_full"
                Checkpoint = (
                    "checkpoint\formal_association_cgic\seed_123\P4_full\" +
                    "GNN_groupTargetTracking\best_association_position.pth"
                )
            }
        )
        foreach ($scalingModel in $scalingModels) {
            $scalingOutput = (
                "results\formal_association_cgic\scaling_" +
                "$($scalingModel.Name).json"
            )
            if (Test-CheckpointReport $scalingOutput $scalingModel.Checkpoint) {
                Write-Output "SKIP scaling benchmark $($scalingModel.Name)"
            } else {
                & $sggtPython evaluation\benchmark_model_scaling_v3.py `
                    --checkpoint $scalingModel.Checkpoint `
                    --data_dir $dataRoot --output $scalingOutput `
                    --device cuda --target_counts 5 10 20 40 `
                    --warmup 10 --repetitions 100
                if ($LASTEXITCODE -ne 0) {
                    throw "scaling benchmark failed for $($scalingModel.Name)"
                }
            }
        }

        $bpModels = @(
            @{
                Name = "seed123_P2_temporal"
                Checkpoint = (
                    "checkpoint\formal_innovation_v1\seed_123\A_temporal\" +
                    "GNN_groupTargetTracking\best_prediction.pth"
                )
            },
            @{
                Name = "seed123_P4_full"
                Checkpoint = (
                    "checkpoint\formal_association_cgic\seed_123\P4_full\" +
                    "GNN_groupTargetTracking\best_association_position.pth"
                )
            }
        )
        foreach ($bpModel in $bpModels) {
            & powershell -NoProfile -ExecutionPolicy Bypass -File `
                scripts\run_bp_tracker_robustness_matrix.ps1 `
                -Checkpoint $bpModel.Checkpoint -ModelName $bpModel.Name `
                -MonteCarlo 20
            if ($LASTEXITCODE -ne 0) {
                throw "full BP robustness failed for $($bpModel.Name)"
            }
        }
    }
}

Write-Output "ALL FORMAL ASSOCIATION-CGIC TRAINING FINISHED at $(Get-Date -Format o)"
