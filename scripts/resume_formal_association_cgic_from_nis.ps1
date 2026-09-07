param(
    [int[]]$Seeds = @(123),
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONUNBUFFERED = "1"
$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $workspace

$sggtPython = "H:\Anaconda3\envs\py310_torch\python.exe"
$dataRoot = "E:\Dataset\GroupTargetsDataset_dataRate_2_v9_pack"
$resultRoot = "results\formal_association_cgic"
$amendmentFile = "$resultRoot\protocol_amendment_nis_fix_20260817.json"
$evaluator = "evaluation\evaluate_innovation_calibration_v3.py"
$evaluatorSha = (
    Get-FileHash -LiteralPath $evaluator -Algorithm SHA256
).Hash.ToLowerInvariant()

if (-not (Test-Path -LiteralPath $amendmentFile)) {
    throw "Missing auditable NIS protocol amendment: $amendmentFile"
}
$amendment = Get-Content -LiteralPath $amendmentFile -Raw | ConvertFrom-Json
if ([string]$amendment.fixed_evaluator.sha256 -ne $evaluatorSha) {
    throw "NIS evaluator does not match the recorded protocol amendment"
}

function Get-Sha256 {
    param([Parameter(Mandatory=$true)][string]$Path)
    return (
        Get-FileHash -LiteralPath $Path -Algorithm SHA256
    ).Hash.ToLowerInvariant()
}

function Test-CheckpointReport {
    param(
        [Parameter(Mandatory=$true)][string]$Report,
        [Parameter(Mandatory=$true)][string]$Checkpoint
    )
    if (-not (Test-Path -LiteralPath $Report)) { return $false }
    $payload = Get-Content -LiteralPath $Report -Raw | ConvertFrom-Json
    $expected = Get-Sha256 $Checkpoint
    $observed = [string]$payload.protocol.checkpoint_sha256
    if ($observed.ToLowerInvariant() -ne $expected) {
        throw "Stale report $Report uses checkpoint $observed; expected $expected"
    }
    return $true
}

function Test-NisReport {
    param(
        [Parameter(Mandatory=$true)][string]$Report,
        [Parameter(Mandatory=$true)][string]$Checkpoint
    )
    if (-not (Test-CheckpointReport $Report $Checkpoint)) { return $false }
    $payload = Get-Content -LiteralPath $Report -Raw | ConvertFrom-Json
    $observedEvaluator = [string]$payload.protocol.evaluator_sha256
    if ($observedEvaluator.ToLowerInvariant() -ne $evaluatorSha) {
        Write-Output "REBUILD stale NIS report from another evaluator: $Report"
        return $false
    }
    return $true
}

function Test-BootstrapReport {
    param(
        [Parameter(Mandatory=$true)][string]$Report,
        [Parameter(Mandatory=$true)][string]$Left,
        [Parameter(Mandatory=$true)][string]$Right
    )
    if (-not (Test-Path -LiteralPath $Report)) { return $false }
    $payload = Get-Content -LiteralPath $Report -Raw | ConvertFrom-Json
    $expectedLeft = Get-Sha256 $Left
    $expectedRight = Get-Sha256 $Right
    $leftName = [string]$payload.protocol.left_name
    $rightName = [string]$payload.protocol.right_name
    $observedLeft = [string]$payload.protocol.reports.$leftName.sha256
    $observedRight = [string]$payload.protocol.reports.$rightName.sha256
    if (
        $observedLeft.ToLowerInvariant() -ne $expectedLeft -or
        $observedRight.ToLowerInvariant() -ne $expectedRight
    ) {
        throw "Stale bootstrap report $Report does not match source reports"
    }
    return $true
}

function Invoke-PythonStage {
    param(
        [Parameter(Mandatory=$true)][string]$Name,
        [Parameter(Mandatory=$true)][string[]]$Arguments
    )
    Write-Output "START $Name at $(Get-Date -Format o)"
    if ($DryRun) {
        Write-Output ((@($sggtPython) + $Arguments) -join " ")
        return
    }
    & $sggtPython @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
    Write-Output "FINISH $Name at $(Get-Date -Format o)"
}

$transcript = "log\formal_association_cgic_nis_resume.log"
Start-Transcript -LiteralPath $transcript -Append | Out-Null
try {
    Write-Output (
        "NIS RESUME START $(Get-Date -Format o) seeds=$($Seeds -join ',') " +
        "evaluator=$evaluatorSha"
    )
    foreach ($seed in $Seeds) {
        $seedRoot = "$resultRoot\seed_$seed"
        $temporalCheckpoint = (
            "checkpoint\formal_innovation_v1\seed_$seed\A_temporal\" +
            "GNN_groupTargetTracking\best_prediction.pth"
        )
        $fullCheckpoint = (
            "checkpoint\formal_association_cgic\seed_$seed\P4_full\" +
            "GNN_groupTargetTracking\best_association_position.pth"
        )
        foreach ($checkpoint in @($temporalCheckpoint, $fullCheckpoint)) {
            if (-not (Test-Path -LiteralPath $checkpoint)) {
                throw "Missing frozen checkpoint: $checkpoint"
            }
        }
        $calibrationModels = @(
            @{ Id = "P2_temporal"; Checkpoint = $temporalCheckpoint },
            @{ Id = "P4_full"; Checkpoint = $fullCheckpoint }
        )
        foreach ($model in $calibrationModels) {
            foreach ($split in @("val", "iid_test", "ood_test")) {
                $output = "$seedRoot\$($model.Id)_nis_${split}.json"
                if (Test-NisReport $output $model.Checkpoint) {
                    Write-Output "SKIP calibrated $seed $($model.Id) $split"
                    continue
                }
                Invoke-PythonStage "seed $seed $($model.Id) NIS $split" @(
                    $evaluator,
                    "--checkpoint", $model.Checkpoint,
                    "--data_dir", $dataRoot,
                    "--output", $output,
                    "--split", $split,
                    "--max_trajectories", "0",
                    "--mode", "expert",
                    "--device", "cuda"
                )
            }
            $associationOutput = (
                "$seedRoot\$($model.Id)_nis_val_association.json"
            )
            if (Test-NisReport $associationOutput $model.Checkpoint) {
                Write-Output "SKIP calibrated $seed $($model.Id) val association"
            } else {
                Invoke-PythonStage "seed $seed $($model.Id) NIS val association" @(
                    $evaluator,
                    "--checkpoint", $model.Checkpoint,
                    "--data_dir", $dataRoot,
                    "--output", $associationOutput,
                    "--split", "val",
                    "--max_trajectories", "0",
                    "--mode", "expert",
                    "--association_augmentation",
                    "--association_seed", "20260813",
                    "--device", "cuda"
                )
            }
        }

        foreach ($split in @("val", "iid_test", "ood_test", "val_association")) {
            $comparisons = @(
                @{ Left = "P2_temporal"; Right = "P4_full" },
                @{ Left = "P3_own_only"; Right = "P4_full" },
                @{ Left = "P5_no_association_context"; Right = "P4_full" }
            )
            foreach ($comparison in $comparisons) {
                $left = "$seedRoot\$($comparison.Left)_${split}.json"
                $right = "$seedRoot\$($comparison.Right)_${split}.json"
                $output = (
                    "$seedRoot\bootstrap_$($comparison.Right)_over_" +
                    "$($comparison.Left)_${split}.json"
                )
                if (Test-BootstrapReport $output $left $right) {
                    Write-Output "SKIP bootstrap $seed $($comparison.Right) over $($comparison.Left) $split"
                    continue
                }
                Invoke-PythonStage (
                    "seed $seed bootstrap $($comparison.Right) over " +
                    "$($comparison.Left) $split"
                ) @(
                    "evaluation\bootstrap_paired_reports_v3.py",
                    "--left_name", $comparison.Left,
                    "--left_report", $left,
                    "--right_name", $comparison.Right,
                    "--right_report", $right,
                    "--output", $output,
                    "--samples", "10000",
                    "--seed", "20260813"
                )
            }
        }
    }

    $summaryArguments = @(
        "evaluation\summarize_formal_association_cgic.py",
        "--result_root", $resultRoot,
        "--seeds"
    ) + @($Seeds | ForEach-Object { "$_" }) + @(
        "--output", "$resultRoot\SUMMARY.json"
    )
    Invoke-PythonStage "formal single-seed summary" $summaryArguments

    if ($Seeds -contains 123) {
        $postModels = @(
            @{
                Name = "seed123_P2_temporal";
                Checkpoint = (
                    "checkpoint\formal_innovation_v1\seed_123\A_temporal\" +
                    "GNN_groupTargetTracking\best_prediction.pth"
                )
            },
            @{
                Name = "seed123_P4_full";
                Checkpoint = (
                    "checkpoint\formal_association_cgic\seed_123\P4_full\" +
                    "GNN_groupTargetTracking\best_association_position.pth"
                )
            }
        )
        foreach ($model in $postModels) {
            $scalingOutput = "$resultRoot\scaling_$($model.Name).json"
            if (Test-CheckpointReport $scalingOutput $model.Checkpoint) {
                Write-Output "SKIP scaling benchmark $($model.Name)"
            } else {
                Invoke-PythonStage "scaling benchmark $($model.Name)" @(
                    "evaluation\benchmark_model_scaling_v3.py",
                    "--checkpoint", $model.Checkpoint,
                    "--data_dir", $dataRoot,
                    "--output", $scalingOutput,
                    "--device", "cuda",
                    "--target_counts", "5", "10", "20", "40",
                    "--warmup", "10",
                    "--repetitions", "100"
                )
            }
        }
        foreach ($model in $postModels) {
            Write-Output "START full BP robustness $($model.Name) at $(Get-Date -Format o)"
            if ($DryRun) {
                Write-Output (
                    "powershell -File scripts/run_bp_tracker_robustness_matrix.ps1 " +
                    "-Checkpoint $($model.Checkpoint) -ModelName $($model.Name) -MonteCarlo 20"
                )
                continue
            }
            & powershell -NoProfile -ExecutionPolicy Bypass -File `
                scripts\run_bp_tracker_robustness_matrix.ps1 `
                -Checkpoint $model.Checkpoint `
                -ModelName $model.Name `
                -MonteCarlo 20
            if ($LASTEXITCODE -ne 0) {
                throw "full BP robustness failed for $($model.Name)"
            }
            Write-Output "FINISH full BP robustness $($model.Name) at $(Get-Date -Format o)"
        }
    }
    Write-Output "NIS RESUME FINISH $(Get-Date -Format o)"
}
catch {
    Write-Error "NIS RESUME FAILED $(Get-Date -Format o): $($_.Exception.Message)"
    throw
}
finally {
    Stop-Transcript | Out-Null
}
