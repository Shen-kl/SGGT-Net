param(
    [int]$TuneTrajectories = 200,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$sggtPython = "H:\Anaconda3\envs\py310_torch\python.exe"
$dataRoot = "E:\Dataset\GroupTargetsDataset_dataRate_2_v9_pack"
$tuning = "results\formal_association_cgic\physical_baseline_tuning.json"

$tuneArguments = @(
    "evaluation\tune_physical_baselines_v3.py",
    "--data_dir", $dataRoot,
    "--output", $tuning,
    "--max_trajectories", "$TuneTrajectories"
)
if (-not (Test-Path -LiteralPath $tuning)) {
    if ($DryRun) {
        Write-Output ((@($sggtPython) + $tuneArguments) -join " ")
    } else {
        & $sggtPython @tuneArguments
        if ($LASTEXITCODE -ne 0) { throw "physical baseline tuning failed" }
    }
} else {
    Write-Output "SKIP tuning; frozen validation selection exists: $tuning"
}

if ($DryRun -and -not (Test-Path -LiteralPath $tuning)) {
    Write-Output "Evaluation commands require the selected tuning artifact."
    exit 0
}

$selected = Get-Content -LiteralPath $tuning -Raw | ConvertFrom-Json
foreach ($model in @("cv", "imm")) {
    $modelId = if ($model -eq "cv") { "P0_cv" } else { "P1_imm" }
    foreach ($split in @("val", "iid_test", "ood_test")) {
        $arguments = @(
            "evaluation\evaluate_physical_baseline_v3.py",
            "--data_dir", $dataRoot,
            "--output", "results\formal_association_cgic\${modelId}_${split}.json",
            "--split", $split,
            "--model", $model
        )
        $output = "results\formal_association_cgic\${modelId}_${split}.json"
        if (Test-Path -LiteralPath $output) {
            Write-Output "SKIP $model $split; report exists: $output"
            continue
        }
        if ($model -eq "cv") {
            $arguments += @(
                "--cv_acceleration_std",
                "$($selected.selected.cv.parameters.cv_acceleration_std)"
            )
        } else {
            $transition = $selected.selected.imm.parameters.imm_transition
            $arguments += @(
                "--imm_low_acceleration_std",
                "$($selected.selected.imm.parameters.imm_low_acceleration_std)",
                "--imm_high_acceleration_std",
                "$($selected.selected.imm.parameters.imm_high_acceleration_std)",
                "--imm_transition",
                "$($transition[0])", "$($transition[1])",
                "$($transition[2])", "$($transition[3])"
            )
        }
        if ($DryRun) {
            Write-Output ((@($sggtPython) + $arguments) -join " ")
            continue
        }
        & $sggtPython @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "$model $split physical baseline evaluation failed"
        }
    }
}
