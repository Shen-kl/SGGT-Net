param(
    [ValidateSet("Smoke", "Formal", "All")]
    [string]$Stage = "All",
    [int]$Seed = 123,
    [int]$SmokeEpochs = 6,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot
$sggtPython = "H:\Anaconda3\envs\py310_torch\python.exe"
$dataRoot = "E:\Dataset\GroupTargetsDataset_dataRate_2_v9_pack"
$experimentRoot = "checkpoint\formal_mtp_go_v1\seed_$Seed"
$resultRoot = "results\formal_mtp_go_v1\seed_$Seed"
$logRoot = "log\formal_mtp_go_v1\seed_$Seed"

New-Item -ItemType Directory -Force -Path $resultRoot, $logRoot | Out-Null
$pipelineTranscript = "$logRoot\pipeline_transcript.log"
Start-Transcript -Path $pipelineTranscript -Append | Out-Null
trap {
    Write-Error $_
    Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
    exit 1
}

$common = @(
    "main_v3.py",
    "--data_dir", $dataRoot,
    "--device", "cuda",
    "--seed", "$Seed",
    "--model_variant", "sggt_v2",
    "--train_batch_size", "4",
    "--eval_batch_size", "4",
    "--loader_workers", "0",
    "--pin_memory",
    "--skip_final_tests",
    "--lr", "0.0006",
    "--min_lr", "0.00001",
    "--weight_decay", "0.0001",
    "--warmup_start_factor", "0.1",
    "--early_stopping_min_delta", "0.05",
    "--checkpoint_metric", "prediction",
    "--full_time_supervision",
    "--encoder_execution_mode", "stream",
    "--encoder_stream_reset_interval", "0",
    "--optimizer_step_mode", "trajectory_batch",
    "--train_metric_interval", "16",
    "--grad_clip_norm", "5",
    "--covariance_transition", "cv",
    "--no-decoder_use_GASF",
    "--no-decoder_use_SEAN",
    "--no-innovation_decomposition",
    "--no-causal_innovation_correction",
    "--no-association_augmentation",
    "--no-association_validation",
    "--training_stage", "joint",
    "--lambda_struct", "0",
    "--lambda_temporal", "0",
    "--lambda_nll", "0.0001",
    "--adaptive_q_mode", "legacy"
)

function Invoke-Training([string]$Name, [string[]]$ExtraArguments) {
    Write-Output "START $Name at $(Get-Date -Format o)"
    $arguments = $common + $ExtraArguments
    if ($DryRun) {
        Write-Output ((@($sggtPython) + $arguments) -join " ")
        return
    }
    $env:KMP_DUPLICATE_LIB_OK = "TRUE"
    & $sggtPython @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
    Write-Output "FINISH $Name at $(Get-Date -Format o)"
}

$smokeDir = "$experimentRoot\P1_mtp_go_smoke_200_50"
$smokeLog = "$logRoot\P1_mtp_go_smoke_200_50.log"
$smokeReport = "$resultRoot\P1_mtp_go_smoke_200_50_gate.json"
$formalDir = "$experimentRoot\P1_mtp_go_formal"
$formalLog = "$logRoot\P1_mtp_go_formal.log"
$formalReport = "$resultRoot\P1_mtp_go_formal_gate.json"

if ($Stage -in @("Smoke", "All")) {
    Invoke-Training "P1 smoke 200/50" @(
        "--output_dir", $smokeDir,
        "--log_dir", $smokeLog,
        "--debug_train_samples", "200",
        "--debug_train_start", "0",
        "--debug_val_samples", "50",
        "--debug_val_start", "0",
        "--train_epochs", "$SmokeEpochs",
        "--warmup_epochs", "1",
        "--early_stopping_patience", "$SmokeEpochs",
        "--finite_check_interval", "64"
    )
    if (-not $DryRun) {
        & $sggtPython scripts\validate_p1_mtp_go_run.py `
            --run-dir $smokeDir `
            --expected-validation-trajectories 50 `
            --minimum-prediction-improvement-m 0.05 `
            --report $smokeReport
        if ($LASTEXITCODE -ne 0) {
            throw "P1 smoke gate failed; formal training was not started"
        }
    }
}

if ($Stage -in @("Formal", "All")) {
    if ($Stage -eq "All" -and -not $DryRun -and -not (Test-Path -LiteralPath $smokeReport)) {
        throw "Smoke gate report is missing: $smokeReport"
    }
    if ($Stage -eq "All" -and -not $DryRun) {
        $gate = Get-Content -LiteralPath $smokeReport -Raw | ConvertFrom-Json
        if ($gate.status -ne "pass") {
            throw "Smoke gate status is not pass; formal training was not started"
        }
    }
    Invoke-Training "P1 formal full-data" @(
        "--output_dir", $formalDir,
        "--log_dir", $formalLog,
        "--debug_train_samples", "0",
        "--debug_val_samples", "0",
        "--train_epochs", "60",
        "--warmup_epochs", "5",
        "--early_stopping_patience", "8",
        "--finite_check_interval", "0"
    )
    if (-not $DryRun) {
        & $sggtPython scripts\validate_p1_mtp_go_run.py `
            --run-dir $formalDir `
            --expected-validation-trajectories 200 `
            --minimum-prediction-improvement-m 0.05 `
            --report $formalReport
        if ($LASTEXITCODE -ne 0) {
            throw "P1 formal run completed but failed its integrity gate"
        }
    }
}

Stop-Transcript | Out-Null
