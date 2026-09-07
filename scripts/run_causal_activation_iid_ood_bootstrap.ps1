param(
    [int]$BootstrapSamples = 10000,
    [int]$BootstrapSeed = 20260813
)

$ErrorActionPreference = "Stop"
$sggtPython = "H:\Anaconda3\envs\py310_torch\python.exe"
$dataRoot = "E:\Dataset\GroupTargetsDataset_dataRate_2_v9_pack"
$expertCheckpoint = (
    "checkpoint\causal_innovation_formal_seed123\" +
    "GNN_groupTargetTracking\best_position.pth"
)
$dynamicCheckpoint = (
    "checkpoint\causal_innovation_gate_formal_seed123\" +
    "GNN_groupTargetTracking\best_position.pth"
)
$resultRoot = "evaluation_results\causal_activation_formal_seed123"
$bootstrapRoot = "results\causal_activation_formal_seed123"
$logRoot = "log\causal_activation_formal_seed123"

New-Item -ItemType Directory -Force -Path $resultRoot | Out-Null
New-Item -ItemType Directory -Force -Path $bootstrapRoot | Out-Null
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

function Test-CompleteReport {
    param([string]$Path, [int]$ExpectedTrajectories)
    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }
    try {
        $report = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
        return (
            [int]$report.protocol.trajectory_count -eq $ExpectedTrajectories -and
            @($report.trajectories).Count -eq $ExpectedTrajectories
        )
    }
    catch {
        return $false
    }
}

function Invoke-FormalEvaluation {
    param(
        [string]$Split,
        [int]$TrajectoryCount,
        [string]$Name,
        [string]$Checkpoint,
        [Nullable[double]]$Alpha
    )
    $output = Join-Path $resultRoot "${Split}_${Name}.json"
    $log = Join-Path $logRoot "${Split}_${Name}.log"
    if (Test-CompleteReport -Path $output -ExpectedTrajectories $TrajectoryCount) {
        Write-Output "SKIP complete $Split/$Name at $(Get-Date -Format o)"
        return
    }
    $arguments = @(
        "evaluation\evaluate_checkpoint_events_v3.py",
        "--checkpoint", $Checkpoint,
        "--data_dir", $dataRoot,
        "--output", $output,
        "--split", $Split,
        "--batch_size", "4",
        "--trajectory_sse",
        "--device", "cuda"
    )
    if ($null -ne $Alpha) {
        $arguments += @("--causal_alpha", "$Alpha")
    }
    Write-Output "START $Split/$Name at $(Get-Date -Format o)"
    # Windows PowerShell promotes tqdm's harmless stderr progress stream to a
    # NativeCommandError under Stop.  Judge native tools by their exit code.
    $savedErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $sggtPython @arguments *> $log
    $evaluationExitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedErrorActionPreference
    if ($evaluationExitCode -ne 0) {
        throw "$Split/$Name failed with exit code $evaluationExitCode; see $log"
    }
    if (-not (Test-CompleteReport -Path $output -ExpectedTrajectories $TrajectoryCount)) {
        throw "$Split/$Name produced an incomplete report"
    }
    Write-Output "FINISH $Split/$Name at $(Get-Date -Format o)"
}

function Invoke-Bootstrap {
    param([string]$Split)
    $output = Join-Path $bootstrapRoot "${Split}_bootstrap.json"
    $reports = @(
        "--report", "temporal", (Join-Path $resultRoot "${Split}_temporal.json"),
        "--report", "fixed05", (Join-Path $resultRoot "${Split}_fixed05.json"),
        "--report", "full", (Join-Path $resultRoot "${Split}_full.json"),
        "--report", "dynamic", (Join-Path $resultRoot "${Split}_dynamic.json")
    )
    Write-Output "START $Split trajectory bootstrap at $(Get-Date -Format o)"
    $savedErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $sggtPython evaluation\bootstrap_causal_activation_v3.py `
        @reports --output $output --samples $BootstrapSamples `
        --seed $BootstrapSeed *> (Join-Path $logRoot "${Split}_bootstrap.log")
    $bootstrapExitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedErrorActionPreference
    if ($bootstrapExitCode -ne 0) {
        throw "$Split bootstrap failed with exit code $bootstrapExitCode"
    }
    Write-Output "FINISH $Split trajectory bootstrap at $(Get-Date -Format o)"
}

$splits = @(
    @{ Name = "iid_test"; Count = 300 },
    @{ Name = "ood_test"; Count = 400 }
)
foreach ($split in $splits) {
    Invoke-FormalEvaluation $split.Name $split.Count "temporal" $expertCheckpoint 0.0
    Invoke-FormalEvaluation $split.Name $split.Count "fixed05" $expertCheckpoint 0.5
    Invoke-FormalEvaluation $split.Name $split.Count "full" $expertCheckpoint 1.0
    Invoke-FormalEvaluation $split.Name $split.Count "dynamic" $dynamicCheckpoint $null
    Invoke-Bootstrap $split.Name
}

Write-Output "ALL IID/OOD evaluations and paired bootstraps finished at $(Get-Date -Format o)"
