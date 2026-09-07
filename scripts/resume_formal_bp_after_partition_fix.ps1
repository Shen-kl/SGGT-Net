param(
    [int]$MonteCarlo = 20,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONUNBUFFERED = "1"
$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $workspace

$amendmentFile = (
    "results\formal_association_cgic\" +
    "protocol_amendment_bp_partition_fix_20260818.json"
)
if (-not (Test-Path -LiteralPath $amendmentFile)) {
    throw "Missing auditable BP protocol amendment: $amendmentFile"
}
$amendment = Get-Content -LiteralPath $amendmentFile -Raw | ConvertFrom-Json
foreach ($path in @(
    "utils/common/radar.py",
    "utils/common/measurement_evaluation.py",
    "scripts/run_bp_tracker_robustness_matrix.ps1"
)) {
    $windowsPath = $path.Replace("/", "\")
    $actualSha = (
        Get-FileHash -LiteralPath $windowsPath -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    $expectedSha = [string]$amendment.fixed_files.$path.new_sha256
    if ($actualSha -ne $expectedSha.ToLowerInvariant()) {
        throw "BP repair hash mismatch for ${path}: $actualSha != $expectedSha"
    }
}

$models = @(
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

$transcript = "log\formal_association_cgic_bp_resume_20260818.log"
Start-Transcript -LiteralPath $transcript -Append | Out-Null
try {
    Write-Output "BP RESUME START $(Get-Date -Format o) MC=$MonteCarlo"
    foreach ($model in $models) {
        if (-not (Test-Path -LiteralPath $model.Checkpoint)) {
            throw "Missing frozen checkpoint: $($model.Checkpoint)"
        }
        Write-Output "START BP matrix $($model.Name) at $(Get-Date -Format o)"
        $arguments = @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", "scripts\run_bp_tracker_robustness_matrix.ps1",
            "-Checkpoint", $model.Checkpoint,
            "-ModelName", $model.Name,
            "-MonteCarlo", "$MonteCarlo"
        )
        if ($DryRun) {
            $arguments += "-DryRun"
        }
        & powershell @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "BP matrix failed for $($model.Name)"
        }
        Write-Output "FINISH BP matrix $($model.Name) at $(Get-Date -Format o)"
    }
    Write-Output "BP RESUME FINISH $(Get-Date -Format o)"
}
catch {
    Write-Error "BP RESUME FAILED $(Get-Date -Format o): $($_.Exception.Message)"
    throw
}
finally {
    Stop-Transcript | Out-Null
}
