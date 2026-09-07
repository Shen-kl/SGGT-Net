param(
    [int[]]$Seeds = @(123, 456, 789),
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONUNBUFFERED = "1"
$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $workspace
$logDirectory = Join-Path $workspace "log"
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$transcript = Join-Path $logDirectory "formal_association_cgic_pipeline.log"

Start-Transcript -LiteralPath $transcript -Append | Out-Null
try {
    Write-Output "PIPELINE START $(Get-Date -Format o) seeds=$($Seeds -join ',')"

    & "$PSScriptRoot\run_missing_temporal_baselines.ps1" `
        -Seeds $Seeds -DryRun:$DryRun

    $missingTemporal = @($Seeds | Where-Object {
        -not (Test-Path -LiteralPath (
            "checkpoint\formal_innovation_v1\seed_$_\A_temporal\" +
            "GNN_groupTargetTracking\best_prediction.pth"
        ))
    })
    if ($DryRun -and $missingTemporal.Count -gt 0) {
        Write-Output (
            "DRY RUN: defer formal stages until Temporal checkpoint(s) exist: " +
            ($missingTemporal -join ",")
        )
    } else {
        & "$PSScriptRoot\run_formal_association_cgic_multiseed.ps1" `
            -Seeds $Seeds -DryRun:$DryRun
    }

    Write-Output "PIPELINE FINISH $(Get-Date -Format o)"
}
catch {
    Write-Error "PIPELINE FAILED $(Get-Date -Format o): $($_.Exception.Message)"
    throw
}
finally {
    Stop-Transcript | Out-Null
}
