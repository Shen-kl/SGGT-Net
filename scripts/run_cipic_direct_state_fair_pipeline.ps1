param(
    [int]$TrainSamples = 200,
    [int]$ValSamples = 50,
    [int]$Epochs = 10,
    [int]$Seed = 123,
    [string]$RunName = "dev_cipic_direct_state_own_peer_seed123"
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot
$resultRoot = Join-Path $repoRoot "results\cipic_output_parameterization_seed123"
New-Item -ItemType Directory -Force -Path $resultRoot | Out-Null
$transcript = Join-Path $resultRoot "pipeline.log"
$sggtPython = "H:\Anaconda3\envs\py310_torch\python.exe"

Start-Transcript -Path $transcript -Append | Out-Null
try {
    & (Join-Path $PSScriptRoot "run_cipic_direct_state_fair_dev.ps1") `
        -TrainSamples $TrainSamples `
        -ValSamples $ValSamples `
        -Epochs $Epochs `
        -Seed $Seed `
        -RunName $RunName
    if ($LASTEXITCODE -ne 0) {
        throw "Direct-state training failed with exit code $LASTEXITCODE"
    }

    & $sggtPython evaluation\summarize_cipic_output_parameterization.py `
        --acceleration_summary results\cipic_acceleration_dev_seed123\summary.json `
        --direct_run "checkpoint\$RunName" `
        --output (Join-Path $resultRoot "summary.json")
    if ($LASTEXITCODE -ne 0) {
        throw "Direct-state comparison failed with exit code $LASTEXITCODE"
    }
}
finally {
    Stop-Transcript | Out-Null
}
