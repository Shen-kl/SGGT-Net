param(
    [int]$TrainSamples = 200,
    [int]$ValSamples = 50,
    [int]$Epochs = 10,
    [int]$Seed = 123
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot
$resultRoot = Join-Path $repoRoot "results\cipic_state_channel_ablation_seed123"
New-Item -ItemType Directory -Force -Path $resultRoot | Out-Null
$transcript = Join-Path $resultRoot "pipeline.log"
$sggtPython = "H:\Anaconda3\envs\py310_torch\python.exe"
$positionRun = "dev_cipic_direct_state_position_only_seed123"
$velocityRun = "dev_cipic_direct_state_velocity_only_seed123"

Start-Transcript -Path $transcript -Append | Out-Null
try {
    & (Join-Path $PSScriptRoot "run_cipic_direct_state_fair_dev.ps1") `
        -TrainSamples $TrainSamples -ValSamples $ValSamples `
        -Epochs $Epochs -Seed $Seed `
        -StateChannels "position_only" -RunName $positionRun
    if ($LASTEXITCODE -ne 0) {
        throw "Position-only training failed with exit code $LASTEXITCODE"
    }

    & (Join-Path $PSScriptRoot "run_cipic_direct_state_fair_dev.ps1") `
        -TrainSamples $TrainSamples -ValSamples $ValSamples `
        -Epochs $Epochs -Seed $Seed `
        -StateChannels "velocity_only" -RunName $velocityRun
    if ($LASTEXITCODE -ne 0) {
        throw "Velocity-only training failed with exit code $LASTEXITCODE"
    }

    & $sggtPython evaluation\summarize_cipic_state_channel_ablation.py `
        --full_summary results\cipic_output_parameterization_seed123\summary.json `
        --position_run "checkpoint\$positionRun" `
        --velocity_run "checkpoint\$velocityRun" `
        --output (Join-Path $resultRoot "summary.json")
    if ($LASTEXITCODE -ne 0) {
        throw "Channel ablation summary failed with exit code $LASTEXITCODE"
    }
}
finally {
    Stop-Transcript | Out-Null
}
