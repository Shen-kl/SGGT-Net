param(
    [int]$TrainSamples = 200,
    [int]$ValSamples = 50,
    [int]$Epochs = 10,
    [int]$Seed = 123,
    [double]$MaxAccelerationMps2 = 5.0
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))
$ownName = "dev_cipic_acceleration_own_seed$Seed"
$peerName = "dev_cipic_acceleration_own_peer_seed$Seed"
$ownRoot = "checkpoint\$ownName"
$peerRoot = "checkpoint\$peerName"
$summary = "results\cipic_acceleration_dev_seed$Seed\summary.json"
$sggtPython = "H:\Anaconda3\envs\py310_torch\python.exe"
$pipelineLog = "log\cipic_acceleration_dev_seed$Seed.pipeline.log"

Start-Transcript -Path $pipelineLog -Force
try {
    & "$PSScriptRoot\run_cipic_acceleration_dev.ps1" `
        -TrainSamples $TrainSamples -ValSamples $ValSamples -Epochs $Epochs `
        -Seed $Seed -MaxAccelerationMps2 $MaxAccelerationMps2 `
        -RunName $ownName -OwnOnly
    if ($LASTEXITCODE -ne 0) {
        throw "own-only CIPIC training failed with exit code $LASTEXITCODE"
    }

    & "$PSScriptRoot\run_cipic_acceleration_dev.ps1" `
        -TrainSamples $TrainSamples -ValSamples $ValSamples -Epochs $Epochs `
        -Seed $Seed -MaxAccelerationMps2 $MaxAccelerationMps2 `
        -RunName $peerName
    if ($LASTEXITCODE -ne 0) {
        throw "own+peer CIPIC training failed with exit code $LASTEXITCODE"
    }

    & $sggtPython evaluation\summarize_cipic_acceleration_dev.py `
        --own_only $ownRoot --own_peer $peerRoot --output $summary `
        --expected_val_trajectories $ValSamples
    if ($LASTEXITCODE -ne 0) {
        throw "CIPIC comparison summary failed with exit code $LASTEXITCODE"
    }
} finally {
    Stop-Transcript
}
