param(
    [double[]]$Alphas = @(0.1, 0.25, 0.5, 0.75, 1.0),
    [int]$ValSamples = 16,
    [int]$Seed = 123
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot
$sggtPython = "H:\Anaconda3\envs\py310_torch\python.exe"
$dataRoot = "E:\Dataset\GroupTargetsDataset_dataRate_2_v9_pack"
$checkpoint = (
    "checkpoint\formal_innovation_v1\seed_$Seed\A_temporal\" +
    "GNN_groupTargetTracking\best_prediction.pth"
)
$resultRoot = "results\dev_gasf_recurrent_abc\seed_$Seed\alpha_sweep_val$ValSamples"
New-Item -ItemType Directory -Force -Path $resultRoot | Out-Null

foreach ($alpha in $Alphas) {
    $label = $alpha.ToString("0.00", [Globalization.CultureInfo]::InvariantCulture).Replace(".", "p")
    $output = "$resultRoot\alpha_$label.json"
    & $sggtPython evaluation\evaluate_checkpoint_events_v3.py `
        --checkpoint $checkpoint `
        --data_dir $dataRoot `
        --output $output `
        --split val `
        --max_trajectories $ValSamples `
        --batch_size 4 `
        --device cuda `
        --gasf_recurrent_alpha $alpha
    if ($LASTEXITCODE -ne 0) {
        throw "alpha=$alpha evaluation failed with exit code $LASTEXITCODE"
    }
}

& $sggtPython evaluation\summarize_gasf_recurrent_alpha_sweep.py `
    --input_dir $resultRoot `
    --output "$resultRoot\summary.json"
if ($LASTEXITCODE -ne 0) {
    throw "alpha sweep summary failed with exit code $LASTEXITCODE"
}
