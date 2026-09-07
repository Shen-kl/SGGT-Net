param(
    [string]$Python = "H:\Anaconda3\envs\py310_torch\python.exe",
    [string]$McstRepository = "E:\PycharmProjects\Neural Network\MCST_2D_update",
    [string]$OutputDirectory = "E:\PycharmProjects\Neural Network\SGGT-Net\checkpoint\mcst_v9_full_seed123"
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$logRoot = "E:\PycharmProjects\Neural Network\SGGT-Net\log"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$stdout = Join-Path $logRoot "mcst_v9_full_seed123.stdout.log"
$stderr = Join-Path $logRoot "mcst_v9_full_seed123.stderr.log"
$entrypoint = Join-Path $McstRepository "main_v9.py"
$arguments = @(
    $entrypoint,
    "--data_dir", "E:\Dataset\GroupTargetsDataset_dataRate_2_v9_pack",
    "--output_dir", $OutputDirectory,
    "--seed", "123",
    "--batch_size", "16",
    "--num_workers", "2",
    "--max_epochs", "100",
    "--tbptt_steps", "16",
    "--early_stop_patience", "10",
    "--min_delta_m", "0.05",
    "--no-amp"
)
$last = Join-Path $OutputDirectory "last.pth"
if (Test-Path -LiteralPath $last) {
    $arguments += @("--resume", $last)
}

& $Python @arguments 1>> $stdout 2>> $stderr
exit $LASTEXITCODE
