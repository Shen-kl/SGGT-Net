param(
    [string]$Python = "H:\Anaconda3\envs\py310_torch\python.exe",
    [string]$McstRepository = "E:\PycharmProjects\Neural Network\MCST_2D_update",
    [string]$TrainingDirectory = "E:\PycharmProjects\Neural Network\SGGT-Net\checkpoint\mcst_v9_full_seed123",
    [string]$ResultDirectory = "E:\PycharmProjects\Neural Network\SGGT-Net\results\mcst_v9_seed123"
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$diagnosticLog = "E:\PycharmProjects\Neural Network\SGGT-Net\log\mcst_v9_post_monitor.log"
trap {
    $message = "{0} ERROR: {1}`n{2}" -f (Get-Date).ToString("o"), $_.Exception.Message, $_.ScriptStackTrace
    Add-Content -LiteralPath $diagnosticLog -Value $message -Encoding UTF8
    exit 1
}
Add-Content -LiteralPath $diagnosticLog -Value ("{0} STARTED PID={1}" -f (Get-Date).ToString("o"), $PID) -Encoding UTF8
$summary = Join-Path $TrainingDirectory "summary.json"
$deadline = (Get-Date).AddHours(72)
while (-not (Test-Path -LiteralPath $summary)) {
    if ((Get-Date) -gt $deadline) {
        throw "Timed out waiting for MCST full training: $summary"
    }
    Start-Sleep -Seconds 300
}

$checkpoint = Join-Path $TrainingDirectory "best_prior_position.pth"
$history = Join-Path $TrainingDirectory "history.jsonl"
$candidate = Join-Path $TrainingDirectory "best_exact_candidate.pth"
$candidateRecord = Join-Path $TrainingDirectory "best_exact_candidate.json"
$bestReadyDeadline = (Get-Date).AddMinutes(15)
$bestReady = $false
while (-not $bestReady) {
    if ((Get-Date) -gt $bestReadyDeadline) {
        throw "Timed out waiting for the exact numerical-best checkpoint"
    }
    if (
        (Test-Path -LiteralPath $history) -and
        (Test-Path -LiteralPath $candidate) -and
        (Test-Path -LiteralPath $candidateRecord)
    ) {
        $validationRows = Get-Content -LiteralPath $history | ForEach-Object {
            try { $_ | ConvertFrom-Json } catch { $null }
        } | Where-Object { $_ -and $_.split -eq "validation" -and $_.epoch -ge 0 }
        if ($validationRows) {
            $minimum = $validationRows | Sort-Object prior_position_rmse | Select-Object -First 1
            $record = Get-Content -LiteralPath $candidateRecord -Raw | ConvertFrom-Json
            $metricDifference = [math]::Abs(
                [double]$minimum.prior_position_rmse - [double]$record.prior_position_rmse
            )
            if ([int]$minimum.epoch -eq [int]$record.epoch -and $metricDifference -lt 1e-9) {
                Copy-Item -LiteralPath $candidate -Destination $checkpoint -Force
                $bestReady = $true
                break
            }
        }
    }
    Start-Sleep -Seconds 15
}

New-Item -ItemType Directory -Force -Path $ResultDirectory | Out-Null
$iidOod = Join-Path $ResultDirectory "iid_ood.json"
& $Python (Join-Path $McstRepository "evaluate_v9.py") `
    --checkpoint $checkpoint `
    --data_dir "E:\Dataset\GroupTargetsDataset_dataRate_2_v9_pack" `
    --splits iid_test ood_test `
    --batch_size 16 --num_workers 2 --device cuda `
    --output $iidOod
if ($LASTEXITCODE -ne 0) { throw "MCST IID/OOD evaluation failed" }

$bpReport = Join-Path $ResultDirectory "simulation_scenario_mc20.json"
$trajectory = "E:\PycharmProjects\Neural Network\GNN_groupTargetTracking_v9_norm_without_struct\generate_trajectory\scene02\trajectory01.pt"
& $Python "E:\PycharmProjects\Neural Network\SGGT-Net\evaluation\run_bp_tracker_robustness_v3.py" `
    --model_backend mcst `
    --mcst_repository $McstRepository `
    --checkpoint $checkpoint `
    --trajectory $trajectory `
    --output $bpReport `
    --actual_pd 0.99 --actual_clutter_mean 2 `
    --nominal_pd 0.99 --nominal_clutter_mean 2 `
    --range_noise_std_m 3 --azimuth_noise_std_deg 0.01 `
    --monte_carlo 20 --seed 20260824 --device cuda `
    --ospa_cutoff_m 100 --ospa_order 1 `
    --neural_process_noise_scale 6.874306 `
    --neural_position_q_variance_m2 1.316616
if ($LASTEXITCODE -ne 0) { throw "MCST-BP simulation scenario evaluation failed" }

& $Python "E:\PycharmProjects\Neural Network\SGGT-Net\evaluation\summarize_mcst_bp_v3.py" `
    --input $bpReport --output_dir $ResultDirectory
if ($LASTEXITCODE -ne 0) { throw "MCST-BP aggregation failed" }

Set-Content -LiteralPath (Join-Path $ResultDirectory "DONE") -Value (Get-Date).ToString("o")
Add-Content -LiteralPath $diagnosticLog -Value ("{0} COMPLETED PID={1}" -f (Get-Date).ToString("o"), $PID) -Encoding UTF8
