param(
    [string]$TrainingDirectory = "E:\PycharmProjects\Neural Network\SGGT-Net\checkpoint\mcst_v9_full_seed123"
)

$ErrorActionPreference = "Stop"
$history = Join-Path $TrainingDirectory "history.jsonl"
$last = Join-Path $TrainingDirectory "last.pth"
$candidate = Join-Path $TrainingDirectory "best_exact_candidate.pth"
$record = Join-Path $TrainingDirectory "best_exact_candidate.json"
$summary = Join-Path $TrainingDirectory "summary.json"
$ready = Join-Path $TrainingDirectory "EXACT_BEST_READY"
$bestMetric = [double]::PositiveInfinity
$bestEpoch = -1

while ($true) {
    if (Test-Path -LiteralPath $history) {
        $validationRows = Get-Content -LiteralPath $history | ForEach-Object {
            try { $_ | ConvertFrom-Json } catch { $null }
        } | Where-Object { $_ -and $_.split -eq "validation" -and $_.epoch -ge 0 }
        if ($validationRows) {
            $minimum = $validationRows | Sort-Object prior_position_rmse | Select-Object -First 1
            $latest = $validationRows | Sort-Object epoch -Descending | Select-Object -First 1
            if (
                [int]$minimum.epoch -eq [int]$latest.epoch -and
                [double]$minimum.prior_position_rmse -lt $bestMetric
            ) {
                Start-Sleep -Seconds 5
                if (Test-Path -LiteralPath $last) {
                    Copy-Item -LiteralPath $last -Destination $candidate -Force
                    $bestMetric = [double]$minimum.prior_position_rmse
                    $bestEpoch = [int]$minimum.epoch
                    @{
                        epoch = $bestEpoch
                        prior_position_rmse = $bestMetric
                        captured_at = (Get-Date).ToString("o")
                    } | ConvertTo-Json | Set-Content -LiteralPath $record -Encoding UTF8
                }
            }
        }
    }
    if (Test-Path -LiteralPath $summary) {
        if (Test-Path -LiteralPath $candidate) {
            Copy-Item -LiteralPath $candidate `
                -Destination (Join-Path $TrainingDirectory "best_prior_position.pth") -Force
            Set-Content -LiteralPath $ready -Value (Get-Date).ToString("o")
        } else {
            throw "Training finished without an exact-best checkpoint candidate"
        }
        break
    }
    Start-Sleep -Seconds 60
}
