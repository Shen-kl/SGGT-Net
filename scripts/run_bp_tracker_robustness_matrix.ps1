param(
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9_-]+$')]
    [string]$ModelName,
    [int]$MonteCarlo = 20,
    [string[]]$Trajectories = @(
        "generate_trajectory\scene01\trajectory01.pt",
        "generate_trajectory\scene02\trajectory01.pt",
        "generate_trajectory\scene03\trajectory01.pt"
    ),
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$sggtPython = "H:\Anaconda3\envs\py310_torch\python.exe"
$nominalPd = 0.85
$nominalClutter = 20
$checkpointPath = (Resolve-Path -LiteralPath $Checkpoint).Path
$checkpointSha256 = (Get-FileHash -LiteralPath $checkpointPath -Algorithm SHA256).Hash.ToLowerInvariant()
$resultRoot = "results\bp_tracker_robustness\$ModelName"
$logRoot = "log\bp_tracker_robustness\$ModelName"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

function Invoke-PythonWithLogs {
    param(
        [Parameter(Mandatory=$true)][string]$Name,
        [Parameter(Mandatory=$true)][string[]]$Arguments,
        [Parameter(Mandatory=$true)][string]$StdoutLog,
        [Parameter(Mandatory=$true)][string]$StderrLog
    )
    if ($DryRun) {
        Write-Output ((@($sggtPython) + $Arguments) -join " ")
        return
    }
    $previousPreference = $ErrorActionPreference
    try {
        # Native stderr must be persisted verbatim instead of being converted
        # into a terminating PowerShell NativeCommandError record.
        $ErrorActionPreference = "Continue"
        & $sggtPython @Arguments 1> $StdoutLog 2> $StderrLog
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($exitCode -ne 0) {
        Write-Output "$Name failed with exit code $exitCode"
        Write-Output "stdout: $StdoutLog"
        Write-Output "stderr: $StderrLog"
        if (Test-Path -LiteralPath $StderrLog) {
            Get-Content -LiteralPath $StderrLog -Tail 100 | Write-Output
        }
        throw "$Name failed with exit code $exitCode; traceback: $StderrLog"
    }
}

function Test-CompletedReport {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    $report = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    $observed = [string]$report.protocol.checkpoint_sha256
    if ($observed.ToLowerInvariant() -ne $checkpointSha256) {
        throw "Refusing stale BP report for ${ModelName}: $Path uses $observed, expected $checkpointSha256"
    }
    return $true
}

foreach ($actualPd in @(0.75, 0.85, 0.95)) {
    foreach ($actualClutter in @(10, 20, 30)) {
        $pdToken = ("{0:F2}" -f $actualPd).Replace(".", "p")
        $cellOutputs = @()
        foreach ($trajectory in $Trajectories) {
            $scene = Split-Path (Split-Path $trajectory -Parent) -Leaf
            $output = (
                "$resultRoot\" +
                "pd_${pdToken}_clutter_${actualClutter}_${scene}.json"
            )
            $cellOutputs += $output
            if ((-not $DryRun) -and (Test-CompletedReport $output)) {
                Write-Output "SKIP completed $ModelName Pd=$actualPd clutter=$actualClutter scene=$scene"
                continue
            }
            $arguments = @(
                "evaluation\run_bp_tracker_robustness_v3.py",
                "--checkpoint", $checkpointPath,
                "--trajectory", $trajectory,
                "--output", $output,
                "--actual_pd", "$actualPd",
                "--actual_clutter_mean", "$actualClutter",
                "--nominal_pd", "$nominalPd",
                "--nominal_clutter_mean", "$nominalClutter",
                "--monte_carlo", "$MonteCarlo",
                "--seed", "20260813",
                "--device", "cuda"
            )
            Write-Output (
                "START $ModelName actual Pd=$actualPd clutter=$actualClutter scene=$scene"
            )
            $cellToken = "pd_${pdToken}_clutter_${actualClutter}_${scene}"
            Invoke-PythonWithLogs `
                -Name "BP cell $cellToken" `
                -Arguments $arguments `
                -StdoutLog "$logRoot\${cellToken}.stdout.log" `
                -StderrLog "$logRoot\${cellToken}.stderr.log"
        }
        $summaryOutput = (
            "$resultRoot\" +
            "pd_${pdToken}_clutter_${actualClutter}_SUMMARY.json"
        )
        $summaryArguments = @(
            "evaluation\summarize_bp_tracker_robustness_v3.py",
            "--output", $summaryOutput
        )
        foreach ($cellOutput in $cellOutputs) {
            $summaryArguments += @("--input", $cellOutput)
        }
        $summaryToken = "pd_${pdToken}_clutter_${actualClutter}_SUMMARY"
        Invoke-PythonWithLogs `
            -Name "BP summary $summaryToken" `
            -Arguments $summaryArguments `
            -StdoutLog "$logRoot\${summaryToken}.stdout.log" `
            -StderrLog "$logRoot\${summaryToken}.stderr.log"
    }
}
