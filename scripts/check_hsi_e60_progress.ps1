param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [switch]$HashCheckpoint
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)
$QueueName = "hsi_e60_validation_2026-09-15"
$RunName = "ablation_s1024_b4_hsi16_shared_p005_995_e60_r2"
$TargetEpochs = 60
$QueueRoot = Join-Path $ProjectRoot ("artifacts\background_queue\" + $QueueName)
$StatePath = Join-Path $QueueRoot "state.json"
$RunDir = Join-Path $ProjectRoot ("runs\" + $RunName)
$ResultsPath = Join-Path $RunDir "results.csv"
$LastCheckpoint = Join-Path $RunDir "weights\last.pt"

if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot ".git"))) {
    throw "Refusing to inspect outside the expected Git project: $ProjectRoot"
}

$state = $null
if (Test-Path -LiteralPath $StatePath -PathType Leaf) {
    $state = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json
}

$resultRows = @()
if (Test-Path -LiteralPath $ResultsPath -PathType Leaf) {
    $resultRows = @(Import-Csv -LiteralPath $ResultsPath)
}
$completedEpoch = if ($resultRows.Count -gt 0) {
    [int][double]$resultRows[-1].epoch
}
else {
    0
}
$best = $resultRows |
    Where-Object { $_.'metrics/mAP50-95(B)' -ne "" } |
    Sort-Object { [double]$_.'metrics/mAP50-95(B)' } -Descending |
    Select-Object -First 1

$runnerPid = if ($null -ne $state -and $null -ne $state.runner_pid) {
    [int]$state.runner_pid
}
else {
    0
}
$runnerProcess = if ($runnerPid -gt 0) {
    Get-Process -Id $runnerPid -ErrorAction SilentlyContinue
}
else {
    $null
}
$runnerIdentityMatches = $false
if ($null -ne $runnerProcess -and $null -ne $state.job.started_at -and $state.job.started_at -ne "") {
    $expectedStart = [DateTimeOffset]::Parse([string]$state.job.started_at)
    $actualStart = [DateTimeOffset]$runnerProcess.StartTime
    $runnerIdentityMatches =
        $runnerProcess.ProcessName -like "powershell*" -and
        [Math]::Abs(($actualStart - $expectedStart).TotalSeconds) -lt 30
}

$checkpointInfo = Get-Item -LiteralPath $LastCheckpoint -ErrorAction SilentlyContinue
$checkpointHash = $null
if ($HashCheckpoint -and $null -ne $checkpointInfo) {
    $before = Get-Item -LiteralPath $LastCheckpoint
    $hash = (Get-FileHash -LiteralPath $LastCheckpoint -Algorithm SHA256).Hash
    $after = Get-Item -LiteralPath $LastCheckpoint
    if ($before.Length -ne $after.Length -or $before.LastWriteTimeUtc -ne $after.LastWriteTimeUtc) {
        throw "last.pt changed while hashing; wait for the current epoch to finish and retry"
    }
    $checkpointHash = $hash
}

[ordered]@{
    checked_at = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssK")
    queue_name = $QueueName
    queue_status = if ($null -ne $state) { [string]$state.queue_status } else { "state_missing" }
    recorded_runner_pid = $runnerPid
    runner_process_exists = $null -ne $runnerProcess
    runner_identity_matches = $runnerIdentityMatches
    run_name = $RunName
    completed_epoch = $completedEpoch
    target_epochs = $TargetEpochs
    best_epoch = if ($null -ne $best) { [int][double]$best.epoch } else { $null }
    best_map50_95 = if ($null -ne $best) { [double]$best.'metrics/mAP50-95(B)' } else { $null }
    last_checkpoint = $LastCheckpoint
    last_checkpoint_exists = $null -ne $checkpointInfo
    last_checkpoint_bytes = if ($null -ne $checkpointInfo) { $checkpointInfo.Length } else { $null }
    last_checkpoint_written_at = if ($null -ne $checkpointInfo) { $checkpointInfo.LastWriteTime.ToString("yyyy-MM-ddTHH:mm:ssK") } else { $null }
    last_checkpoint_sha256 = $checkpointHash
} | ConvertTo-Json -Depth 4
