param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)
$QueueName = "hsi_e60_validation_2026-09-15"
$DependencyQueueName = "score_search_2026-09-13"
$RunName = "ablation_s1024_b4_hsi16_shared_p005_995_e60_r2"
$TargetEpochs = 60
$QueueRoot = Join-Path $ProjectRoot ("artifacts\background_queue\" + $QueueName)
$StatePath = Join-Path $QueueRoot "state.json"
$CompleteSentinel = Join-Path $QueueRoot "queue.complete"
$PausedSentinel = Join-Path $QueueRoot "queue.paused"
$RunDir = Join-Path $ProjectRoot ("runs\" + $RunName)
$ResultsPath = Join-Path $RunDir "results.csv"
$LastCheckpoint = Join-Path $RunDir "weights\last.pt"
$RunnerScript = Join-Path $ProjectRoot "scripts\run_multispectral_followup.ps1"
$PowerShell = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"

foreach ($requiredPath in @(
    (Join-Path $ProjectRoot ".git"),
    $RunnerScript,
    (Join-Path $ProjectRoot ".venv\Scripts\python.exe"),
    (Join-Path $ProjectRoot "scripts\train_baseline.py"),
    (Join-Path $ProjectRoot "data\processed\hsi16_shared_p005_995\dataset.yaml"),
    $LastCheckpoint,
    $PowerShell
)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required recovery path was not found: $requiredPath"
    }
}

$resultRows = if (Test-Path -LiteralPath $ResultsPath -PathType Leaf) {
    @(Import-Csv -LiteralPath $ResultsPath)
}
else {
    @()
}
$completedEpoch = if ($resultRows.Count -gt 0) {
    [int][double]$resultRows[-1].epoch
}
else {
    0
}
if ($completedEpoch -ge $TargetEpochs) {
    Write-Output "Training is already complete at epoch $completedEpoch/$TargetEpochs; nothing was started."
    exit 0
}

$state = $null
if (Test-Path -LiteralPath $StatePath -PathType Leaf) {
    $state = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json
}
if ($null -ne $state -and $null -ne $state.runner_pid) {
    $recordedPid = [int]$state.runner_pid
    $existingRunner = Get-Process -Id $recordedPid -ErrorAction SilentlyContinue
    if ($null -ne $existingRunner -and $null -ne $state.job.started_at -and $state.job.started_at -ne "") {
        $expectedStart = [DateTimeOffset]::Parse([string]$state.job.started_at)
        $actualStart = [DateTimeOffset]$existingRunner.StartTime
        $identityMatches =
            $existingRunner.ProcessName -like "powershell*" -and
            [Math]::Abs(($actualStart - $expectedStart).TotalSeconds) -lt 30
        if ($identityMatches) {
            Write-Output "The recorded runner is already active (PID $recordedPid) at epoch $completedEpoch/$TargetEpochs; refusing to launch a duplicate."
            exit 0
        }
    }
}

# A real Ultralytics checkpoint is tens of megabytes. Check it twice so a
# concurrently written or truncated file is never used for resume.
$checkpointBefore = Get-Item -LiteralPath $LastCheckpoint
Start-Sleep -Seconds 2
$checkpointAfter = Get-Item -LiteralPath $LastCheckpoint
if ($checkpointAfter.Length -lt 10000000) {
    throw "last.pt is unexpectedly small ($($checkpointAfter.Length) bytes); refusing to resume"
}
if ($checkpointBefore.Length -ne $checkpointAfter.Length -or $checkpointBefore.LastWriteTimeUtc -ne $checkpointAfter.LastWriteTimeUtc) {
    throw "last.pt is still changing; wait for the current writer to stop, then rerun this script"
}

New-Item -ItemType Directory -Path $QueueRoot -Force | Out-Null
if (Test-Path -LiteralPath $CompleteSentinel -PathType Leaf) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $sentinelBackup = Join-Path $QueueRoot ("queue.complete.pre_resume_" + $stamp + ".bak")
    Move-Item -LiteralPath $CompleteSentinel -Destination $sentinelBackup
    Write-Output "Preserved stale terminal sentinel as $sentinelBackup"
}
if (Test-Path -LiteralPath $PausedSentinel -PathType Leaf) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $pauseBackup = Join-Path $QueueRoot ("queue.paused.resumed_" + $stamp + ".bak")
    Move-Item -LiteralPath $PausedSentinel -Destination $pauseBackup
    Write-Output "Preserved pause marker as $pauseBackup"
}

$launchStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdoutPath = Join-Path $QueueRoot ("resume_after_reboot_" + $launchStamp + ".stdout.log")
$stderrPath = Join-Path $QueueRoot ("resume_after_reboot_" + $launchStamp + ".stderr.log")
$argumentList = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", ('"{0}"' -f $RunnerScript),
    "-ProjectRoot", ('"{0}"' -f $ProjectRoot),
    "-DependencyQueueName", $DependencyQueueName,
    "-QueueName", $QueueName,
    "-RunName", $RunName,
    "-TargetEpochs", [string]$TargetEpochs,
    "-Workers", "0"
)
$startParameters = @{
    FilePath = $PowerShell
    ArgumentList = $argumentList
    WorkingDirectory = $ProjectRoot
    WindowStyle = "Hidden"
    RedirectStandardOutput = $stdoutPath
    RedirectStandardError = $stderrPath
    PassThru = $true
}
$process = Start-Process @startParameters

Start-Sleep -Seconds 2
$startedProcess = Get-Process -Id $process.Id -ErrorAction SilentlyContinue
if ($null -eq $startedProcess) {
    throw "Recovery runner exited immediately; inspect $stderrPath and $stdoutPath"
}

[ordered]@{
    status = "resume_runner_started"
    runner_pid = $process.Id
    resume_from_completed_epoch = $completedEpoch
    target_epochs = $TargetEpochs
    checkpoint = $LastCheckpoint
    stdout_log = $stdoutPath
    stderr_log = $stderrPath
} | ConvertTo-Json -Depth 3
