param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$PriorityQueueName = "priority_hsi_submission_2026-09-14",
    [string]$WatchName = "score_pipeline_watchdog_2026-09-13",
    [int]$PollSeconds = 30
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)
$PriorityRoot = Join-Path $ProjectRoot ("artifacts\background_queue\" + $PriorityQueueName)
$PriorityState = Join-Path $PriorityRoot "state.json"
$PriorityComplete = Join-Path $PriorityRoot "queue.complete"
$WatchScript = Join-Path $ProjectRoot "scripts\watch_background_pipeline.ps1"
$WatchRoot = Join-Path $ProjectRoot ("artifacts\background_queue\" + $WatchName)
$WatchState = Join-Path $WatchRoot "state.json"
$ResumeRoot = Join-Path $ProjectRoot "artifacts\background_queue\resume_after_priority_2026-09-14"
$ResumeState = Join-Path $ResumeRoot "state.json"
$ResumeLog = Join-Path $ResumeRoot "events.log"
$PollSeconds = [Math]::Max(10, $PollSeconds)

foreach ($requiredPath in @($PriorityRoot, $WatchScript)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required path was not found: $requiredPath"
    }
}
New-Item -ItemType Directory -Path $ResumeRoot -Force | Out-Null

function Get-IsoTimestamp {
    return (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssK")
}

function Write-ResumeEvent([string]$Message) {
    $line = "$(Get-IsoTimestamp) $Message"
    Add-Content -LiteralPath $ResumeLog -Value $line -Encoding UTF8
    Write-Output $line
}

function Save-ResumeState([string]$Status, [object]$Details) {
    [ordered]@{
        schema_version = 1
        status = $Status
        runner_pid = $PID
        project_root = $ProjectRoot
        priority_queue = $PriorityQueueName
        watch_name = $WatchName
        updated_at = Get-IsoTimestamp
        details = $Details
    } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ResumeState -Encoding UTF8
}

Save-ResumeState "waiting" $null
Write-ResumeEvent "waiting for priority queue to finish before resuming the long search"

$priorityTerminalStatus = $null
while ($null -eq $priorityTerminalStatus) {
    if (Test-Path -LiteralPath $PriorityComplete -PathType Leaf) {
        $priorityTerminalStatus = "complete"
        break
    }
    if (Test-Path -LiteralPath $PriorityState -PathType Leaf) {
        try {
            $state = Get-Content -LiteralPath $PriorityState -Raw | ConvertFrom-Json
            if ($state.queue_status -eq "runner_failed") {
                $priorityTerminalStatus = "runner_failed"
                break
            }
            $priorityRunner = Get-Process -Id ([int]$state.runner_pid) -ErrorAction SilentlyContinue
            if ($null -eq $priorityRunner -and $state.queue_status -notin @("complete", "runner_failed")) {
                $priorityTerminalStatus = "runner_stopped_unexpectedly"
                break
            }
        }
        catch {
            Write-ResumeEvent "priority state read deferred: $($_.Exception.Message)"
        }
    }
    Start-Sleep -Seconds $PollSeconds
}

Write-ResumeEvent "priority queue terminal status=$priorityTerminalStatus; checking main watchdog"
$watchdogActive = $false
if (Test-Path -LiteralPath $WatchState -PathType Leaf) {
    try {
        $currentWatchState = Get-Content -LiteralPath $WatchState -Raw | ConvertFrom-Json
        $currentWatchdog = Get-Process -Id ([int]$currentWatchState.watchdog_pid) -ErrorAction SilentlyContinue
        $watchdogActive = $null -ne $currentWatchdog -and $currentWatchdog.ProcessName -eq "powershell"
    }
    catch {
        Write-ResumeEvent "watchdog state could not be read; a new watchdog will be started"
    }
}

$details = [ordered]@{
    priority_terminal_status = $priorityTerminalStatus
    watchdog_was_active = $watchdogActive
    watchdog_pid = $null
}
if (-not $watchdogActive) {
    $watchStdout = Join-Path $WatchRoot ("watchdog_priority_resume_" + (Get-Date).ToString("yyyyMMdd_HHmmss") + ".stdout.log")
    $watchStderr = Join-Path $WatchRoot ("watchdog_priority_resume_" + (Get-Date).ToString("yyyyMMdd_HHmmss") + ".stderr.log")
    $watchdog = Start-Process -FilePath "C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe" -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $WatchScript,
        "-ProjectRoot", $ProjectRoot,
        "-WatchName", $WatchName
    ) -WorkingDirectory $ProjectRoot -WindowStyle Hidden -RedirectStandardOutput $watchStdout -RedirectStandardError $watchStderr -PassThru
    $details.watchdog_pid = $watchdog.Id
    Write-ResumeEvent "restarted main watchdog; pid=$($watchdog.Id)"
}
else {
    $details.watchdog_pid = $currentWatchdog.Id
    Write-ResumeEvent "main watchdog is already active; no duplicate was started"
}

Save-ResumeState "complete" $details
