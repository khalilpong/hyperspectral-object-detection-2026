param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 60)]
    [int]$TargetCompletedEpoch,
    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 2147483647)]
    [int]$ExpectedRunnerPid
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)
$QueueName = "hsi_e60_validation_2026-09-15"
$RunName = "ablation_s1024_b4_hsi16_shared_p005_995_e60_r2"
$QueueRoot = Join-Path $ProjectRoot ("artifacts\background_queue\" + $QueueName)
$StatePath = Join-Path $QueueRoot "state.json"
$MonitorStatePath = Join-Path $QueueRoot "pause_monitor.json"
$EventLog = Join-Path $QueueRoot "events.log"
$CompleteSentinel = Join-Path $QueueRoot "queue.complete"
$PausedSentinel = Join-Path $QueueRoot "queue.paused"
$RunDir = Join-Path $ProjectRoot ("runs\" + $RunName)
$ResultsPath = Join-Path $RunDir "results.csv"
$LastCheckpoint = Join-Path $RunDir "weights\last.pt"

foreach ($requiredPath in @(
    (Join-Path $ProjectRoot ".git"),
    $StatePath,
    $ResultsPath,
    $LastCheckpoint
)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required pause path was not found: $requiredPath"
    }
}

function Get-IsoTimestamp {
    return (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssK")
}

function Get-CompletedEpoch {
    $rows = @(Import-Csv -LiteralPath $ResultsPath)
    if ($rows.Count -eq 0) {
        return 0
    }
    return [int][double]$rows[-1].epoch
}

function Write-MonitorState([string]$Status, [string]$Message) {
    [ordered]@{
        schema_version = 1
        status = $Status
        message = $Message
        monitor_pid = $PID
        expected_runner_pid = $ExpectedRunnerPid
        run_name = $RunName
        target_completed_epoch = $TargetCompletedEpoch
        observed_completed_epoch = Get-CompletedEpoch
        updated_at = Get-IsoTimestamp
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $MonitorStatePath -Encoding UTF8
    Add-Content -LiteralPath $EventLog -Value "$(Get-IsoTimestamp) pause-monitor: $Message" -Encoding UTF8
}

function Assert-RunnerIdentity {
    $state = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json
    if ([int]$state.runner_pid -ne $ExpectedRunnerPid) {
        throw "Queue state runner PID $($state.runner_pid) does not match expected PID $ExpectedRunnerPid"
    }
    $runner = Get-Process -Id $ExpectedRunnerPid -ErrorAction SilentlyContinue
    if ($null -eq $runner) {
        throw "Expected runner PID $ExpectedRunnerPid is no longer running"
    }
    if ($runner.ProcessName -notlike "powershell*") {
        throw "Expected runner PID $ExpectedRunnerPid is $($runner.ProcessName), not PowerShell"
    }
    $expectedStart = [DateTimeOffset]::Parse([string]$state.job.started_at)
    $actualStart = [DateTimeOffset]$runner.StartTime
    if ([Math]::Abs(($actualStart - $expectedStart).TotalSeconds) -ge 30) {
        throw "Runner PID exists but its start time does not match queue state"
    }
    return $runner
}

function Get-DescendantProcesses([int]$RootPid) {
    $allProcesses = @(Get-CimInstance Win32_Process)
    $frontier = @([pscustomobject]@{ ProcessId = $RootPid; Depth = 0 })
    $descendants = @()
    while ($frontier.Count -gt 0) {
        $current = $frontier[0]
        if ($frontier.Count -eq 1) {
            $frontier = @()
        }
        else {
            $frontier = @($frontier[1..($frontier.Count - 1)])
        }
        $children = @($allProcesses | Where-Object { [int]$_.ParentProcessId -eq [int]$current.ProcessId })
        foreach ($child in $children) {
            $node = [pscustomobject]@{
                ProcessId = [int]$child.ProcessId
                ParentProcessId = [int]$child.ParentProcessId
                Name = [string]$child.Name
                CommandLine = [string]$child.CommandLine
                Depth = [int]$current.Depth + 1
            }
            $descendants += $node
            $frontier += $node
        }
    }
    return $descendants
}

try {
    [void](Assert-RunnerIdentity)
    $initialEpoch = Get-CompletedEpoch
    if ($initialEpoch -gt $TargetCompletedEpoch) {
        throw "Training already passed requested pause epoch: current=$initialEpoch target=$TargetCompletedEpoch"
    }
    Write-MonitorState "armed" "armed at epoch $initialEpoch; will pause immediately after stable epoch $TargetCompletedEpoch checkpoint"

    while ((Get-CompletedEpoch) -lt $TargetCompletedEpoch) {
        Start-Sleep -Seconds 5
        [void](Assert-RunnerIdentity)
        Write-MonitorState "waiting" "waiting for results.csv epoch $TargetCompletedEpoch"
    }

    Write-MonitorState "checkpoint_wait" "epoch $TargetCompletedEpoch row observed; waiting for last.pt to become stable"
    $checkpointStable = $false
    for ($attempt = 1; $attempt -le 30 -and -not $checkpointStable; $attempt++) {
        $resultsInfo = Get-Item -LiteralPath $ResultsPath
        $before = Get-Item -LiteralPath $LastCheckpoint
        Start-Sleep -Seconds 2
        $after = Get-Item -LiteralPath $LastCheckpoint
        $checkpointStable =
            $after.Length -ge 10000000 -and
            $before.Length -eq $after.Length -and
            $before.LastWriteTimeUtc -eq $after.LastWriteTimeUtc -and
            $after.LastWriteTimeUtc -ge $resultsInfo.LastWriteTimeUtc
    }
    if (-not $checkpointStable) {
        throw "Epoch row was written but last.pt did not become stable within 60 seconds"
    }

    [void](Assert-RunnerIdentity)
    $descendants = @(Get-DescendantProcesses -RootPid $ExpectedRunnerPid)
    $trainingProcesses = @($descendants | Where-Object {
        $_.Name -match '^python(w)?\.exe$' -and
        ($_.CommandLine -like ("*" + $RunName + "*") -or $_.CommandLine -like "*scripts\train_baseline.py*")
    })
    if ($trainingProcesses.Count -eq 0) {
        $summary = ($descendants | ForEach-Object { "$($_.ProcessId):$($_.Name)" }) -join ", "
        throw "No exact training Python descendant was found under runner $ExpectedRunnerPid. Descendants: $summary"
    }

    Write-MonitorState "stopping" "stable epoch $TargetCompletedEpoch checkpoint confirmed; stopping exact training descendants"
    foreach ($trainingProcess in @($trainingProcesses | Sort-Object Depth -Descending)) {
        Stop-Process -Id $trainingProcess.ProcessId -Force -ErrorAction Stop
    }

    $runnerExited = $false
    for ($attempt = 1; $attempt -le 30 -and -not $runnerExited; $attempt++) {
        Start-Sleep -Seconds 1
        $runnerExited = $null -eq (Get-Process -Id $ExpectedRunnerPid -ErrorAction SilentlyContinue)
    }
    if (-not $runnerExited) {
        Stop-Process -Id $ExpectedRunnerPid -Force -ErrorAction Stop
        Start-Sleep -Seconds 1
    }

    if (Test-Path -LiteralPath $CompleteSentinel -PathType Leaf) {
        $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $sentinelBackup = Join-Path $QueueRoot ("queue.complete.paused_epoch_" + $TargetCompletedEpoch + "_" + $stamp + ".bak")
        Move-Item -LiteralPath $CompleteSentinel -Destination $sentinelBackup
    }

    $completedEpoch = Get-CompletedEpoch
    if ($completedEpoch -ne $TargetCompletedEpoch) {
        throw "Pause completed at unexpected epoch $completedEpoch; expected $TargetCompletedEpoch"
    }
    $pausedAt = Get-IsoTimestamp
    $state = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json
    $state.queue_status = "paused"
    $state.updated_at = $pausedAt
    $state.job.status = "paused"
    $state.job.completed_epoch = $completedEpoch
    $state.job.exit_code = $null
    $state.job | Add-Member -NotePropertyName paused_at -NotePropertyValue $pausedAt -Force
    $state.job | Add-Member -NotePropertyName pause_reason -NotePropertyValue "user_requested_after_epoch_$TargetCompletedEpoch" -Force
    $state | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $StatePath -Encoding UTF8
    Set-Content -LiteralPath $PausedSentinel -Value "$pausedAt paused epoch=$completedEpoch" -Encoding UTF8
    Write-MonitorState "paused" "training paused safely at epoch $completedEpoch; checkpoint is stable"
}
catch {
    Write-MonitorState "failed" $_.Exception.Message
    throw
}
