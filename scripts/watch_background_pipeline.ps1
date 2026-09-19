param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$WatchName = "score_pipeline_watchdog_2026-09-13",
    [int]$PollSeconds = 60,
    [int]$MaxRestartsPerStage = 3
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)
$WatchRoot = Join-Path $ProjectRoot ("artifacts\background_queue\" + $WatchName)
$StatePath = Join-Path $WatchRoot "state.json"
$EventLog = Join-Path $WatchRoot "events.log"
$CompleteSentinel = Join-Path $WatchRoot "pipeline.complete"
$PollSeconds = [Math]::Max(10, $PollSeconds)
$MaxRestartsPerStage = [Math]::Max(1, $MaxRestartsPerStage)

if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot ".git"))) {
    throw "Refusing to run outside the expected Git project: $ProjectRoot"
}
New-Item -ItemType Directory -Path $WatchRoot -Force | Out-Null

$Stages = @(
    [ordered]@{
        name = "score_search_2026-09-13"
        script = Join-Path $ProjectRoot "scripts\run_background_experiments.ps1"
        complete_file = "queue.complete"
    },
    [ordered]@{
        name = "multispectral_followup_2026-09-13"
        script = Join-Path $ProjectRoot "scripts\run_multispectral_followup.ps1"
        complete_file = "queue.complete"
    },
    [ordered]@{
        name = "geometry_followup_2026-09-13"
        script = Join-Path $ProjectRoot "scripts\run_geometry_followup.ps1"
        complete_file = "queue.complete"
    },
    [ordered]@{
        name = "final_candidate_followup_v2_2026-09-13"
        script = Join-Path $ProjectRoot "scripts\run_final_candidate_followup.ps1"
        complete_file = "queue.complete"
    },
    [ordered]@{
        name = "submission_followup_2026-09-13"
        script = Join-Path $ProjectRoot "scripts\run_submission_followup.ps1"
        complete_file = "queue.complete"
    }
)

foreach ($stage in $Stages) {
    if (-not (Test-Path -LiteralPath $stage.script -PathType Leaf)) {
        throw "Stage script was not found: $($stage.script)"
    }
}

function Get-IsoTimestamp {
    return (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssK")
}

function Write-WatchEvent([string]$Message) {
    $line = "$(Get-IsoTimestamp) $Message"
    Add-Content -LiteralPath $EventLog -Value $line -Encoding UTF8
    Write-Output $line
}

function Save-WatchState([string]$Status, [object]$StageStates, [object]$RestartCounts) {
    $state = [ordered]@{
        schema_version = 1
        watch_name = $WatchName
        status = $Status
        watchdog_pid = $PID
        project_root = $ProjectRoot
        poll_seconds = $PollSeconds
        max_restarts_per_stage = $MaxRestartsPerStage
        updated_at = Get-IsoTimestamp
        restart_counts = $RestartCounts
        stages = $StageStates
    }
    $state | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $StatePath -Encoding UTF8
}

function Read-StageState([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $null
    }
    try {
        return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    }
    catch {
        Write-WatchEvent "state read deferred for $Path because it was not parseable: $($_.Exception.Message)"
        return $null
    }
}

if (Test-Path -LiteralPath $CompleteSentinel -PathType Leaf) {
    Write-Output "Pipeline watchdog already recorded completion: $CompleteSentinel"
    exit 0
}

$restartCounts = [ordered]@{}
foreach ($stage in $Stages) {
    $restartCounts[$stage.name] = 0
}

Write-WatchEvent "watchdog started; pid=$PID; stages=$($Stages.Count); automatic_upload=false"
$lastHeartbeat = Get-Date
while ($true) {
    $stageStates = @()
    foreach ($stage in $Stages) {
        $queueRoot = Join-Path $ProjectRoot ("artifacts\background_queue\" + $stage.name)
        $queueStatePath = Join-Path $queueRoot "state.json"
        $queueCompletePath = Join-Path $queueRoot $stage.complete_file
        $isComplete = Test-Path -LiteralPath $queueCompletePath -PathType Leaf
        $queueState = Read-StageState $queueStatePath
        $runnerPid = if ($null -ne $queueState) { [int]$queueState.runner_pid } else { $null }
        $runner = if ($null -ne $runnerPid) { Get-Process -Id $runnerPid -ErrorAction SilentlyContinue } else { $null }
        $isActive = $null -ne $runner -and $runner.ProcessName -eq "powershell"

        if (-not $isComplete -and -not $isActive -and $null -ne $queueState) {
            $restartCount = [int]$restartCounts[$stage.name]
            if ($restartCount -lt $MaxRestartsPerStage) {
                $newRunner = Start-Process -FilePath "powershell.exe" -ArgumentList @(
                    "-NoProfile",
                    "-ExecutionPolicy", "Bypass",
                    "-File", $stage.script,
                    "-ProjectRoot", $ProjectRoot
                ) -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru
                $restartCounts[$stage.name] = $restartCount + 1
                Write-WatchEvent (
                    "restarted stage=$($stage.name); previous_pid=$runnerPid; " +
                    "new_pid=$($newRunner.Id); prior_status=$($queueState.queue_status); " +
                    "attempt=$($restartCounts[$stage.name])"
                )
                $runnerPid = $newRunner.Id
                $isActive = $true
            }
            else {
                Write-WatchEvent "restart limit reached for stage=$($stage.name); manual inspection required"
            }
        }

        $stageStates += [ordered]@{
            name = $stage.name
            queue_status = if ($null -ne $queueState) { $queueState.queue_status } else { "state_unavailable" }
            runner_pid = $runnerPid
            active = $isActive
            complete = $isComplete
            restart_count = [int]$restartCounts[$stage.name]
        }
    }

    $terminalStage = $stageStates[-1]
    if ($terminalStage.complete) {
        $completedAt = Get-IsoTimestamp
        Save-WatchState "complete" $stageStates $restartCounts
        Set-Content -LiteralPath $CompleteSentinel -Value "$completedAt complete" -Encoding UTF8
        Write-WatchEvent "pipeline reached terminal submission stage; watchdog exiting"
        exit 0
    }

    Save-WatchState "watching" $stageStates $restartCounts
    if (((Get-Date) - $lastHeartbeat).TotalMinutes -ge 10) {
        $activeNames = @($stageStates | Where-Object { $_.active } | ForEach-Object { $_.name }) -join ","
        Write-WatchEvent "heartbeat; active_stages=$activeNames"
        $lastHeartbeat = Get-Date
    }
    Start-Sleep -Seconds $PollSeconds
}
