param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$DependencyQueueName = "score_search_2026-09-13",
    [string]$QueueName = "multispectral_followup_2026-09-13",
    [string]$RunName = "ablation_s1024_b4_hsi16_shared_p005_995_r1",
    [ValidateRange(1, 1000)]
    [int]$TargetEpochs = 30,
    [ValidateRange(0, 16)]
    [int]$Workers = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$TrainScript = Join-Path $ProjectRoot "scripts\train_baseline.py"
$DataPath = Join-Path $ProjectRoot "data\processed\hsi16_shared_p005_995\dataset.yaml"
$ModelPath = Join-Path $ProjectRoot "yolo26s.pt"
$DependencyRoot = Join-Path $ProjectRoot ("artifacts\background_queue\" + $DependencyQueueName)
$DependencySentinel = Join-Path $DependencyRoot "queue.complete"
$QueueRoot = Join-Path $ProjectRoot ("artifacts\background_queue\" + $QueueName)
$StatePath = Join-Path $QueueRoot "state.json"
$EventLog = Join-Path $QueueRoot "events.log"
$CompleteSentinel = Join-Path $QueueRoot "queue.complete"
$RunDir = Join-Path $ProjectRoot ("runs\" + $RunName)
$ResultsPath = Join-Path $RunDir "results.csv"
$LastCheckpoint = Join-Path $RunDir "weights\last.pt"

foreach ($requiredFile in @($Python, $TrainScript, $DataPath, $ModelPath)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required file was not found: $requiredFile"
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot ".git"))) {
    throw "Refusing to run outside the expected Git project: $ProjectRoot"
}

New-Item -ItemType Directory -Path $QueueRoot -Force | Out-Null

function Get-IsoTimestamp {
    return (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssK")
}

function Write-QueueEvent([string]$Message) {
    $line = "$(Get-IsoTimestamp) $Message"
    Add-Content -LiteralPath $EventLog -Value $line -Encoding UTF8
    Write-Output $line
}

function Get-CompletedEpoch {
    if (-not (Test-Path -LiteralPath $ResultsPath -PathType Leaf)) {
        return 0
    }
    $rows = @(Import-Csv -LiteralPath $ResultsPath)
    if ($rows.Count -eq 0) {
        return 0
    }
    return [int][double]$rows[-1].epoch
}

function Get-BestMetric {
    if (-not (Test-Path -LiteralPath $ResultsPath -PathType Leaf)) {
        return $null
    }
    $best = @(Import-Csv -LiteralPath $ResultsPath) |
        Where-Object { $_.'metrics/mAP50-95(B)' -ne "" } |
        Sort-Object { [double]$_.'metrics/mAP50-95(B)' } -Descending |
        Select-Object -First 1
    if ($null -eq $best) {
        return $null
    }
    return [ordered]@{
        epoch = [int][double]$best.epoch
        map50_95 = [double]$best.'metrics/mAP50-95(B)'
        map50 = [double]$best.'metrics/mAP50(B)'
    }
}

function Save-State(
    [string]$QueueStatus,
    [string]$JobStatus,
    [Nullable[int]]$ExitCode = $null,
    [string]$StartedAt = $null,
    [string]$CompletedAt = $null
) {
    $state = [ordered]@{
        schema_version = 1
        queue_name = $QueueName
        queue_status = $QueueStatus
        runner_pid = $PID
        project_root = $ProjectRoot
        dependency_queue = $DependencyQueueName
        dependency_sentinel = $DependencySentinel
        updated_at = Get-IsoTimestamp
        job = [ordered]@{
            name = $RunName
            rationale = "Native 16-channel P0.5-P99.5 shared scaling preserves cross-band structure and exposes all hyperspectral bands to the detector."
            target_epochs = $TargetEpochs
            status = $JobStatus
            started_at = $StartedAt
            completed_at = $CompletedAt
            exit_code = $ExitCode
            completed_epoch = Get-CompletedEpoch
            best_metric = Get-BestMetric
            log = (Join-Path $QueueRoot ($RunName + ".log"))
        }
    }
    $state | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $StatePath -Encoding UTF8
}

if (Test-Path -LiteralPath $CompleteSentinel -PathType Leaf) {
    Write-Output "Follow-up queue is already complete: $CompleteSentinel"
    exit 0
}

$startedAt = $null
Push-Location $ProjectRoot
try {
    $env:YOLO_CONFIG_DIR = (Join-Path $ProjectRoot ".ultralytics")
    Save-State "waiting_for_dependency" "pending"
    Write-QueueEvent "queue started; runner_pid=$PID; waiting_for=$DependencyQueueName"

    $lastHeartbeat = Get-Date
    while (-not (Test-Path -LiteralPath $DependencySentinel -PathType Leaf)) {
        Start-Sleep -Seconds 30
        if (((Get-Date) - $lastHeartbeat).TotalMinutes -ge 5) {
            Save-State "waiting_for_dependency" "pending"
            Write-QueueEvent "still waiting for dependency queue $DependencyQueueName"
            $lastHeartbeat = Get-Date
        }
    }
    $dependencyResult = (Get-Content -LiteralPath $DependencySentinel -Raw).Trim()
    Write-QueueEvent "dependency complete: $dependencyResult"

    $completedEpoch = Get-CompletedEpoch
    if ($completedEpoch -ge $TargetEpochs) {
        $completedAt = Get-IsoTimestamp
        Save-State "complete" "skipped_complete" 0 $null $completedAt
        Set-Content -LiteralPath $CompleteSentinel -Value "$completedAt complete" -Encoding UTF8
        Write-QueueEvent "job already complete at epoch $completedEpoch; skipped"
        exit 0
    }

    $startedAt = Get-IsoTimestamp
    Save-State "running" "running" $null $startedAt $null
    Write-QueueEvent "job $RunName starting; target_epochs=$TargetEpochs"

    if (Test-Path -LiteralPath $RunDir) {
        if (-not (Test-Path -LiteralPath $LastCheckpoint -PathType Leaf)) {
            $completedAt = Get-IsoTimestamp
            Save-State "complete_with_failures" "failed" -2 $startedAt $completedAt
            Set-Content -LiteralPath $CompleteSentinel -Value "$completedAt complete_with_failures" -Encoding UTF8
            Write-QueueEvent "job failed safely: existing incomplete directory has no last.pt"
            exit 2
        }
        $commandArguments = @(
            $TrainScript,
            "--resume", $LastCheckpoint,
            "--workers", [string]$Workers
        )
        Write-QueueEvent "job resuming from $LastCheckpoint"
    }
    else {
        $commandArguments = @(
            $TrainScript,
            "--model", $ModelPath,
            "--data", $DataPath,
            "--epochs", [string]$TargetEpochs,
            "--imgsz", "1024",
            "--batch", "4",
            "--device", "0",
            "--workers", [string]$Workers,
            "--seed", "2026",
            "--name", $RunName,
            "--hsv-h", "0",
            "--hsv-s", "0",
            "--hsv-v", "0",
            "--extra-channel-init", "zero"
        )
    }

    $logPath = Join-Path $QueueRoot ($RunName + ".log")
    # Windows PowerShell converts each native stderr line into an ErrorRecord.
    # With the script-wide Stop preference, a real Python traceback would stop
    # this pipeline after its first line and discard the actionable exception.
    # Let the native process finish streaming, then handle its exit code below.
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Python @commandArguments 2>&1 |
            Tee-Object -FilePath $logPath -Append
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    $completedAt = Get-IsoTimestamp
    $completedEpoch = Get-CompletedEpoch
    if ($exitCode -eq 0 -and $completedEpoch -ge $TargetEpochs) {
        Save-State "complete" "complete" $exitCode $startedAt $completedAt
        Set-Content -LiteralPath $CompleteSentinel -Value "$completedAt complete" -Encoding UTF8
        Write-QueueEvent "job complete; best_map50_95=$((Get-BestMetric).map50_95)"
    }
    else {
        Save-State "complete_with_failures" "failed" $exitCode $startedAt $completedAt
        Set-Content -LiteralPath $CompleteSentinel -Value "$completedAt complete_with_failures" -Encoding UTF8
        Write-QueueEvent "job failed; exit_code=$exitCode; completed_epoch=$completedEpoch"
        exit 1
    }
}
catch {
    $completedAt = Get-IsoTimestamp
    Save-State "runner_failed" "failed" -3 $startedAt $completedAt
    Write-QueueEvent "queue runner failed: $($_.Exception.Message)"
    throw
}
finally {
    Pop-Location
}
