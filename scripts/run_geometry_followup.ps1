param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$DependencyQueueName = "multispectral_followup_2026-09-13",
    [string]$QueueName = "geometry_followup_2026-09-13"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$TrainScript = Join-Path $ProjectRoot "scripts\train_baseline.py"
$PseudoRgbDataPath = Join-Path $ProjectRoot "data\processed\pseudo_rgb\dataset.yaml"
$HsiDataPath = Join-Path $ProjectRoot "data\processed\hsi16_shared_p005_995\dataset.yaml"
$PretrainedWeights = Join-Path $ProjectRoot "yolo26s.pt"
$MediumPretrainedWeights = Join-Path $ProjectRoot "yolo26m.pt"
$LargePretrainedWeights = Join-Path $ProjectRoot "yolo26l.pt"
$DependencyRoot = Join-Path $ProjectRoot ("artifacts\background_queue\" + $DependencyQueueName)
$DependencySentinel = Join-Path $DependencyRoot "queue.complete"
$QueueRoot = Join-Path $ProjectRoot ("artifacts\background_queue\" + $QueueName)
$StatePath = Join-Path $QueueRoot "state.json"
$EventLog = Join-Path $QueueRoot "events.log"
$CompleteSentinel = Join-Path $QueueRoot "queue.complete"

foreach ($requiredFile in @(
    $Python,
    $TrainScript,
    $PseudoRgbDataPath,
    $HsiDataPath,
    $PretrainedWeights,
    $MediumPretrainedWeights,
    $LargePretrainedWeights
)) {
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

function Get-CompletedEpoch([string]$ResultsPath) {
    if (-not (Test-Path -LiteralPath $ResultsPath -PathType Leaf)) {
        return 0
    }
    $rows = @(Import-Csv -LiteralPath $ResultsPath)
    if ($rows.Count -eq 0) {
        return 0
    }
    return [int][double]$rows[-1].epoch
}

function Get-BestMetric([string]$ResultsPath) {
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

function Save-State([string]$QueueStatus, [object[]]$JobStates) {
    $state = [ordered]@{
        schema_version = 1
        queue_name = $QueueName
        queue_status = $QueueStatus
        runner_pid = $PID
        project_root = $ProjectRoot
        dependency_queue = $DependencyQueueName
        dependency_sentinel = $DependencySentinel
        updated_at = Get-IsoTimestamp
        jobs = $JobStates
    }
    $state | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $StatePath -Encoding UTF8
}

# These are independent geometry tests on the same frozen 2400/600 split.
# They are sequential so a single 8 GB GPU is never oversubscribed.
$Jobs = @(
    [ordered]@{
        name = "ablation_s1024_b3_hsi16_shared_p005_995_p2_e60_r1"
        rationale = "Combine the two modality-specific improvements supported by the public reference: all 16 bands for material identity and a stride-4 P2 head for box tightness."
        model = "yolo26s-p2.yaml"
        data = $HsiDataPath
        epochs = 60
        batch = 3
        workers = 0
        arguments = @(
            "--load-weights", $PretrainedWeights,
            "--hsv-h", "0", "--hsv-s", "0", "--hsv-v", "0",
            "--extra-channel-init", "zero"
        )
    },
    [ordered]@{
        name = "ablation_s1024_b4_b5-8-13_multiscale025_e60_r1"
        rationale = "Batch-level 0.75x-1.25x multi-scale training directly targets the observed within-class scale spread while preserving the baseline affine augmentation."
        model = $PretrainedWeights
        data = $PseudoRgbDataPath
        epochs = 60
        batch = 4
        workers = 4
        arguments = @("--multi-scale", "0.25")
    },
    [ordered]@{
        name = "ablation_m1024_b3_b5-8-13_e60_clean_r1"
        rationale = "A fresh 60-epoch YOLO26m run tests the strong uncontaminated epoch-24 signal without reusing the earlier mixed-resolution resume."
        model = $MediumPretrainedWeights
        data = $PseudoRgbDataPath
        epochs = 60
        batch = 3
        workers = 4
        arguments = @()
    },
    [ordered]@{
        name = "ablation_l1024_b2_b5-8-13_e60_clean_r1"
        rationale = "YOLO26l is a bounded final capacity test: only 17 percent more parameters than m, with batch 2 chosen for the 8 GB GPU."
        model = $LargePretrainedWeights
        data = $PseudoRgbDataPath
        epochs = 60
        batch = 2
        workers = 4
        arguments = @()
    },
    [ordered]@{
        name = "ablation_s1024_b4_b5-8-13_scale075_e60_r1"
        rationale = "Wider random-affine scale range targets the large within-class size variation of car, e-bike, people, and stone_block."
        model = $PretrainedWeights
        data = $PseudoRgbDataPath
        epochs = 60
        batch = 4
        workers = 4
        arguments = @("--scale", "0.75")
    },
    [ordered]@{
        name = "ablation_s1024_b3_b5-8-13_p2_e60_r1"
        rationale = "The architecture-only YOLO26s P2 head adds stride-4 predictions for tighter small-object boxes while transferring compatible YOLO26s weights."
        model = "yolo26s-p2.yaml"
        data = $PseudoRgbDataPath
        epochs = 60
        batch = 3
        workers = 4
        arguments = @("--load-weights", $PretrainedWeights)
    }
)

$JobStates = @()
foreach ($job in $Jobs) {
    $JobStates += [ordered]@{
        name = $job.name
        rationale = $job.rationale
        target_epochs = $job.epochs
        batch = $job.batch
        workers = $job.workers
        status = "pending"
        started_at = $null
        completed_at = $null
        exit_code = $null
        completed_epoch = 0
        best_metric = $null
        log = (Join-Path $QueueRoot ($job.name + ".log"))
    }
}

if (Test-Path -LiteralPath $CompleteSentinel -PathType Leaf) {
    Write-Output "Geometry queue is already complete: $CompleteSentinel"
    exit 0
}

Push-Location $ProjectRoot
try {
    $env:YOLO_CONFIG_DIR = (Join-Path $ProjectRoot ".ultralytics")
    Save-State "waiting_for_dependency" $JobStates
    Write-QueueEvent "queue started; runner_pid=$PID; waiting_for=$DependencyQueueName"

    $lastHeartbeat = Get-Date
    while (-not (Test-Path -LiteralPath $DependencySentinel -PathType Leaf)) {
        Start-Sleep -Seconds 30
        if (((Get-Date) - $lastHeartbeat).TotalMinutes -ge 5) {
            Save-State "waiting_for_dependency" $JobStates
            Write-QueueEvent "still waiting for dependency queue $DependencyQueueName"
            $lastHeartbeat = Get-Date
        }
    }
    $dependencyResult = (Get-Content -LiteralPath $DependencySentinel -Raw).Trim()
    Write-QueueEvent "dependency complete: $dependencyResult"

    for ($index = 0; $index -lt $Jobs.Count; $index++) {
        $job = $Jobs[$index]
        $jobState = $JobStates[$index]
        $runDir = Join-Path $ProjectRoot ("runs\" + $job.name)
        $resultsPath = Join-Path $runDir "results.csv"
        $lastCheckpoint = Join-Path $runDir "weights\last.pt"
        $completedEpoch = Get-CompletedEpoch $resultsPath
        $jobState.completed_epoch = $completedEpoch

        if ($completedEpoch -ge $job.epochs) {
            $jobState.status = "skipped_complete"
            $jobState.best_metric = Get-BestMetric $resultsPath
            $jobState.completed_at = Get-IsoTimestamp
            Write-QueueEvent "job $($job.name) already complete at epoch $completedEpoch; skipped"
            Save-State "running" $JobStates
            continue
        }

        $jobState.status = "running"
        $jobState.started_at = Get-IsoTimestamp
        Save-State "running" $JobStates
        Write-QueueEvent "job $($job.name) starting; target_epochs=$($job.epochs); batch=$($job.batch); workers=$($job.workers)"

        if (Test-Path -LiteralPath $runDir) {
            if (-not (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf)) {
                $jobState.status = "failed"
                $jobState.exit_code = -2
                $jobState.completed_at = Get-IsoTimestamp
                Write-QueueEvent "job $($job.name) failed safely: existing incomplete directory has no last.pt"
                Save-State "running" $JobStates
                continue
            }
            $commandArguments = @(
                $TrainScript,
                "--resume", $lastCheckpoint,
                "--workers", [string]$job.workers
            )
            Write-QueueEvent "job $($job.name) resuming from $lastCheckpoint"
        }
        else {
            $commandArguments = @(
                $TrainScript,
                "--model", $job.model,
                "--data", $job.data,
                "--epochs", [string]$job.epochs,
                "--imgsz", "1024",
                "--batch", [string]$job.batch,
                "--device", "0",
                "--workers", [string]$job.workers,
                "--seed", "2026",
                "--name", $job.name
            ) + $job.arguments
        }

        $previousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & $Python @commandArguments 2>&1 |
                Tee-Object -FilePath $jobState.log -Append
            $exitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $previousErrorActionPreference
        }
        $jobState.exit_code = $exitCode
        $jobState.completed_epoch = Get-CompletedEpoch $resultsPath
        $jobState.best_metric = Get-BestMetric $resultsPath
        $jobState.completed_at = Get-IsoTimestamp
        if ($exitCode -eq 0 -and $jobState.completed_epoch -ge $job.epochs) {
            $jobState.status = "complete"
            Write-QueueEvent "job $($job.name) complete; best_map50_95=$($jobState.best_metric.map50_95)"
        }
        else {
            $jobState.status = "failed"
            Write-QueueEvent "job $($job.name) failed; exit_code=$exitCode; completed_epoch=$($jobState.completed_epoch)"
        }
        Save-State "running" $JobStates
    }

    $failedCount = @($JobStates | Where-Object { $_.status -eq "failed" }).Count
    $queueStatus = if ($failedCount -eq 0) { "complete" } else { "complete_with_failures" }
    Save-State $queueStatus $JobStates
    Set-Content -LiteralPath $CompleteSentinel -Value "$(Get-IsoTimestamp) $queueStatus" -Encoding UTF8
    Write-QueueEvent "queue finished; status=$queueStatus; failed_jobs=$failedCount"
}
catch {
    Save-State "runner_failed" $JobStates
    Write-QueueEvent "queue runner failed: $($_.Exception.Message)"
    throw
}
finally {
    Pop-Location
}
