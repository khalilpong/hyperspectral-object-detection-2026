param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$QueueName = "score_search_2026-09-13"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$TrainScript = Join-Path $ProjectRoot "scripts\train_baseline.py"
$QueueRoot = Join-Path $ProjectRoot ("artifacts\background_queue\" + $QueueName)
$StatePath = Join-Path $QueueRoot "state.json"
$EventLog = Join-Path $QueueRoot "events.log"
$CompleteSentinel = Join-Path $QueueRoot "queue.complete"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Project Python was not found: $Python"
}
if (-not (Test-Path -LiteralPath $TrainScript -PathType Leaf)) {
    throw "Training script was not found: $TrainScript"
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

function Save-State([string]$QueueStatus, [object[]]$JobStates) {
    $state = [ordered]@{
        schema_version = 1
        queue_name = $QueueName
        queue_status = $QueueStatus
        runner_pid = $PID
        project_root = $ProjectRoot
        updated_at = Get-IsoTimestamp
        jobs = $JobStates
    }
    $state | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $StatePath -Encoding UTF8
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
    $rows = @(Import-Csv -LiteralPath $ResultsPath)
    $best = $rows |
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

# These jobs are deliberately sequential so the single 8 GB GPU is never
# oversubscribed. They only train and validate; no Kaggle upload is performed.
$Jobs = @(
    [ordered]@{
        name = "finetune_s1024_b5-8-13_clean_valueonly_e12_r1"
        rationale = "Cheap continuation probe from the current held-out best; low LR, no mosaic, value-only illumination jitter."
        model = "runs\ablation_s1024_b4_b5-8-13_r1\weights\best.pt"
        data = "data\processed\pseudo_rgb\dataset.yaml"
        epochs = 12
        arguments = @(
            "--optimizer", "AdamW", "--lr0", "0.0002", "--lrf", "0.1",
            "--warmup-epochs", "1", "--hsv-h", "0", "--hsv-s", "0",
            "--hsv-v", "0.2", "--mosaic", "0", "--close-mosaic", "0"
        )
    },
    [ordered]@{
        name = "ablation_s1024_b4_b5-8-13_valueonly_r1"
        rationale = "One-factor test of preserving pseudo-spectral channel ratios while retaining illumination jitter."
        model = "yolo26s.pt"
        data = "data\processed\pseudo_rgb\dataset.yaml"
        epochs = 30
        arguments = @("--hsv-h", "0", "--hsv-s", "0", "--hsv-v", "0.2")
    },
    [ordered]@{
        name = "ablation_s1024_b4_b5-8-13_e60_r1"
        rationale = "Test whether the 30-epoch baseline stopped while validation mAP was still improving."
        model = "yolo26s.pt"
        data = "data\processed\pseudo_rgb\dataset.yaml"
        epochs = 60
        arguments = @()
    },
    [ordered]@{
        name = "ablation_s1024_b4_b5-8-13_box10_r1"
        rationale = "One-factor localization-loss test because mAP50 is high but strict-IoU AP is the main gap."
        model = "yolo26s.pt"
        data = "data\processed\pseudo_rgb\dataset.yaml"
        epochs = 30
        arguments = @("--box", "10")
    }
)

$JobStates = @()
foreach ($job in $Jobs) {
    $JobStates += [ordered]@{
        name = $job.name
        rationale = $job.rationale
        target_epochs = $job.epochs
        status = "pending"
        started_at = $null
        completed_at = $null
        exit_code = $null
        completed_epoch = 0
        best_metric = $null
        log = (Join-Path $QueueRoot ($job.name + ".log"))
    }
}

Push-Location $ProjectRoot
try {
    $env:YOLO_CONFIG_DIR = (Join-Path $ProjectRoot ".ultralytics")
    Save-State "running" $JobStates
    Write-QueueEvent "queue started; runner_pid=$PID; jobs=$($Jobs.Count)"

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
        Write-QueueEvent "job $($job.name) starting; target_epochs=$($job.epochs)"

        if (Test-Path -LiteralPath $runDir) {
            if (-not (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf)) {
                $jobState.status = "failed"
                $jobState.exit_code = -2
                $jobState.completed_at = Get-IsoTimestamp
                Write-QueueEvent "job $($job.name) failed safely: existing incomplete directory has no last.pt"
                Save-State "running" $JobStates
                continue
            }
            $commandArguments = @($TrainScript, "--resume", $lastCheckpoint)
            Write-QueueEvent "job $($job.name) resuming from $lastCheckpoint"
        }
        else {
            $modelPath = if ([IO.Path]::IsPathRooted($job.model)) { $job.model } else { Join-Path $ProjectRoot $job.model }
            $dataPath = if ([IO.Path]::IsPathRooted($job.data)) { $job.data } else { Join-Path $ProjectRoot $job.data }
            $commandArguments = @(
                $TrainScript,
                "--model", $modelPath,
                "--data", $dataPath,
                "--epochs", [string]$job.epochs,
                "--imgsz", "1024",
                "--batch", "4",
                "--device", "0",
                "--workers", "4",
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
