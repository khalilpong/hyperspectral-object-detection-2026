param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$DependencyQueueName = "geometry_followup_2026-09-13",
    [string]$QueueName = "final_candidate_followup_v2_2026-09-13",
    [double]$BaselineMap = 0.6887668231,
    [double]$MinimumGain = 0.003
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$TrainScript = Join-Path $ProjectRoot "scripts\train_baseline.py"
$InferenceComparisonScript = Join-Path $ProjectRoot "scripts\compare_inference_modes.py"
$ModelPath = Join-Path $ProjectRoot "yolo26s.pt"
$MediumModelPath = Join-Path $ProjectRoot "yolo26m.pt"
$LargeModelPath = Join-Path $ProjectRoot "yolo26l.pt"
$DependencyRoot = Join-Path $ProjectRoot ("artifacts\background_queue\" + $DependencyQueueName)
$DependencySentinel = Join-Path $DependencyRoot "queue.complete"
$QueueRoot = Join-Path $ProjectRoot ("artifacts\background_queue\" + $QueueName)
$StatePath = Join-Path $QueueRoot "state.json"
$EventLog = Join-Path $QueueRoot "events.log"
$CompleteSentinel = Join-Path $QueueRoot "queue.complete"

foreach ($requiredFile in @(
    $Python,
    $TrainScript,
    $InferenceComparisonScript,
    $ModelPath,
    $MediumModelPath,
    $LargeModelPath
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

function Save-State(
    [string]$QueueStatus,
    [object]$CandidateTable,
    [object]$Selection,
    [object]$Job
) {
    $state = [ordered]@{
        schema_version = 1
        queue_name = $QueueName
        queue_status = $QueueStatus
        runner_pid = $PID
        project_root = $ProjectRoot
        dependency_queue = $DependencyQueueName
        dependency_sentinel = $DependencySentinel
        baseline_map50_95 = $BaselineMap
        minimum_gain = $MinimumGain
        updated_at = Get-IsoTimestamp
        candidates = $CandidateTable
        selection = $Selection
        job = $Job
    }
    $state | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $StatePath -Encoding UTF8
}

$Candidates = @(
    [ordered]@{
        name = "hsi16_shared_p005_995_p2_e60"
        validation_run = "ablation_s1024_b3_hsi16_shared_p005_995_p2_e60_r1"
        final_name = "final_s1024_all3000_hsi16_shared_p005_995_p2_e60_r1"
        model = "yolo26s-p2.yaml"
        data = "data\processed\hsi16_shared_p005_995\dataset_all.yaml"
        validation_data = "data\processed\hsi16_shared_p005_995\dataset.yaml"
        epochs = 60
        batch = 3
        workers = 0
        arguments = @(
            "--load-weights", $ModelPath,
            "--hsv-h", "0", "--hsv-s", "0", "--hsv-v", "0",
            "--extra-channel-init", "zero"
        )
    },
    [ordered]@{
        name = "pseudo_rgb_multiscale025_e60"
        validation_run = "ablation_s1024_b4_b5-8-13_multiscale025_e60_r1"
        final_name = "final_s1024_all3000_b5-8-13_multiscale025_e60_r1"
        model = $ModelPath
        data = "data\processed\pseudo_rgb\dataset_all.yaml"
        validation_data = "data\processed\pseudo_rgb\dataset.yaml"
        epochs = 60
        batch = 4
        workers = 4
        arguments = @("--multi-scale", "0.25")
    },
    [ordered]@{
        name = "pseudo_rgb_yolo26m_e60"
        validation_run = "ablation_m1024_b3_b5-8-13_e60_clean_r1"
        final_name = "final_m1024_all3000_b5-8-13_e60_clean_r1"
        model = $MediumModelPath
        data = "data\processed\pseudo_rgb\dataset_all.yaml"
        validation_data = "data\processed\pseudo_rgb\dataset.yaml"
        epochs = 60
        batch = 3
        workers = 4
        arguments = @()
    },
    [ordered]@{
        name = "pseudo_rgb_yolo26l_e60"
        validation_run = "ablation_l1024_b2_b5-8-13_e60_clean_r1"
        final_name = "final_l1024_all3000_b5-8-13_e60_clean_r1"
        model = $LargeModelPath
        data = "data\processed\pseudo_rgb\dataset_all.yaml"
        validation_data = "data\processed\pseudo_rgb\dataset.yaml"
        epochs = 60
        batch = 2
        workers = 4
        arguments = @()
    },
    [ordered]@{
        name = "pseudo_rgb_e60"
        validation_run = "ablation_s1024_b4_b5-8-13_e60_r1"
        final_name = "final_s1024_all3000_b5-8-13_e60_r1"
        model = $ModelPath
        data = "data\processed\pseudo_rgb\dataset_all.yaml"
        validation_data = "data\processed\pseudo_rgb\dataset.yaml"
        epochs = 60
        batch = 4
        workers = 4
        arguments = @()
    },
    [ordered]@{
        name = "pseudo_rgb_box10"
        validation_run = "ablation_s1024_b4_b5-8-13_box10_r1"
        final_name = "final_s1024_all3000_b5-8-13_box10_r1"
        model = $ModelPath
        data = "data\processed\pseudo_rgb\dataset_all.yaml"
        validation_data = "data\processed\pseudo_rgb\dataset.yaml"
        epochs = 30
        batch = 4
        workers = 4
        arguments = @("--box", "10")
    },
    [ordered]@{
        name = "hsi16_shared_p005_995"
        validation_run = "ablation_s1024_b4_hsi16_shared_p005_995_r1"
        final_name = "final_s1024_all3000_hsi16_shared_p005_995_r1"
        model = $ModelPath
        data = "data\processed\hsi16_shared_p005_995\dataset_all.yaml"
        validation_data = "data\processed\hsi16_shared_p005_995\dataset.yaml"
        epochs = 30
        batch = 4
        workers = 0
        arguments = @(
            "--hsv-h", "0", "--hsv-s", "0", "--hsv-v", "0",
            "--extra-channel-init", "zero"
        )
    },
    [ordered]@{
        name = "pseudo_rgb_scale075"
        validation_run = "ablation_s1024_b4_b5-8-13_scale075_e60_r1"
        final_name = "final_s1024_all3000_b5-8-13_scale075_e60_r1"
        model = $ModelPath
        data = "data\processed\pseudo_rgb\dataset_all.yaml"
        validation_data = "data\processed\pseudo_rgb\dataset.yaml"
        epochs = 60
        batch = 4
        workers = 4
        arguments = @("--scale", "0.75")
    },
    [ordered]@{
        name = "pseudo_rgb_p2"
        validation_run = "ablation_s1024_b3_b5-8-13_p2_e60_r1"
        final_name = "final_s1024_all3000_b5-8-13_p2_e60_r1"
        model = "yolo26s-p2.yaml"
        data = "data\processed\pseudo_rgb\dataset_all.yaml"
        validation_data = "data\processed\pseudo_rgb\dataset.yaml"
        epochs = 60
        batch = 3
        workers = 4
        arguments = @("--load-weights", $ModelPath)
    }
)

if (Test-Path -LiteralPath $CompleteSentinel -PathType Leaf) {
    Write-Output "Final-candidate queue is already complete: $CompleteSentinel"
    exit 0
}

$candidateTable = @()
$selection = $null
$jobState = $null
$startedAt = $null
Push-Location $ProjectRoot
try {
    $env:YOLO_CONFIG_DIR = (Join-Path $ProjectRoot ".ultralytics")
    Save-State "waiting_for_dependency" $candidateTable $selection $jobState
    Write-QueueEvent "queue started; runner_pid=$PID; waiting_for=$DependencyQueueName"

    $lastHeartbeat = Get-Date
    while (-not (Test-Path -LiteralPath $DependencySentinel -PathType Leaf)) {
        Start-Sleep -Seconds 30
        if (((Get-Date) - $lastHeartbeat).TotalMinutes -ge 5) {
            Save-State "waiting_for_dependency" $candidateTable $selection $jobState
            Write-QueueEvent "still waiting for dependency queue $DependencyQueueName"
            $lastHeartbeat = Get-Date
        }
    }
    $dependencyResult = (Get-Content -LiteralPath $DependencySentinel -Raw).Trim()
    Write-QueueEvent "dependency complete: $dependencyResult"

    $candidateTable = foreach ($candidate in $Candidates) {
        $resultsPath = Join-Path $ProjectRoot ("runs\" + $candidate.validation_run + "\results.csv")
        $metric = Get-BestMetric $resultsPath
        [ordered]@{
            name = $candidate.name
            validation_run = $candidate.validation_run
            available = $null -ne $metric
            best_metric = $metric
            gain_over_baseline = if ($null -ne $metric) { $metric.map50_95 - $BaselineMap } else { $null }
        }
    }
    $available = @($candidateTable | Where-Object { $_.available })
    if ($available.Count -eq 0) {
        $completedAt = Get-IsoTimestamp
        Save-State "complete_no_candidate" $candidateTable $selection $jobState
        Set-Content -LiteralPath $CompleteSentinel -Value "$completedAt complete_no_candidate" -Encoding UTF8
        Write-QueueEvent "no validation candidate produced a readable metric; final training not started"
        exit 0
    }

    $bestRow = $available |
        Sort-Object { [double]$_.best_metric.map50_95 } -Descending |
        Select-Object -First 1
    $selectedCandidate = $Candidates |
        Where-Object { $_.name -eq $bestRow.name } |
        Select-Object -First 1
    $selection = [ordered]@{
        name = $selectedCandidate.name
        validation_run = $selectedCandidate.validation_run
        validation_best = $bestRow.best_metric
        gain_over_baseline = $bestRow.gain_over_baseline
        threshold_passed = $bestRow.gain_over_baseline -ge $MinimumGain
        final_name = $selectedCandidate.final_name
        workers = $selectedCandidate.workers
        inference_mode = "standard"
        inference_augment = $false
        inference_multi_scale = @()
        inference_fusion_iou = $null
        inference_comparison = $null
    }
    if (-not $selection.threshold_passed) {
        $completedAt = Get-IsoTimestamp
        Save-State "complete_below_threshold" $candidateTable $selection $jobState
        Set-Content -LiteralPath $CompleteSentinel -Value "$completedAt complete_below_threshold" -Encoding UTF8
        Write-QueueEvent "best candidate $($selection.name) gain=$($selection.gain_over_baseline) below threshold=$MinimumGain; final training not started"
        exit 0
    }

    $validationWeights = Join-Path $ProjectRoot (
        "runs\" + $selectedCandidate.validation_run + "\weights\best.pt"
    )
    $validationDataPath = Join-Path $ProjectRoot $selectedCandidate.validation_data
    $comparisonOutput = Join-Path $QueueRoot (
        $selectedCandidate.name + "_inference_modes.json"
    )
    $comparisonLog = Join-Path $QueueRoot (
        $selectedCandidate.name + "_inference_modes.log"
    )
    $selection.inference_comparison = [ordered]@{
        status = "pending"
        weights = $validationWeights
        data = $validationDataPath
        output = $comparisonOutput
        log = $comparisonLog
        exit_code = $null
        standard_map50_95 = $null
        tta_map50_95 = $null
        tta_gain_map50_95 = $null
        best_alternate_gain_map50_95 = $null
        custom_evaluator_control_passed = $null
        recommended_mode = "standard"
        augment_recommended = $false
        error = $null
    }
    Save-State "comparing_inference_modes" $candidateTable $selection $jobState
    Write-QueueEvent "comparing standard and TTA inference on the selected held-out checkpoint"
    try {
        if (-not (Test-Path -LiteralPath $validationWeights -PathType Leaf)) {
            throw "Selected validation checkpoint was not found: $validationWeights"
        }
        if (-not (Test-Path -LiteralPath $validationDataPath -PathType Leaf)) {
            throw "Selected validation dataset was not found: $validationDataPath"
        }
        $comparisonArguments = @(
            $InferenceComparisonScript,
            "--weights", $validationWeights,
            "--data", $validationDataPath,
            "--output", $comparisonOutput,
            "--name", $selectedCandidate.name,
            "--imgsz", "1024",
            "--batch", "2",
            "--device", "0",
            "--workers", [string]$selectedCandidate.workers,
            "--conf", "0.0001",
            "--iou", "0.7",
            "--max-det", "300",
            "--minimum-gain", "0.001"
        )
        $previousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & $Python @comparisonArguments 2>&1 |
                Tee-Object -FilePath $comparisonLog -Append
            $comparisonExitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $previousErrorActionPreference
        }
        $selection.inference_comparison.exit_code = $comparisonExitCode
        if ($comparisonExitCode -ne 0) {
            throw "Inference-mode comparison failed with exit code $comparisonExitCode"
        }
        $comparisonReport = Get-Content -LiteralPath $comparisonOutput -Raw | ConvertFrom-Json
        $selection.inference_mode = [string]$comparisonReport.recommendation.mode
        $selection.inference_augment = $selection.inference_mode -eq "tta"
        $selection.inference_multi_scale = @($comparisonReport.recommendation.multi_scale)
        $selection.inference_fusion_iou = $comparisonReport.recommendation.fusion_iou
        $selection.inference_comparison.status = "complete"
        $selection.inference_comparison.standard_map50_95 = [double]$comparisonReport.results.standard.map50_95
        $selection.inference_comparison.tta_map50_95 = [double]$comparisonReport.results.tta.map50_95
        $selection.inference_comparison.tta_gain_map50_95 = [double]$comparisonReport.recommendation.tta_gain_map50_95
        $selection.inference_comparison.best_alternate_gain_map50_95 = [double]$comparisonReport.recommendation.gain_map50_95
        $selection.inference_comparison.custom_evaluator_control_passed = [bool]$comparisonReport.custom_evaluator_control.passed
        $selection.inference_comparison.recommended_mode = $selection.inference_mode
        $selection.inference_comparison.augment_recommended = $selection.inference_augment
        Write-QueueEvent "inference comparison complete; mode=$($selection.inference_mode); gain=$($selection.inference_comparison.best_alternate_gain_map50_95)"
    }
    catch {
        $selection.inference_mode = "standard"
        $selection.inference_augment = $false
        $selection.inference_multi_scale = @()
        $selection.inference_fusion_iou = $null
        $selection.inference_comparison.status = "failed_fallback_standard"
        $selection.inference_comparison.error = $_.Exception.Message
        Write-QueueEvent "inference comparison unavailable; falling back to standard inference: $($_.Exception.Message)"
    }
    Save-State "inference_modes_compared" $candidateTable $selection $jobState

    $dataPath = Join-Path $ProjectRoot $selectedCandidate.data
    if (-not (Test-Path -LiteralPath $dataPath -PathType Leaf)) {
        throw "Selected dataset was not found: $dataPath"
    }
    $runDir = Join-Path $ProjectRoot ("runs\" + $selectedCandidate.final_name)
    $resultsPath = Join-Path $runDir "results.csv"
    $lastCheckpoint = Join-Path $runDir "weights\last.pt"
    $logPath = Join-Path $QueueRoot ($selectedCandidate.final_name + ".log")
    $completedEpoch = Get-CompletedEpoch $resultsPath
    $jobState = [ordered]@{
        name = $selectedCandidate.final_name
        candidate = $selectedCandidate.name
        target_epochs = $selectedCandidate.epochs
        workers = $selectedCandidate.workers
        status = "pending"
        started_at = $null
        completed_at = $null
        exit_code = $null
        completed_epoch = $completedEpoch
        checkpoint_for_inference = $lastCheckpoint
        log = $logPath
    }

    if ($completedEpoch -ge $selectedCandidate.epochs) {
        $completedAt = Get-IsoTimestamp
        $jobState.status = "skipped_complete"
        $jobState.completed_at = $completedAt
        $jobState.exit_code = 0
        Save-State "complete" $candidateTable $selection $jobState
        Set-Content -LiteralPath $CompleteSentinel -Value "$completedAt complete" -Encoding UTF8
        Write-QueueEvent "selected full-data run already complete at epoch $completedEpoch; skipped"
        exit 0
    }

    $startedAt = Get-IsoTimestamp
    $jobState.status = "running"
    $jobState.started_at = $startedAt
    Save-State "running" $candidateTable $selection $jobState
    Write-QueueEvent "selected $($selection.name) with gain=$($selection.gain_over_baseline); full-data job $($jobState.name) starting"

    if (Test-Path -LiteralPath $runDir) {
        if (-not (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf)) {
            $completedAt = Get-IsoTimestamp
            $jobState.status = "failed"
            $jobState.exit_code = -2
            $jobState.completed_at = $completedAt
            Save-State "complete_with_failures" $candidateTable $selection $jobState
            Set-Content -LiteralPath $CompleteSentinel -Value "$completedAt complete_with_failures" -Encoding UTF8
            Write-QueueEvent "job failed safely: existing incomplete directory has no last.pt"
            exit 2
        }
        $commandArguments = @(
            $TrainScript,
            "--resume", $lastCheckpoint,
            "--workers", [string]$selectedCandidate.workers
        )
        Write-QueueEvent "job resuming from $lastCheckpoint"
    }
    else {
        $commandArguments = @(
            $TrainScript,
            "--model", $selectedCandidate.model,
            "--data", $dataPath,
            "--epochs", [string]$selectedCandidate.epochs,
            "--imgsz", "1024",
            "--batch", [string]$selectedCandidate.batch,
            "--device", "0",
            "--workers", [string]$selectedCandidate.workers,
            "--seed", "2026",
            "--name", $selectedCandidate.final_name,
            "--no-val",
            "--no-plots"
        ) + $selectedCandidate.arguments
    }

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
    $completedEpoch = Get-CompletedEpoch $resultsPath
    $jobState.exit_code = $exitCode
    $jobState.completed_epoch = $completedEpoch
    $jobState.completed_at = $completedAt
    if ($exitCode -eq 0 -and $completedEpoch -ge $selectedCandidate.epochs) {
        $jobState.status = "complete"
        Save-State "complete" $candidateTable $selection $jobState
        Set-Content -LiteralPath $CompleteSentinel -Value "$completedAt complete" -Encoding UTF8
        Write-QueueEvent "full-data job complete; inference checkpoint=$lastCheckpoint"
    }
    else {
        $jobState.status = "failed"
        Save-State "complete_with_failures" $candidateTable $selection $jobState
        Set-Content -LiteralPath $CompleteSentinel -Value "$completedAt complete_with_failures" -Encoding UTF8
        Write-QueueEvent "full-data job failed; exit_code=$exitCode; completed_epoch=$completedEpoch"
        exit 1
    }
}
catch {
    $completedAt = Get-IsoTimestamp
    if ($null -eq $jobState) {
        $jobState = [ordered]@{ status = "failed"; completed_at = $completedAt; exit_code = -3 }
    }
    else {
        $jobState.status = "failed"
        $jobState.completed_at = $completedAt
        $jobState.exit_code = -3
    }
    Save-State "runner_failed" $candidateTable $selection $jobState
    Write-QueueEvent "queue runner failed: $($_.Exception.Message)"
    throw
}
finally {
    Pop-Location
}
