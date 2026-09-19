param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$DependencyQueueName = "final_candidate_followup_v2_2026-09-13",
    [string]$QueueName = "submission_followup_2026-09-13"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$PredictScript = Join-Path $ProjectRoot "scripts\predict_submission.py"
$CheckScript = Join-Path $ProjectRoot "scripts\check_submission.py"
$DependencyRoot = Join-Path $ProjectRoot ("artifacts\background_queue\" + $DependencyQueueName)
$DependencySentinel = Join-Path $DependencyRoot "queue.complete"
$DependencyState = Join-Path $DependencyRoot "state.json"
$QueueRoot = Join-Path $ProjectRoot ("artifacts\background_queue\" + $QueueName)
$StatePath = Join-Path $QueueRoot "state.json"
$EventLog = Join-Path $QueueRoot "events.log"
$CompleteSentinel = Join-Path $QueueRoot "queue.complete"

foreach ($requiredFile in @($Python, $PredictScript, $CheckScript)) {
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

function Save-State([string]$QueueStatus, [object]$SubmissionState) {
    $state = [ordered]@{
        schema_version = 1
        queue_name = $QueueName
        queue_status = $QueueStatus
        runner_pid = $PID
        project_root = $ProjectRoot
        dependency_queue = $DependencyQueueName
        dependency_sentinel = $DependencySentinel
        updated_at = Get-IsoTimestamp
        submission = $SubmissionState
        automatic_upload = $false
    }
    $state | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $StatePath -Encoding UTF8
}

if (Test-Path -LiteralPath $CompleteSentinel -PathType Leaf) {
    Write-Output "Submission follow-up queue is already complete: $CompleteSentinel"
    exit 0
}

$submissionState = $null
Push-Location $ProjectRoot
try {
    $env:YOLO_CONFIG_DIR = (Join-Path $ProjectRoot ".ultralytics")
    Save-State "waiting_for_dependency" $submissionState
    Write-QueueEvent "queue started; runner_pid=$PID; waiting_for=$DependencyQueueName"

    $lastHeartbeat = Get-Date
    while (-not (Test-Path -LiteralPath $DependencySentinel -PathType Leaf)) {
        Start-Sleep -Seconds 30
        if (((Get-Date) - $lastHeartbeat).TotalMinutes -ge 5) {
            Save-State "waiting_for_dependency" $submissionState
            Write-QueueEvent "still waiting for dependency queue $DependencyQueueName"
            $lastHeartbeat = Get-Date
        }
    }
    $dependencyResult = (Get-Content -LiteralPath $DependencySentinel -Raw).Trim()
    Write-QueueEvent "dependency complete: $dependencyResult"

    if (-not (Test-Path -LiteralPath $DependencyState -PathType Leaf)) {
        throw "Final-candidate state was not found: $DependencyState"
    }
    $finalState = Get-Content -LiteralPath $DependencyState -Raw | ConvertFrom-Json
    $successfulJob = $null -ne $finalState.job -and $finalState.job.status -in @("complete", "skipped_complete")
    if ($finalState.queue_status -ne "complete" -or -not $successfulJob) {
        $completedAt = Get-IsoTimestamp
        $submissionState = [ordered]@{
            status = "not_created"
            reason = "final candidate queue did not produce a complete full-data checkpoint"
            dependency_queue_status = $finalState.queue_status
            completed_at = $completedAt
        }
        Save-State "complete_no_submission" $submissionState
        Set-Content -LiteralPath $CompleteSentinel -Value "$completedAt complete_no_submission" -Encoding UTF8
        Write-QueueEvent "no submission created because final queue status was $($finalState.queue_status)"
        exit 0
    }

    $checkpoint = [IO.Path]::GetFullPath([string]$finalState.job.checkpoint_for_inference)
    if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
        throw "Selected inference checkpoint was not found: $checkpoint"
    }
    $candidate = [string]$finalState.selection.name
    $images = if ($candidate -eq "hsi16_shared_p005_995") {
        Join-Path $ProjectRoot "data\processed\hsi16_shared_p005_995\images\test"
    }
    else {
        Join-Path $ProjectRoot "data\processed\pseudo_rgb\images\test"
    }
    if (-not (Test-Path -LiteralPath $images -PathType Container)) {
        throw "Selected test image directory was not found: $images"
    }

    $inferenceMode = if ($null -ne $finalState.selection.PSObject.Properties["inference_mode"]) {
        [string]$finalState.selection.inference_mode
    }
    elseif (
        $null -ne $finalState.selection.PSObject.Properties["inference_augment"] -and
        [bool]$finalState.selection.inference_augment
    ) {
        "tta"
    }
    else {
        "standard"
    }
    if ($inferenceMode -notin @("standard", "tta") -and -not $inferenceMode.StartsWith("multiscale_box_vote")) {
        throw "Unsupported selected inference mode: $inferenceMode"
    }
    $inferenceAugment = $inferenceMode -eq "tta"
    $inferenceScales = @($finalState.selection.inference_multi_scale | ForEach-Object { [int]$_ })
    $fusionIou = $finalState.selection.inference_fusion_iou
    $inferenceSuffix = if ($inferenceAugment) {
        "_tta"
    }
    elseif ($inferenceMode.StartsWith("multiscale_box_vote")) {
        "_ms" + (($inferenceScales | ForEach-Object { [string]$_ }) -join "-") +
            "_vote" + ([string]$fusionIou).Replace(".", "")
    }
    else {
        ""
    }
    $outputName = ([string]$finalState.selection.final_name) + "_conf0001" + $inferenceSuffix + ".csv"
    $outputPath = Join-Path $ProjectRoot ("submissions\" + $outputName)
    $logPath = Join-Path $QueueRoot ($outputName + ".log")
    $startedAt = Get-IsoTimestamp
    $submissionState = [ordered]@{
        status = "running"
        candidate = $candidate
        checkpoint = $checkpoint
        images = $images
        output = $outputPath
        inference = [ordered]@{
            imgsz = 1024
            batch = 1
            device = 0
            half = $true
            conf = 0.0001
            iou = 0.7
            max_det = 300
            mode = $inferenceMode
            augment = $inferenceAugment
            multi_scale = $inferenceScales
            fusion_iou = $fusionIou
        }
        started_at = $startedAt
        completed_at = $null
        exit_code = $null
        rows = $null
        images_with_detections = $null
        sha256 = $null
        log = $logPath
    }
    Save-State "running" $submissionState
    Write-QueueEvent "creating local CSV for candidate=$candidate; checkpoint=$checkpoint; inference_mode=$inferenceMode; automatic_upload=false"

    $predictionArguments = @(
        $PredictScript,
        "--weights", $checkpoint,
        "--images", $images,
        "--output", $outputPath,
        "--imgsz", "1024",
        "--batch", "1",
        "--device", "0",
        "--half",
        "--conf", "0.0001",
        "--iou", "0.7",
        "--max-det", "300"
    )
    if ($inferenceAugment) {
        $predictionArguments += "--augment"
    }
    elseif ($inferenceMode.StartsWith("multiscale_box_vote")) {
        if ($inferenceScales.Count -lt 2 -or $null -eq $fusionIou) {
            throw "Selected multi-scale inference settings are incomplete"
        }
        $predictionArguments += "--multi-scale"
        $predictionArguments += @($inferenceScales | ForEach-Object { [string]$_ })
        $predictionArguments += @("--fusion-iou", [string]$fusionIou)
    }
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Python @predictionArguments 2>&1 |
            Tee-Object -FilePath $logPath -Append
        $predictExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($predictExitCode -ne 0) {
        throw "Prediction failed with exit code $predictExitCode"
    }

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Python $CheckScript $outputPath --images $images 2>&1 |
            Tee-Object -FilePath $logPath -Append
        $checkExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($checkExitCode -ne 0) {
        throw "Submission validation failed with exit code $checkExitCode"
    }

    $frame = @(Import-Csv -LiteralPath $outputPath)
    $hash = (Get-FileHash -LiteralPath $outputPath -Algorithm SHA256).Hash
    $completedAt = Get-IsoTimestamp
    $submissionState.status = "complete"
    $submissionState.completed_at = $completedAt
    $submissionState.exit_code = 0
    $submissionState.rows = $frame.Count
    $submissionState.images_with_detections = @($frame.image_id | Sort-Object -Unique).Count
    $submissionState.sha256 = $hash
    Save-State "complete" $submissionState
    Set-Content -LiteralPath $CompleteSentinel -Value "$completedAt complete" -Encoding UTF8
    Write-QueueEvent "validated CSV ready at $outputPath; rows=$($frame.Count); sha256=$hash; not uploaded"
}
catch {
    $completedAt = Get-IsoTimestamp
    if ($null -eq $submissionState) {
        $submissionState = [ordered]@{ status = "failed"; completed_at = $completedAt; exit_code = -3 }
    }
    else {
        $submissionState.status = "failed"
        $submissionState.completed_at = $completedAt
        $submissionState.exit_code = -3
    }
    Save-State "runner_failed" $submissionState
    Write-QueueEvent "queue runner failed: $($_.Exception.Message)"
    throw
}
finally {
    Pop-Location
}
