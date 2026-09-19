param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$QueueName = "priority_hsi_submission_2026-09-14"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$TrainScript = Join-Path $ProjectRoot "scripts\train_baseline.py"
$CompareScript = Join-Path $ProjectRoot "scripts\compare_inference_modes.py"
$PredictScript = Join-Path $ProjectRoot "scripts\predict_submission.py"
$CheckScript = Join-Path $ProjectRoot "scripts\check_submission.py"
$ValidationData = Join-Path $ProjectRoot "data\processed\hsi16_shared_p005_995\dataset.yaml"
$FullData = Join-Path $ProjectRoot "data\processed\hsi16_shared_p005_995\dataset_all.yaml"
$TestImages = Join-Path $ProjectRoot "data\processed\hsi16_shared_p005_995\images\test"
$PretrainedWeights = Join-Path $ProjectRoot "yolo26s.pt"
$ValidationRun = "ablation_s1024_b4_hsi16_shared_p005_995_r1"
$ValidationWeights = Join-Path $ProjectRoot ("runs\" + $ValidationRun + "\weights\best.pt")
$FinalRun = "final_s1024_all3000_hsi16_shared_p005_995_r1"
$FinalRunDir = Join-Path $ProjectRoot ("runs\" + $FinalRun)
$FinalResults = Join-Path $FinalRunDir "results.csv"
$FinalCheckpoint = Join-Path $FinalRunDir "weights\last.pt"
$QueueRoot = Join-Path $ProjectRoot ("artifacts\background_queue\" + $QueueName)
$StatePath = Join-Path $QueueRoot "state.json"
$EventLog = Join-Path $QueueRoot "events.log"
$CompleteSentinel = Join-Path $QueueRoot "queue.complete"
$ComparisonOutput = Join-Path $QueueRoot "hsi16_inference_modes.json"
$ComparisonLog = Join-Path $QueueRoot "hsi16_inference_modes.log"
$TrainingLog = Join-Path $QueueRoot ($FinalRun + ".log")

foreach ($requiredFile in @(
    $Python,
    $TrainScript,
    $CompareScript,
    $PredictScript,
    $CheckScript,
    $ValidationData,
    $FullData,
    $PretrainedWeights,
    $ValidationWeights
)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required file was not found: $requiredFile"
    }
}
if (-not (Test-Path -LiteralPath $TestImages -PathType Container)) {
    throw "Test image directory was not found: $TestImages"
}
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot ".git"))) {
    throw "Refusing to run outside the expected Git project: $ProjectRoot"
}

New-Item -ItemType Directory -Path $QueueRoot -Force | Out-Null

function Get-IsoTimestamp {
    return (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssK")
}

function Get-Sha256([string]$Path) {
    $stream = [IO.File]::OpenRead($Path)
    try {
        $sha256 = [Security.Cryptography.SHA256]::Create()
        try {
            return ([BitConverter]::ToString($sha256.ComputeHash($stream))).Replace("-", "")
        }
        finally {
            $sha256.Dispose()
        }
    }
    finally {
        $stream.Dispose()
    }
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

function Save-State([string]$QueueStatus, [object]$State) {
    $record = [ordered]@{
        schema_version = 1
        queue_name = $QueueName
        queue_status = $QueueStatus
        runner_pid = $PID
        project_root = $ProjectRoot
        updated_at = Get-IsoTimestamp
        automatic_upload = $false
        work = $State
    }
    $record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $StatePath -Encoding UTF8
}

function Invoke-NativeLogged([string[]]$Arguments, [string]$LogPath) {
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Python @Arguments 2>&1 |
            Tee-Object -FilePath $LogPath -Append |
            ForEach-Object { Write-Host $_ }
        $exitCode = $LASTEXITCODE
        return $exitCode
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
}

if (Test-Path -LiteralPath $CompleteSentinel -PathType Leaf) {
    Write-Output "Priority HSI submission candidate is already complete: $CompleteSentinel"
    exit 0
}

$workState = [ordered]@{
    candidate = "hsi16_shared_p005_995"
    validation_run = $ValidationRun
    validation_checkpoint = $ValidationWeights
    validation_map50_95 = 0.69817
    comparison = [ordered]@{
        status = "pending"
        output = $ComparisonOutput
        log = $ComparisonLog
        mode = "standard"
        multi_scale = @()
        fusion_iou = $null
        gain_map50_95 = 0.0
        exit_code = $null
        error = $null
    }
    training = [ordered]@{
        status = "pending"
        run = $FinalRun
        target_epochs = 30
        completed_epoch = Get-CompletedEpoch $FinalResults
        checkpoint = $FinalCheckpoint
        log = $TrainingLog
        started_at = $null
        completed_at = $null
        exit_code = $null
    }
    submission = [ordered]@{
        status = "pending"
        output = $null
        rows = $null
        images_with_detections = $null
        sha256 = $null
        log = $null
        started_at = $null
        completed_at = $null
        exit_code = $null
    }
}

Push-Location $ProjectRoot
try {
    $env:YOLO_CONFIG_DIR = (Join-Path $ProjectRoot ".ultralytics")
    Save-State "comparing_inference_modes" $workState
    Write-QueueEvent "priority queue started; comparing inference modes for $ValidationRun"

    try {
        $comparisonArguments = @(
            $CompareScript,
            "--weights", $ValidationWeights,
            "--data", $ValidationData,
            "--output", $ComparisonOutput,
            "--name", "hsi16_shared_p005_995_priority",
            "--imgsz", "1024",
            "--batch", "2",
            "--device", "0",
            "--workers", "0",
            "--conf", "0.0001",
            "--iou", "0.7",
            "--max-det", "300",
            "--minimum-gain", "0.001"
        )
        $comparisonExitCode = Invoke-NativeLogged $comparisonArguments $ComparisonLog
        $workState.comparison.exit_code = $comparisonExitCode
        if ($comparisonExitCode -ne 0) {
            throw "Inference comparison failed with exit code $comparisonExitCode"
        }
        $comparisonReport = Get-Content -LiteralPath $ComparisonOutput -Raw | ConvertFrom-Json
        $workState.comparison.status = "complete"
        $workState.comparison.mode = [string]$comparisonReport.recommendation.mode
        $workState.comparison.multi_scale = @($comparisonReport.recommendation.multi_scale)
        $workState.comparison.fusion_iou = $comparisonReport.recommendation.fusion_iou
        $workState.comparison.gain_map50_95 = [double]$comparisonReport.recommendation.gain_map50_95
        Write-QueueEvent "inference comparison complete; mode=$($workState.comparison.mode); gain=$($workState.comparison.gain_map50_95)"
    }
    catch {
        $workState.comparison.status = "failed_fallback_standard"
        $workState.comparison.mode = "standard"
        $workState.comparison.multi_scale = @()
        $workState.comparison.fusion_iou = $null
        $workState.comparison.error = $_.Exception.Message
        Write-QueueEvent "inference comparison unavailable; using standard mode: $($_.Exception.Message)"
    }

    $completedEpoch = Get-CompletedEpoch $FinalResults
    $workState.training.completed_epoch = $completedEpoch
    if ($completedEpoch -lt $workState.training.target_epochs) {
        $workState.training.status = "running"
        $workState.training.started_at = Get-IsoTimestamp
        Save-State "training_full_data" $workState
        if (Test-Path -LiteralPath $FinalRunDir -PathType Container) {
            if (-not (Test-Path -LiteralPath $FinalCheckpoint -PathType Leaf)) {
                throw "Existing incomplete full-data run has no last checkpoint: $FinalRunDir"
            }
            $trainingArguments = @(
                $TrainScript,
                "--resume", $FinalCheckpoint,
                "--workers", "0"
            )
            Write-QueueEvent "resuming full-data HSI training from $FinalCheckpoint"
        }
        else {
            $trainingArguments = @(
                $TrainScript,
                "--model", $PretrainedWeights,
                "--data", $FullData,
                "--epochs", "30",
                "--imgsz", "1024",
                "--batch", "4",
                "--device", "0",
                "--workers", "0",
                "--seed", "2026",
                "--name", $FinalRun,
                "--no-val",
                "--no-plots",
                "--hsv-h", "0",
                "--hsv-s", "0",
                "--hsv-v", "0",
                "--extra-channel-init", "zero"
            )
            Write-QueueEvent "starting full-data HSI training; target_epochs=30; workers=0"
        }
        $trainingExitCode = Invoke-NativeLogged $trainingArguments $TrainingLog
        $workState.training.exit_code = $trainingExitCode
        $workState.training.completed_epoch = Get-CompletedEpoch $FinalResults
        $workState.training.completed_at = Get-IsoTimestamp
        if ($trainingExitCode -ne 0 -or $workState.training.completed_epoch -lt $workState.training.target_epochs) {
            $workState.training.status = "failed"
            throw "Full-data HSI training failed; exit_code=$trainingExitCode; completed_epoch=$($workState.training.completed_epoch)"
        }
        $workState.training.status = "complete"
    }
    else {
        $workState.training.status = "skipped_complete"
        $workState.training.exit_code = 0
        $workState.training.completed_at = Get-IsoTimestamp
        Write-QueueEvent "full-data HSI run already complete at epoch $completedEpoch; skipped training"
    }

    if (-not (Test-Path -LiteralPath $FinalCheckpoint -PathType Leaf)) {
        throw "Full-data inference checkpoint was not found: $FinalCheckpoint"
    }

    $mode = [string]$workState.comparison.mode
    $suffix = if ($mode -eq "tta") {
        "_tta"
    }
    elseif ($mode.StartsWith("multiscale_box_vote")) {
        "_ms" + (($workState.comparison.multi_scale | ForEach-Object { [string]$_ }) -join "-") +
            "_vote" + ([string]$workState.comparison.fusion_iou).Replace(".", "")
    }
    else {
        ""
    }
    $outputName = "submission_" + $FinalRun + "_conf0001" + $suffix + ".csv"
    $outputPath = Join-Path $ProjectRoot ("submissions\" + $outputName)
    $predictionLog = Join-Path $QueueRoot ($outputName + ".log")
    $workState.submission.output = $outputPath
    $workState.submission.log = $predictionLog
    $workState.submission.status = "running"
    $workState.submission.started_at = Get-IsoTimestamp
    Save-State "creating_submission" $workState
    Write-QueueEvent "creating local HSI CSV; mode=$mode; automatic_upload=false"

    $predictionArguments = @(
        $PredictScript,
        "--weights", $FinalCheckpoint,
        "--images", $TestImages,
        "--output", $outputPath,
        "--imgsz", "1024",
        "--batch", "1",
        "--device", "0",
        "--half",
        "--conf", "0.0001",
        "--iou", "0.7",
        "--max-det", "300"
    )
    if ($mode -eq "tta") {
        $predictionArguments += "--augment"
    }
    elseif ($mode.StartsWith("multiscale_box_vote")) {
        if ($workState.comparison.multi_scale.Count -lt 2 -or $null -eq $workState.comparison.fusion_iou) {
            throw "Selected multi-scale inference settings are incomplete"
        }
        $predictionArguments += "--multi-scale"
        $predictionArguments += @($workState.comparison.multi_scale | ForEach-Object { [string]$_ })
        $predictionArguments += @("--fusion-iou", [string]$workState.comparison.fusion_iou)
    }
    $predictionExitCode = Invoke-NativeLogged $predictionArguments $predictionLog
    if ($predictionExitCode -ne 0) {
        $workState.submission.exit_code = $predictionExitCode
        throw "Prediction failed with exit code $predictionExitCode"
    }
    $checkExitCode = Invoke-NativeLogged @($CheckScript, $outputPath, "--images", $TestImages) $predictionLog
    $workState.submission.exit_code = $checkExitCode
    if ($checkExitCode -ne 0) {
        throw "Submission validation failed with exit code $checkExitCode"
    }

    $frame = @(Import-Csv -LiteralPath $outputPath)
    $workState.submission.status = "complete"
    $workState.submission.rows = $frame.Count
    $workState.submission.images_with_detections = @($frame.image_id | Sort-Object -Unique).Count
    $workState.submission.sha256 = Get-Sha256 $outputPath
    $workState.submission.completed_at = Get-IsoTimestamp
    Save-State "complete" $workState
    Set-Content -LiteralPath $CompleteSentinel -Value "$(Get-IsoTimestamp) complete" -Encoding UTF8
    Write-QueueEvent "validated priority CSV ready: $outputPath; rows=$($workState.submission.rows); sha256=$($workState.submission.sha256); not uploaded"
}
catch {
    if ($workState.training.status -eq "running") {
        $workState.training.status = "failed"
        $workState.training.completed_epoch = Get-CompletedEpoch $FinalResults
        $workState.training.completed_at = Get-IsoTimestamp
    }
    if ($workState.submission.status -eq "running") {
        $workState.submission.status = "failed"
        $workState.submission.completed_at = Get-IsoTimestamp
    }
    Save-State "runner_failed" $workState
    Write-QueueEvent "priority queue failed: $($_.Exception.Message)"
    throw
}
finally {
    Pop-Location
}
