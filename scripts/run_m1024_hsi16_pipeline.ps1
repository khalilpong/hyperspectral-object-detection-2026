param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$QueueName = "m1024_hsi16_submission_2026-09-14",
    [int]$TargetEpochs = 30,
    [int]$Batch = 2,
    [float]$Conf = 0.0001,
    [float]$Iou = 0.70
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$TrainScript = Join-Path $ProjectRoot "scripts\train_baseline.py"
$PredictScript = Join-Path $ProjectRoot "scripts\predict_submission.py"
$CheckScript = Join-Path $ProjectRoot "scripts\check_submission.py"
$FullData = Join-Path $ProjectRoot "data\processed\hsi16_shared_p005_995\dataset_all.yaml"
$TestImages = Join-Path $ProjectRoot "data\processed\hsi16_shared_p005_995\images\test"
$PretrainedWeights = Join-Path $ProjectRoot "yolo26m.pt"
$FinalRun = "final_m1024_all3000_hsi16_shared_p005_995_r1"
$FinalRunDir = Join-Path $ProjectRoot ("runs\" + $FinalRun)
$FinalResults = Join-Path $FinalRunDir "results.csv"
$FinalCheckpoint = Join-Path $FinalRunDir "weights\last.pt"
$QueueRoot = Join-Path $ProjectRoot ("artifacts\background_queue\" + $QueueName)
$StatePath = Join-Path $QueueRoot "state.json"
$EventLog = Join-Path $QueueRoot "events.log"
$CompleteSentinel = Join-Path $QueueRoot "queue.complete"
$TrainingLog = Join-Path $QueueRoot ($FinalRun + ".log")
$OutputName = "submission_" + $FinalRun + "_conf0001.csv"
$OutputPath = Join-Path $ProjectRoot ("submissions\" + $OutputName)
$PredictionLog = Join-Path $QueueRoot ($OutputName + ".log")

foreach ($requiredFile in @(
    $Python,
    $TrainScript,
    $PredictScript,
    $CheckScript,
    $FullData,
    $PretrainedWeights
)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required file was not found: $requiredFile"
    }
}
if (-not (Test-Path -LiteralPath $TestImages -PathType Container)) {
    throw "Test image directory was not found: $TestImages"
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
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    return $exitCode
}

$workState = [ordered]@{
    model = "yolo26m.pt"
    dataset = "hsi16_shared_p005_995"
    training = [ordered]@{
        status = "pending"
        run = $FinalRun
        target_epochs = $TargetEpochs
        batch = $Batch
        completed_epoch = Get-CompletedEpoch $FinalResults
        checkpoint = $FinalCheckpoint
        log = $TrainingLog
        started_at = $null
        completed_at = $null
        exit_code = $null
    }
    submission = [ordered]@{
        status = "pending"
        output = $OutputPath
        rows = $null
        images_with_detections = $null
        sha256 = $null
        log = $PredictionLog
        started_at = $null
        completed_at = $null
        exit_code = $null
        kaggle_ref = $null
        kaggle_score = $null
    }
}

Push-Location $ProjectRoot
try {
    $env:YOLO_CONFIG_DIR = (Join-Path $ProjectRoot ".ultralytics")
    Write-QueueEvent "starting YOLO26m HSI16 pipeline; target_epochs=$TargetEpochs; batch=$Batch"
    
    $completedEpoch = Get-CompletedEpoch $FinalResults
    $workState.training.completed_epoch = $completedEpoch
    if ($completedEpoch -lt $TargetEpochs) {
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
            Write-QueueEvent "resuming full-data YOLO26m HSI training from $FinalCheckpoint"
        }
        else {
            $trainingArguments = @(
                $TrainScript,
                "--model", $PretrainedWeights,
                "--data", $FullData,
                "--epochs", [string]$TargetEpochs,
                "--imgsz", "1024",
                "--batch", [string]$Batch,
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
            Write-QueueEvent "starting fresh full-data YOLO26m HSI training; target_epochs=$TargetEpochs; batch=$Batch; workers=0"
        }
        $trainingExitCode = Invoke-NativeLogged $trainingArguments $TrainingLog
        $workState.training.exit_code = $trainingExitCode
        $workState.training.completed_epoch = Get-CompletedEpoch $FinalResults
        $workState.training.completed_at = Get-IsoTimestamp
        if ($trainingExitCode -ne 0 -or $workState.training.completed_epoch -lt $TargetEpochs) {
            throw "YOLO26m full-data training exited with code $trainingExitCode; completed_epoch=$($workState.training.completed_epoch)"
        }
        $workState.training.status = "complete"
        Write-QueueEvent "YOLO26m full-data training finished successfully at epoch $($workState.training.completed_epoch)"
    }
    else {
        $workState.training.status = "skipped_complete"
        $workState.training.completed_at = Get-IsoTimestamp
        Write-QueueEvent "YOLO26m full-data training already complete at epoch $completedEpoch; skipping to prediction"
    }
    Save-State "training_complete" $workState

    if (-not (Test-Path -LiteralPath $FinalCheckpoint -PathType Leaf)) {
        throw "Final checkpoint was not found: $FinalCheckpoint"
    }

    $workState.submission.status = "running"
    $workState.submission.started_at = Get-IsoTimestamp
    Save-State "creating_submission" $workState
    Write-QueueEvent "generating test predictions with YOLO26m checkpoint: $FinalCheckpoint"

    $predictionArguments = @(
        $PredictScript,
        "--weights", $FinalCheckpoint,
        "--images", $TestImages,
        "--output", $OutputPath,
        "--imgsz", "1024",
        "--batch", "1",
        "--device", "0",
        "--half",
        "--conf", [string]$Conf,
        "--iou", [string]$Iou,
        "--max-det", "300"
    )
    $predictionExitCode = Invoke-NativeLogged $predictionArguments $PredictionLog
    if ($predictionExitCode -ne 0) {
        $workState.submission.exit_code = $predictionExitCode
        throw "Prediction failed with exit code $predictionExitCode"
    }

    $checkExitCode = Invoke-NativeLogged @($CheckScript, $OutputPath, "--images", $TestImages) $PredictionLog
    $workState.submission.exit_code = $checkExitCode
    if ($checkExitCode -ne 0) {
        throw "Submission validation failed with exit code $checkExitCode"
    }

    $frame = @(Import-Csv -LiteralPath $OutputPath)
    $workState.submission.status = "validated"
    $workState.submission.rows = $frame.Count
    $workState.submission.images_with_detections = @($frame.image_id | Sort-Object -Unique).Count
    $workState.submission.sha256 = Get-Sha256 $OutputPath
    $workState.submission.completed_at = Get-IsoTimestamp
    Save-State "validated_ready_to_upload" $workState
    Write-QueueEvent "validated YOLO26m CSV ready: $OutputPath; rows=$($workState.submission.rows); sha256=$($workState.submission.sha256)"

    Write-QueueEvent "submitting to Kaggle competition: hyperspectral-object-detection-challenge-2026"
    $submitOutput = kaggle competitions submit -c hyperspectral-object-detection-challenge-2026 -f $OutputPath -m "YOLO26m HSI16 shared p005_995 all3000 epoch$TargetEpochs imgsz1024 conf$Conf iou$Iou" 2>&1
    Write-QueueEvent "Kaggle submit output: $submitOutput"
    Add-Content -LiteralPath $PredictionLog -Value "`n--- Kaggle Submit Output ---`n$submitOutput" -Encoding UTF8

    Start-Sleep -Seconds 30
    $submissions = kaggle competitions submissions hyperspectral-object-detection-challenge-2026 2>&1
    Write-QueueEvent "Kaggle submissions status: $submissions"

    $workState.submission.status = "complete"
    Save-State "complete" $workState
    Set-Content -LiteralPath $CompleteSentinel -Value "$(Get-IsoTimestamp) complete" -Encoding UTF8
    Write-QueueEvent "pipeline complete"
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
    Write-QueueEvent "pipeline failed: $($_.Exception.Message)"
    throw
}
finally {
    Pop-Location
}
