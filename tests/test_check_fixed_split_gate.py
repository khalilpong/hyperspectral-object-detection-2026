from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.check_fixed_split_gate import audit_fixed_split


def _write_run(
    tmp_path: Path,
    *,
    dfl: float,
    best: float,
    architecture: str = "yolo",
    model: str = "yolo26m.pt",
    degrees: float | None = None,
    extra_channel_init: str = "random",
    num_denoising: int | None = None,
    box_iou_loss: str | None = None,
) -> tuple[Path, Path]:
    status = {
        "config": {
            "MODE": "ablation",
            "ARCHITECTURE": architecture,
            "MODEL": model,
            "EPOCHS": 30,
            "RUN_NAME": f"dfl{dfl}",
            "MULTISCALE": False,
            "DATA": "hsi16",
            "SEED": 2026,
            "EXTRA_CHANNEL_INIT": extra_channel_init,
            "SPECTRAL_STEM": False,
            "LOWER_PERCENTILE": 0.5,
            "UPPER_PERCENTILE": 99.5,
            "CLS_PW": 0.0,
            "SCALE": 0.5,
            "DFL": dfl,
            "PHASE_TARGET_LONG_EDGE": 1024,
            "OBJECT_CROPS": False,
            "TILE_INFERENCE": False,
            "IMGSZ": 1024,
        },
        "steps": {
            "train_attempt_b8_w2_d0": {
                "returncode": 0,
                "cuda_oom": False,
                "shm_error": False,
            }
        },
        "result": "success",
    }
    if degrees is not None:
        status["config"]["DEGREES"] = degrees
    if num_denoising is not None:
        status["config"]["RTDETR_NUM_DENOISING"] = num_denoising
    if box_iou_loss is not None:
        status["config"]["BOX_IOU_LOSS"] = box_iou_loss
    status_path = tmp_path / "status.json"
    status_path.write_text(json.dumps(status), encoding="utf-8")

    results_path = tmp_path / "results.csv"
    with results_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "metrics/mAP50(B)", "metrics/mAP50-95(B)"],
        )
        writer.writeheader()
        for epoch in range(1, 31):
            writer.writerow(
                {
                    "epoch": epoch,
                    "metrics/mAP50(B)": 0.95,
                    "metrics/mAP50-95(B)": best if epoch == 29 else 0.70,
                }
            )
    return status_path, results_path


def test_audit_fixed_split_applies_gate_and_records_hashes(tmp_path: Path) -> None:
    status, results = _write_run(tmp_path, dfl=2.0, best=0.705)

    report = audit_fixed_split(
        status_path=status,
        results_path=results,
        expected_dfl=2.0,
        gate=0.70443,
    )

    assert report["best_epoch"] == 29
    assert report["passes_full_data_gate"] is True
    assert report["decision"] == "eligible_for_single_checkpoint_full_data"
    assert len(report["hashes"]["status_sha256"]) == 64


def test_audit_fixed_split_rejects_failed_gate(tmp_path: Path) -> None:
    status, results = _write_run(tmp_path, dfl=2.5, best=0.703)

    report = audit_fixed_split(
        status_path=status,
        results_path=results,
        expected_dfl=2.5,
        gate=0.70443,
    )

    assert report["passes_full_data_gate"] is False
    assert report["decision"] == "reject_no_full_data"


def test_audit_fixed_split_rejects_config_drift(tmp_path: Path) -> None:
    status, results = _write_run(tmp_path, dfl=2.0, best=0.705)

    with pytest.raises(ValueError, match="config mismatch"):
        audit_fixed_split(
            status_path=status,
            results_path=results,
            expected_dfl=2.5,
            gate=0.70443,
        )


def test_audit_fixed_split_checks_yolo_degrees_variant(tmp_path: Path) -> None:
    status, results = _write_run(
        tmp_path, dfl=1.5, best=0.705, degrees=5.0, num_denoising=100
    )

    report = audit_fixed_split(
        status_path=status,
        results_path=results,
        expected_dfl=1.5,
        expected_degrees=5.0,
        gate=0.70443,
    )

    assert report["contract"]["expected_degrees"] == 5.0
    assert report["passes_full_data_gate"] is True

    with pytest.raises(ValueError, match="config mismatch"):
        audit_fixed_split(
            status_path=status,
            results_path=results,
            expected_dfl=1.5,
            expected_degrees=3.0,
            gate=0.70443,
        )


def test_audit_fixed_split_checks_yolo_eiou_variant(tmp_path: Path) -> None:
    status, results = _write_run(
        tmp_path,
        dfl=1.5,
        best=0.705,
        box_iou_loss="eiou",
    )

    report = audit_fixed_split(
        status_path=status,
        results_path=results,
        expected_dfl=1.5,
        expected_box_iou_loss="eiou",
        gate=0.70443,
    )

    assert report["contract"]["expected_box_iou_loss"] == "eiou"
    assert report["passes_full_data_gate"] is True

    with pytest.raises(ValueError, match="config mismatch"):
        audit_fixed_split(
            status_path=status,
            results_path=results,
            expected_dfl=1.5,
            gate=0.70443,
        )


def test_audit_fixed_split_accepts_rtdetr_contract(tmp_path: Path) -> None:
    status, results = _write_run(
        tmp_path,
        dfl=1.5,
        best=0.705,
        architecture="rtdetr",
        model="rtdetr-l.pt",
    )

    report = audit_fixed_split(
        status_path=status,
        results_path=results,
        expected_architecture="rtdetr",
        expected_model="rtdetr-l.pt",
        gate=0.70443,
    )

    assert report["contract"]["architecture"] == "rtdetr"
    assert report["contract"]["model"] == "rtdetr-l.pt"
    assert "expected_dfl" not in report["contract"]
    assert report["passes_full_data_gate"] is True


def test_audit_fixed_split_checks_rtdetr_denoising_variant(tmp_path: Path) -> None:
    status, results = _write_run(
        tmp_path,
        dfl=1.5,
        best=0.705,
        architecture="rtdetr",
        model="rtdetr-l.pt",
        degrees=0.0,
        num_denoising=200,
    )

    report = audit_fixed_split(
        status_path=status,
        results_path=results,
        expected_architecture="rtdetr",
        expected_model="rtdetr-l.pt",
        expected_rtdetr_num_denoising=200,
        gate=0.70443,
    )

    assert report["contract"]["expected_rtdetr_num_denoising"] == 200
    assert report["passes_full_data_gate"] is True

    with pytest.raises(ValueError, match="config mismatch"):
        audit_fixed_split(
            status_path=status,
            results_path=results,
            expected_architecture="rtdetr",
            expected_model="rtdetr-l.pt",
            expected_rtdetr_num_denoising=100,
            gate=0.70443,
        )


def test_audit_fixed_split_checks_extra_channel_init_variant(tmp_path: Path) -> None:
    status, results = _write_run(
        tmp_path,
        dfl=1.5,
        best=0.705,
        architecture="rtdetr",
        model="rtdetr-l.pt",
        extra_channel_init="zero",
    )

    report = audit_fixed_split(
        status_path=status,
        results_path=results,
        expected_architecture="rtdetr",
        expected_model="rtdetr-l.pt",
        expected_extra_channel_init="zero",
        gate=0.70443,
    )

    assert report["contract"]["expected_extra_channel_init"] == "zero"
    assert report["passes_full_data_gate"] is True

    with pytest.raises(ValueError, match="config mismatch"):
        audit_fixed_split(
            status_path=status,
            results_path=results,
            expected_architecture="rtdetr",
            expected_model="rtdetr-l.pt",
            gate=0.70443,
        )


def test_audit_fixed_split_rejects_yolo_dfl_contract_for_rtdetr(
    tmp_path: Path,
) -> None:
    status, results = _write_run(
        tmp_path,
        dfl=1.5,
        best=0.705,
        architecture="rtdetr",
        model="rtdetr-l.pt",
    )

    with pytest.raises(ValueError, match="does not use"):
        audit_fixed_split(
            status_path=status,
            results_path=results,
            expected_architecture="rtdetr",
            expected_dfl=1.5,
            gate=0.70443,
        )
