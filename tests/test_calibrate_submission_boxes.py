from __future__ import annotations

import json

import pandas as pd
import pytest

from hsi_detection.box_calibration import BoxCalibration
from scripts.calibrate_submission_boxes import (
    calibrate_submission_frame,
    load_audited_calibration,
)


def test_submission_calibration_preserves_non_coordinate_columns_and_clips() -> None:
    source = pd.DataFrame(
        [
            {
                "id": 0,
                "image_id": 14,
                "class_id": 3,
                "confidence": 0.8,
                "x1": 2.0,
                "y1": 4.0,
                "x2": 12.0,
                "y2": 14.0,
            },
            {
                "id": 1,
                "image_id": 14,
                "class_id": 5,
                "confidence": 0.7,
                "x1": 20.0,
                "y1": 10.0,
                "x2": 30.0,
                "y2": 20.0,
            },
        ]
    )

    output = calibrate_submission_frame(
        source,
        {14: (30, 20)},
        BoxCalibration(width_scale=1.2, height_scale=0.8),
    )

    assert output[["id", "image_id", "class_id", "confidence"]].equals(
        source[["id", "image_id", "class_id", "confidence"]]
    )
    assert output.loc[0, ["x1", "y1", "x2", "y2"]].tolist() == [
        1.0,
        5.0,
        13.0,
        13.0,
    ]
    assert output.loc[1, "x2"] == 30.0
    assert output.loc[1, "y2"] == 19.0


def _audit_payload() -> dict[str, object]:
    chosen = {
        "width_scale": 1.01,
        "height_scale": 1.01,
        "center_x_shift": 0.0,
        "center_y_shift": 0.0,
    }
    return {
        "contract": {
            "cache_only": True,
            "single_checkpoint_multiscale": True,
            "global_calibration_only": True,
            "joint_scale_and_shift_search": False,
            "minimum_oof_gain": 0.001,
        },
        "folds": [
            {"chosen": chosen.copy(), "delta_map50_95": delta}
            for delta in (0.0025, 0.0013, 0.0020)
        ],
        "oof": {"delta_map50_95": 0.0018},
        "full_selection": {"chosen": chosen.copy()},
        "passes_stability_gate": True,
    }


def test_load_audited_calibration_requires_unanimous_stable_selection(tmp_path) -> None:
    audit_path = tmp_path / "audit.json"
    payload = _audit_payload()
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    calibration, loaded = load_audited_calibration(audit_path)

    assert calibration == BoxCalibration(width_scale=1.01, height_scale=1.01)
    assert loaded == payload


def test_load_audited_calibration_rejects_fold_disagreement(tmp_path) -> None:
    audit_path = tmp_path / "audit.json"
    payload = _audit_payload()
    payload["folds"][1]["chosen"]["width_scale"] = 1.02
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="unanimously"):
        load_audited_calibration(audit_path)
