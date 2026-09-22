from __future__ import annotations

from pathlib import Path

import numpy as np

from hsi_detection.box_calibration import (
    IDENTITY_BOX_CALIBRATION as IDENTITY,
    BoxCalibration as Calibration,
)
from scripts.eval_box_calibration import (
    _calibrate_prediction,
    _candidate_calibrations,
    _select_calibration,
)
from scripts.predict_submission import PredictionArrays


def _prediction(boxes: list[list[float]]) -> PredictionArrays:
    count = len(boxes)
    return PredictionArrays(
        Path("1.npy"),
        np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        np.arange(count, dtype=np.float32),
        np.linspace(0.9, 0.8, count, dtype=np.float32),
        (20, 30),
    )


def test_identity_calibration_preserves_prediction() -> None:
    prediction = _prediction([[2.0, 4.0, 12.0, 14.0]])

    calibrated = _calibrate_prediction(prediction, IDENTITY)

    np.testing.assert_array_equal(calibrated.boxes, prediction.boxes)
    np.testing.assert_array_equal(calibrated.classes, prediction.classes)
    np.testing.assert_array_equal(calibrated.confidences, prediction.confidences)


def test_scale_shift_and_clip_are_applied_in_box_coordinates() -> None:
    prediction = _prediction([[2.0, 4.0, 12.0, 14.0], [20.0, 10.0, 29.0, 19.0]])
    calibration = Calibration(
        width_scale=1.2,
        height_scale=0.8,
        center_x_shift=0.1,
        center_y_shift=-0.1,
    )

    calibrated = _calibrate_prediction(prediction, calibration)

    np.testing.assert_allclose(calibrated.boxes[0], [2.0, 4.0, 14.0, 12.0])
    assert calibrated.boxes[1, 2] == 30.0
    assert np.all(calibrated.boxes[:, (0, 2)] >= 0.0)
    assert np.all(calibrated.boxes[:, (0, 2)] <= 30.0)
    assert np.all(calibrated.boxes[:, (1, 3)] >= 0.0)
    assert np.all(calibrated.boxes[:, (1, 3)] <= 20.0)


def test_candidate_grid_does_not_jointly_fit_scale_and_shift() -> None:
    candidates = _candidate_calibrations((0.99, 1.0, 1.01), (-0.01, 0.0, 0.01))

    assert IDENTITY in candidates
    assert len(candidates) == 17
    assert all(
        (candidate.width_scale == 1.0 and candidate.height_scale == 1.0)
        or (candidate.center_x_shift == 0.0 and candidate.center_y_shift == 0.0)
        for candidate in candidates
    )


def test_selection_falls_back_to_identity_below_fit_gate() -> None:
    candidate = Calibration(width_scale=1.01)
    metrics = {
        IDENTITY: {"map50_95": 0.7000},
        candidate: {"map50_95": 0.7009},
    }

    assert (
        _select_calibration(metrics, identity_map=0.7000, minimum_fit_gain=0.001)
        == IDENTITY
    )
    assert (
        _select_calibration(metrics, identity_map=0.7000, minimum_fit_gain=0.0005)
        == candidate
    )
