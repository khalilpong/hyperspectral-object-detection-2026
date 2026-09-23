from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.eval_max_det_oof import (
    INCUMBENT_MAX_DET,
    _candidate_name,
    _prediction_count_summary,
    _select_max_det,
)
from scripts.predict_submission import PredictionArrays


def test_candidate_name_is_explicit() -> None:
    assert _candidate_name(400) == "max_det_400"


def test_selection_falls_back_to_incumbent_below_fit_gate() -> None:
    metrics = {
        200: {"map50_95": 0.6990},
        INCUMBENT_MAX_DET: {"map50_95": 0.7000},
        400: {"map50_95": 0.7009},
        500: {"map50_95": 0.7008},
    }

    assert (
        _select_max_det(
            metrics,
            incumbent_max_det=INCUMBENT_MAX_DET,
            minimum_fit_gain=0.001,
        )
        == INCUMBENT_MAX_DET
    )


def test_selection_accepts_best_alternative_after_gate() -> None:
    metrics = {
        200: {"map50_95": 0.6990},
        INCUMBENT_MAX_DET: {"map50_95": 0.7000},
        400: {"map50_95": 0.7012},
        500: {"map50_95": 0.7011},
    }

    assert (
        _select_max_det(
            metrics,
            incumbent_max_det=INCUMBENT_MAX_DET,
            minimum_fit_gain=0.001,
        )
        == 400
    )


def test_selection_prefers_closest_cap_on_exact_tie() -> None:
    metrics = {
        200: {"map50_95": 0.7012},
        INCUMBENT_MAX_DET: {"map50_95": 0.7000},
        400: {"map50_95": 0.7012},
        500: {"map50_95": 0.7012},
    }

    assert (
        _select_max_det(
            metrics,
            incumbent_max_det=INCUMBENT_MAX_DET,
            minimum_fit_gain=0.001,
        )
        == 200
    )


def test_prediction_count_summary_reports_observed_minimum() -> None:
    predictions = [
        PredictionArrays(
            Path(f"{index}.npy"),
            np.zeros((count, 4), dtype=np.float32),
            np.zeros(count, dtype=np.float32),
            np.zeros(count, dtype=np.float32),
            (10, 10),
        )
        for index, count in enumerate((3, 5, 5))
    ]

    assert _prediction_count_summary(predictions, 5) == {
        "total_predictions": 13,
        "minimum_per_image": 3,
        "median_per_image": 5.0,
        "maximum_per_image": 5,
        "images_at_limit": 2,
    }
