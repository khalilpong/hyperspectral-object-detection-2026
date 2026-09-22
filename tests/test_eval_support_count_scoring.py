from __future__ import annotations

import numpy as np

from scripts.eval_support_count_scoring import (
    INCUMBENT_NAME,
    _count_candidate_name,
    _select_candidate,
    _support_count_box_vote,
)
from scripts.predict_submission import _box_vote


def test_zero_count_gain_matches_production_max_confidence_vote() -> None:
    boxes = np.asarray(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 0.0, 11.0, 10.0],
            [30.0, 30.0, 40.0, 40.0],
        ],
        dtype=np.float32,
    )
    classes = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    confidences = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)
    source_ids = np.asarray([0, 1, 0], dtype=np.int64)

    expected = _box_vote(
        boxes,
        classes,
        confidences,
        iou_threshold=0.5,
        max_det=300,
        source_ids=source_ids,
        support_gain=0.0,
        total_sources=3,
    )
    observed = _support_count_box_vote(
        boxes,
        classes,
        confidences,
        iou_threshold=0.5,
        max_det=300,
        source_ids=source_ids,
        count_gain=0.0,
        total_sources=3,
    )

    for observed_array, expected_array in zip(observed, expected, strict=True):
        np.testing.assert_array_equal(observed_array, expected_array)


def test_count_gain_rewards_distinct_sources_only() -> None:
    boxes = np.asarray(
        [
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 20.0, 30.0, 30.0],
        ],
        dtype=np.float32,
    )
    classes = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    confidences = np.asarray([0.60, 0.55, 0.61], dtype=np.float32)
    source_ids = np.asarray([0, 1, 0], dtype=np.int64)

    _, _, scores = _support_count_box_vote(
        boxes,
        classes,
        confidences,
        iou_threshold=0.5,
        max_det=300,
        source_ids=source_ids,
        count_gain=0.06,
        total_sources=3,
    )

    np.testing.assert_allclose(scores, [0.63, 0.61], rtol=0.0, atol=1e-7)


def test_selection_requires_gain_over_incumbent() -> None:
    gains = (0.0, 0.01, 0.02)
    metrics = {
        INCUMBENT_NAME: {"map50_95": 0.7000},
        _count_candidate_name(0.0): {"map50_95": 0.6990},
        _count_candidate_name(0.01): {"map50_95": 0.7009},
        _count_candidate_name(0.02): {"map50_95": 0.7011},
    }

    assert (
        _select_candidate(
            metrics,
            incumbent_name=INCUMBENT_NAME,
            count_gains=gains,
            minimum_fit_gain=0.0012,
        )
        == INCUMBENT_NAME
    )
    assert (
        _select_candidate(
            metrics,
            incumbent_name=INCUMBENT_NAME,
            count_gains=gains,
            minimum_fit_gain=0.001,
        )
        == _count_candidate_name(0.02)
    )
