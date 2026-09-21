from __future__ import annotations

import numpy as np

from scripts.eval_box_vote_geometry import _variant_box_vote
from scripts.predict_submission import _box_vote


def _vote(
    boxes: np.ndarray,
    confidences: np.ndarray,
    *,
    cluster_mode: str,
    coordinate_mode: str,
    threshold: float = 0.65,
):
    return _variant_box_vote(
        boxes,
        np.zeros(len(boxes), dtype=np.float32),
        confidences,
        iou_threshold=threshold,
        max_det=20,
        source_ids=np.arange(len(boxes), dtype=np.int64),
        support_gain=0.125,
        total_sources=len(boxes),
        cluster_mode=cluster_mode,
        coordinate_mode=coordinate_mode,
    )


def test_centroid_mean_matches_production_box_vote() -> None:
    boxes = np.asarray(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 0.0, 11.0, 10.0],
            [20.0, 20.0, 30.0, 30.0],
        ],
        dtype=np.float32,
    )
    classes = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    confidences = np.asarray([0.9, 0.7, 0.8], dtype=np.float32)
    source_ids = np.asarray([0, 1, 0], dtype=np.int64)

    expected = _box_vote(
        boxes,
        classes,
        confidences,
        iou_threshold=0.65,
        max_det=20,
        source_ids=source_ids,
        support_gain=0.125,
        total_sources=2,
    )
    actual = _variant_box_vote(
        boxes,
        classes,
        confidences,
        iou_threshold=0.65,
        max_det=20,
        source_ids=source_ids,
        support_gain=0.125,
        total_sources=2,
        cluster_mode="centroid",
        coordinate_mode="mean",
    )

    for expected_array, actual_array in zip(expected, actual, strict=True):
        np.testing.assert_allclose(actual_array, expected_array, rtol=0.0, atol=1e-6)


def test_anchor_mode_prevents_centroid_chain_merge() -> None:
    boxes = np.asarray(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.5, 0.0, 11.5, 10.0],
            [3.0, 0.0, 13.0, 10.0],
        ],
        dtype=np.float32,
    )
    confidences = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)

    centroid = _vote(
        boxes,
        confidences,
        cluster_mode="centroid",
        coordinate_mode="mean",
        threshold=0.62,
    )
    anchor = _vote(
        boxes,
        confidences,
        cluster_mode="anchor",
        coordinate_mode="mean",
        threshold=0.62,
    )

    assert len(centroid[0]) == 1
    assert len(anchor[0]) == 2


def test_top2_coordinates_ignore_third_member_but_keep_support_score() -> None:
    boxes = np.asarray(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 0.0, 11.0, 10.0],
            [2.0, 0.0, 12.0, 10.0],
        ],
        dtype=np.float32,
    )
    confidences = np.asarray([0.9, 0.8, 0.1], dtype=np.float32)
    top2 = _vote(
        boxes,
        confidences,
        cluster_mode="centroid",
        coordinate_mode="top2",
        threshold=0.5,
    )

    expected_x1 = (0.0 * 0.9 + 1.0 * 0.8) / (0.9 + 0.8)
    assert len(top2[0]) == 1
    assert np.isclose(top2[0][0, 0], expected_x1)
    expected_score = 0.9 + 0.125 * (0.8 + 0.1) / 3
    assert np.isclose(top2[2][0], expected_score)
