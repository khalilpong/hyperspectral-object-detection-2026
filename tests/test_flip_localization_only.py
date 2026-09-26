from pathlib import Path

import numpy as np
import pytest

from scripts.eval_flip_localization_only import refine_identity_geometry
from scripts.predict_submission import PredictionArrays


def record(boxes, classes, confidence):
    return PredictionArrays(Path("1.npy"), np.asarray(boxes, np.float32).reshape(-1, 4),
                            np.asarray(classes, np.float32), np.asarray(confidence, np.float32), (100, 100))


def test_coordinate_only_vote_keeps_classes_scores_and_count():
    identity = record([[0, 0, 10, 10]], [2], [0.8])
    flip = record([[1, 0, 11, 10], [50, 50, 60, 60]], [2, 2], [0.8, 0.9])
    result, matches = refine_identity_geometry(identity, flip)
    assert matches == 1
    np.testing.assert_allclose(result.boxes, [[0.125, 0, 10.125, 10]])
    np.testing.assert_array_equal(result.classes, identity.classes)
    np.testing.assert_array_equal(result.confidences, identity.confidences)
    np.testing.assert_array_equal(identity.boxes, [[0, 0, 10, 10]])


def test_same_flip_cannot_refine_two_detections_or_wrong_class():
    identity = record([[0, 0, 10, 10], [1, 0, 11, 10], [0, 0, 10, 10]], [2, 2, 3], [0.8] * 3)
    flip = record([[1, 0, 11, 10]], [2], [0.8])
    result, matches = refine_identity_geometry(identity, flip)
    assert matches == 1
    np.testing.assert_array_equal(result.boxes, identity.boxes)


def test_empty_flip_and_mismatched_image():
    identity = record([[0, 0, 10, 10]], [2], [0.8])
    result, matches = refine_identity_geometry(identity, record([], [], []))
    assert matches == 0
    np.testing.assert_array_equal(result.boxes, identity.boxes)
    other = record([], [], [])
    other.path = Path("2.npy")
    with pytest.raises(ValueError, match="must match"):
        refine_identity_geometry(identity, other)
