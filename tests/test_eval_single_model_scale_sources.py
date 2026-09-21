from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.eval_single_model_scale_sources import (
    EXPECTED_SCALES,
    SourceSpec,
    _fuse_cached_sources,
    _source_specs,
)
from scripts.predict_submission import PredictionArrays


def test_source_specs_are_fixed_and_cover_leave_one_out() -> None:
    specs = _source_specs(EXPECTED_SCALES)

    assert specs[0].name == "all7_control"
    assert len(specs) == 12
    assert {spec.name for spec in specs[1:8]} == {
        f"drop_{scale}" for scale in EXPECTED_SCALES
    }
    assert next(spec for spec in specs if spec.name == "center3").scales == (
        960,
        1024,
        1088,
    )
    assert next(spec for spec in specs if spec.name == "center_weighted").confidence_weights == (
        0.9,
        0.95,
        1.0,
        1.1,
        1.0,
        0.95,
        0.9,
    )


def test_source_specs_reject_unexpected_cache_scales() -> None:
    with pytest.raises(ValueError, match="exact seven-scale cache"):
        _source_specs((960, 1024, 1088))


def test_scale_confidence_weight_changes_vote_without_second_model() -> None:
    path = Path("3.npy")
    low_scale = PredictionArrays(
        path,
        np.asarray([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
        np.asarray([0.8], dtype=np.float32),
        (20, 20),
    )
    high_scale = PredictionArrays(
        path,
        np.asarray([[2.0, 0.0, 12.0, 10.0]], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
        np.asarray([0.8], dtype=np.float32),
        (20, 20),
    )
    passes = {832: {"3": low_scale}, 896: {"3": high_scale}}
    uniform = _fuse_cached_sources(
        passes,
        [path],
        spec=SourceSpec("uniform", (832, 896), (1.0, 1.0)),
        fusion_iou=0.5,
        support_gain=0.0,
        max_det=10,
    )[0]
    weighted = _fuse_cached_sources(
        passes,
        [path],
        spec=SourceSpec("weighted", (832, 896), (2.0, 1.0)),
        fusion_iou=0.5,
        support_gain=0.0,
        max_det=10,
    )[0]

    assert uniform.boxes[0, 0] == pytest.approx(1.0)
    assert weighted.boxes[0, 0] == pytest.approx(2.0 / 3.0)
    assert weighted.confidences[0] == pytest.approx(1.6)
