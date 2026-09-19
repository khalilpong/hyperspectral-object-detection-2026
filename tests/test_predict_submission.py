from pathlib import Path

import numpy as np
import pytest

from scripts.predict_submission import (
    _batches,
    _box_vote,
    _normalize_multiscale,
    _npy_prediction_batches,
    _numeric_paths,
)


def test_numeric_paths_sort_by_image_id(tmp_path: Path) -> None:
    for name in ("20.npy", "3.npy", "11.npy"):
        (tmp_path / name).touch()

    assert [path.stem for path in _numeric_paths(tmp_path, ".npy")] == ["3", "11", "20"]


def test_batches_reject_non_positive_size() -> None:
    with pytest.raises(ValueError, match="--batch must be positive"):
        list(_batches([Path("1.npy")], 0))


def test_npy_prediction_batches_preserve_paths_and_bound_batch(tmp_path: Path) -> None:
    paths = [tmp_path / f"{image_id}.npy" for image_id in (3, 11, 20)]
    for path in paths:
        np.save(path, np.zeros((5, 7, 16), dtype=np.uint8), allow_pickle=False)

    class FakeModel:
        batch_sizes: list[int] = []

        def predict(self, source: list[np.ndarray], **_: object) -> list[object]:
            self.batch_sizes.append(len(source))
            return [object() for _ in source]

    model = FakeModel()
    items = list(
        _npy_prediction_batches(
            model,  # type: ignore[arg-type]
            paths,
            channels=16,
            batch_size=2,
            predict_kwargs={},
        )
    )

    assert [path for path, _ in items] == paths
    assert model.batch_sizes == [2, 1]


def test_npy_prediction_batches_reject_wrong_channels(tmp_path: Path) -> None:
    path = tmp_path / "3.npy"
    np.save(path, np.zeros((5, 7, 3), dtype=np.uint8), allow_pickle=False)

    with pytest.raises(ValueError, match="Expected H x W x 16"):
        list(
            _npy_prediction_batches(
                object(),  # type: ignore[arg-type]
                [path],
                channels=16,
                batch_size=1,
                predict_kwargs={},
            )
        )


def test_box_vote_averages_same_class_boxes_from_different_scales() -> None:
    boxes = np.asarray(
        [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [20.0, 20.0, 30.0, 30.0]],
        dtype=np.float32,
    )
    classes = np.asarray([2, 2, 2], dtype=np.float32)
    confidences = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)

    fused_boxes, fused_classes, fused_confidences = _box_vote(
        boxes,
        classes,
        confidences,
        iou_threshold=0.5,
        max_det=10,
        source_ids=np.asarray([0, 1, 1]),
    )

    assert fused_boxes.shape == (2, 4)
    assert fused_boxes[0] == pytest.approx([0.470588, 0.470588, 10.470588, 10.470588])
    assert fused_classes.tolist() == [2.0, 2.0]
    assert fused_confidences.tolist() == pytest.approx([0.9, 0.7])


def test_box_vote_never_merges_different_classes() -> None:
    boxes = np.asarray([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
    classes = np.asarray([1, 2], dtype=np.float32)
    confidences = np.asarray([0.9, 0.8], dtype=np.float32)

    fused_boxes, fused_classes, _ = _box_vote(
        boxes,
        classes,
        confidences,
        iou_threshold=0.5,
        max_det=10,
        source_ids=np.asarray([0, 1]),
    )

    assert len(fused_boxes) == 2
    assert fused_classes.tolist() == [1.0, 2.0]


def test_normalize_multiscale_requires_distinct_positive_sizes() -> None:
    assert _normalize_multiscale([960, 1024, 960, 1088]) == (960, 1024, 1088)
    with pytest.raises(ValueError, match="at least two distinct"):
        _normalize_multiscale([1024, 1024])
    with pytest.raises(ValueError, match="must be positive"):
        _normalize_multiscale([1024, 0])
