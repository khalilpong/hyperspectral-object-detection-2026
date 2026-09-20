from pathlib import Path

import numpy as np
import pytest

from scripts.predict_submission import (
    PredictionArrays,
    _batches,
    _box_vote,
    _collect_tiled_predictions,
    _fused_full_and_tiled_predictions,
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


def test_box_vote_support_gain_rewards_independent_agreement() -> None:
    boxes = np.asarray(
        [[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]],
        dtype=np.float32,
    )
    classes = np.asarray([2, 2, 2], dtype=np.float32)
    confidences = np.asarray([0.6, 0.6, 0.65], dtype=np.float32)

    fused_boxes, _, fused_confidences = _box_vote(
        boxes,
        classes,
        confidences,
        iou_threshold=0.5,
        max_det=10,
        source_ids=np.asarray([0, 1, 2]),
        support_gain=0.5,
        total_sources=3,
    )

    assert fused_boxes[0] == pytest.approx([0.0, 0.0, 10.0, 10.0])
    assert fused_confidences.tolist() == pytest.approx([0.7, 0.65])


def test_box_vote_rejects_invalid_support_configuration() -> None:
    boxes = np.asarray([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
    classes = np.asarray([2], dtype=np.float32)
    confidences = np.asarray([0.6], dtype=np.float32)

    with pytest.raises(ValueError, match="support_gain"):
        _box_vote(
            boxes,
            classes,
            confidences,
            iou_threshold=0.5,
            max_det=10,
            support_gain=-0.1,
        )
    with pytest.raises(ValueError, match="total_sources"):
        _box_vote(
            np.repeat(boxes, 2, axis=0),
            np.repeat(classes, 2),
            np.repeat(confidences, 2),
            iou_threshold=0.5,
            max_det=10,
            source_ids=np.asarray([0, 1]),
            support_gain=0.1,
            total_sources=1,
        )


def test_normalize_multiscale_requires_distinct_positive_sizes() -> None:
    assert _normalize_multiscale([960, 1024, 960, 1088]) == (960, 1024, 1088)
    with pytest.raises(ValueError, match="at least two distinct"):
        _normalize_multiscale([1024, 1024])
    with pytest.raises(ValueError, match="must be positive"):
        _normalize_multiscale([1024, 0])


def test_collect_tiled_predictions_maps_core_owned_boxes(tmp_path: Path) -> None:
    path = tmp_path / "3.npy"
    np.save(path, np.zeros((100, 180, 16), dtype=np.uint8), allow_pickle=False)

    class FakeTensor:
        def __init__(self, values: np.ndarray) -> None:
            self.values = values

        def detach(self) -> "FakeTensor":
            return self

        def cpu(self) -> "FakeTensor":
            return self

        def numpy(self) -> np.ndarray:
            return self.values

    class FakeBoxes:
        def __init__(self, height: int, width: int) -> None:
            self.xyxy = FakeTensor(
                np.asarray([[width / 2 - 5, height / 2 - 5, width / 2 + 5, height / 2 + 5]])
            )
            self.cls = FakeTensor(np.asarray([2.0], dtype=np.float32))
            self.conf = FakeTensor(np.asarray([0.8], dtype=np.float32))

    class FakeResult:
        def __init__(self, array: np.ndarray) -> None:
            self.orig_shape = array.shape[:2]
            self.boxes = FakeBoxes(*array.shape[:2])

    class FakeModel:
        def predict(self, source: list[np.ndarray], **_: object) -> list[FakeResult]:
            return [FakeResult(array) for array in source]

    predictions = _collect_tiled_predictions(
        FakeModel(),  # type: ignore[arg-type]
        [path],
        channels=16,
        batch_size=2,
        tile_height=80,
        tile_width=100,
        stride_height=60,
        stride_width=80,
        tile_imgsz=128,
        predict_kwargs={},
    )

    assert len(predictions[path]) == 4
    assert all(record.orig_shape == (100, 180) for record in predictions[path])
    assert sum(len(record.boxes) for record in predictions[path]) == 4


def test_fuse_full_and_tile_predictions_uses_one_checkpoint_sources() -> None:
    path = Path("3.npy")
    full = PredictionArrays(
        path=path,
        boxes=np.asarray([[10, 10, 20, 20]], dtype=np.float32),
        classes=np.asarray([1], dtype=np.float32),
        confidences=np.asarray([0.6], dtype=np.float32),
        orig_shape=(100, 180),
    )
    tile = PredictionArrays(
        path=path,
        boxes=np.asarray([[11, 11, 21, 21]], dtype=np.float32),
        classes=np.asarray([1], dtype=np.float32),
        confidences=np.asarray([0.8], dtype=np.float32),
        orig_shape=(100, 180),
    )

    fused = list(
        _fused_full_and_tiled_predictions(
            {1024: {path: full}},
            {path: [tile]},
            [path],
            fusion_iou=0.5,
            max_det=10,
        )
    )

    assert len(fused) == 1
    assert fused[0].boxes.shape == (1, 4)
    assert fused[0].boxes[0] == pytest.approx(
        [10.571428, 10.571428, 20.571428, 20.571428]
    )
    assert fused[0].confidences.tolist() == pytest.approx([0.8])
