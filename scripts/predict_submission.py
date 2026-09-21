from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterator, Sequence, TypeVar

import numpy as np
import pandas as pd

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".ultralytics").resolve()))

from ultralytics import YOLO

from hsi_detection.submission import clip_xyxy
from hsi_detection.tiling import map_tile_boxes_to_image, tile_windows


_T = TypeVar("_T")


@dataclass
class PredictionArrays:
    path: Path
    boxes: np.ndarray
    classes: np.ndarray
    confidences: np.ndarray
    orig_shape: tuple[int, int]


def _numeric_paths(directory: Path, suffix: str) -> list[Path]:
    paths = list(directory.glob(f"*{suffix}"))
    try:
        return sorted(paths, key=lambda path: int(path.stem))
    except ValueError as error:
        raise ValueError(
            f"Every {suffix} input filename must have a numeric stem in {directory.resolve()}"
        ) from error


def _batches(paths: Sequence[_T], batch_size: int) -> Iterator[Sequence[_T]]:
    if batch_size <= 0:
        raise ValueError("--batch must be positive")
    for start in range(0, len(paths), batch_size):
        yield paths[start : start + batch_size]


def _checkpoint_channels(model: YOLO) -> int:
    channels = getattr(model.model, "yaml", {}).get("channels", 3)
    try:
        channels = int(channels)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Checkpoint declares an invalid input channel count: {channels!r}") from error
    if channels <= 0:
        raise ValueError(f"Checkpoint declares an invalid input channel count: {channels}")
    return channels


def _npy_prediction_batches(
    model: YOLO,
    image_paths: Sequence[Path],
    *,
    channels: int,
    batch_size: int,
    predict_kwargs: dict[str, object],
) -> Iterator[tuple[Path, object]]:
    """Predict 16-channel arrays in bounded batches while retaining their image IDs.

    Ultralytics' path-based prediction loader decodes the PNG preview and does
    not consult the sibling ``.npy`` file used by its training dataset. Passing
    NumPy arrays directly does preserve all checkpoint input channels, but a
    list is treated as one in-memory batch. Chunking here keeps test inference
    within the requested memory bound.
    """
    for path_batch in _batches(image_paths, batch_size):
        arrays: list[np.ndarray] = []
        for path in path_batch:
            array = np.load(path, allow_pickle=False)
            if array.ndim != 3 or array.shape[2] != channels:
                raise ValueError(
                    f"Expected H x W x {channels} input at {path.resolve()}, got {array.shape}"
                )
            if array.dtype != np.uint8:
                raise ValueError(
                    f"Expected uint8 input at {path.resolve()}, got {array.dtype}"
                )
            arrays.append(array)
        results = model.predict(source=arrays, stream=False, **predict_kwargs)
        if len(results) != len(path_batch):
            raise RuntimeError(
                f"Expected {len(path_batch)} prediction results, received {len(results)}"
            )
        yield from zip(path_batch, results, strict=True)


def _png_prediction_batches(
    model: YOLO,
    image_paths: Sequence[Path],
    *,
    image_directory: Path,
    predict_kwargs: dict[str, object],
) -> Iterator[tuple[Path, object]]:
    expected_paths = {path.resolve(): path for path in image_paths}
    results = model.predict(
        # A Python list of paths is treated by Ultralytics as one in-memory
        # image batch, bypassing --batch and causing an OOM on large datasets.
        source=str(image_directory.resolve() / "*.png"),
        stream=True,
        **predict_kwargs,
    )
    seen_paths: set[Path] = set()
    for result in results:
        resolved_result_path = Path(result.path).resolve()
        if resolved_result_path not in expected_paths:
            raise RuntimeError(f"Unexpected prediction result: {resolved_result_path}")
        if resolved_result_path in seen_paths:
            raise RuntimeError(f"Duplicate prediction result: {resolved_result_path}")
        seen_paths.add(resolved_result_path)
        yield expected_paths[resolved_result_path], result
    missing_paths = set(expected_paths) - seen_paths
    if missing_paths:
        preview = ", ".join(str(path) for path in sorted(missing_paths)[:5])
        raise RuntimeError(f"Missing predictions for {len(missing_paths)} images: {preview}")


def _prediction_arrays(
    model: YOLO,
    image_paths: Sequence[Path],
    *,
    image_directory: Path,
    input_format: str,
    channels: int,
    batch_size: int,
    predict_kwargs: dict[str, object],
) -> Iterator[PredictionArrays]:
    if input_format == "npy":
        result_items = _npy_prediction_batches(
            model,
            image_paths,
            channels=channels,
            batch_size=batch_size,
            predict_kwargs=predict_kwargs,
        )
    elif input_format == "png":
        result_items = _png_prediction_batches(
            model,
            image_paths,
            image_directory=image_directory,
            predict_kwargs=predict_kwargs,
        )
    else:
        raise ValueError(f"Unsupported input format: {input_format}")

    for path, result in result_items:
        height, width = result.orig_shape
        if result.boxes is None:
            boxes = np.empty((0, 4), dtype=np.float32)
            classes = np.empty(0, dtype=np.float32)
            confidences = np.empty(0, dtype=np.float32)
        else:
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy()
            confidences = result.boxes.conf.detach().cpu().numpy()
        yield PredictionArrays(
            path=path,
            boxes=boxes,
            classes=classes,
            confidences=confidences,
            orig_shape=(int(height), int(width)),
        )


def _horizontal_flip_xyxy(boxes: np.ndarray, width: int) -> np.ndarray:
    """Map XYXY boxes between an image and its horizontal mirror.

    Ultralytics returns boxes in the original array's continuous pixel-boundary
    coordinates, so the inverse transform is ``x1 = width - x2`` and
    ``x2 = width - x1``.  Using ``width - 1`` here would introduce a one-pixel
    shift.
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1:] != (4,):
        raise ValueError(f"Expected boxes with shape N x 4, got {boxes.shape}")
    if width <= 0:
        raise ValueError("Image width must be positive")
    flipped = boxes.copy()
    flipped[:, 0] = width - boxes[:, 2]
    flipped[:, 2] = width - boxes[:, 0]
    return flipped


def _collect_horizontal_flip_predictions(
    model: YOLO,
    image_paths: Sequence[Path],
    *,
    channels: int,
    batch_size: int,
    predict_kwargs: dict[str, object],
) -> dict[Path, PredictionArrays]:
    """Predict horizontally mirrored NPY inputs and map boxes back to source coordinates."""
    predictions: dict[Path, PredictionArrays] = {}
    flip_kwargs = {**predict_kwargs, "augment": False}
    for path_batch in _batches(image_paths, batch_size):
        source_shapes: list[tuple[int, int]] = []
        flipped_arrays: list[np.ndarray] = []
        for path in path_batch:
            array = np.load(path, allow_pickle=False)
            if array.ndim != 3 or array.shape[2] != channels:
                raise ValueError(
                    f"Expected H x W x {channels} input at {path.resolve()}, got {array.shape}"
                )
            if array.dtype != np.uint8:
                raise ValueError(f"Expected uint8 input at {path.resolve()}, got {array.dtype}")
            source_shapes.append(tuple(map(int, array.shape[:2])))
            # Slicing creates a negative-stride view, which model.predict cannot
            # safely convert to a tensor.  Materialize a contiguous HWC array.
            flipped_arrays.append(np.ascontiguousarray(array[:, ::-1, :]))

        results = model.predict(source=flipped_arrays, stream=False, **flip_kwargs)
        if len(results) != len(path_batch):
            raise RuntimeError(
                f"Expected {len(path_batch)} horizontal-flip results, received {len(results)}"
            )
        for path, source_shape, result in zip(path_batch, source_shapes, results, strict=True):
            result_shape = tuple(map(int, result.orig_shape))
            if result_shape != source_shape:
                raise RuntimeError(
                    f"Horizontal-flip result shape {result_shape} does not match "
                    f"source shape {source_shape} for {path}"
                )
            if result.boxes is None:
                boxes = np.empty((0, 4), dtype=np.float32)
                classes = np.empty(0, dtype=np.float32)
                confidences = np.empty(0, dtype=np.float32)
            else:
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                classes = result.boxes.cls.detach().cpu().numpy()
                confidences = result.boxes.conf.detach().cpu().numpy()
            if path in predictions:
                raise RuntimeError(f"Duplicate horizontal-flip prediction for {path}")
            predictions[path] = PredictionArrays(
                path=path,
                boxes=_horizontal_flip_xyxy(boxes, source_shape[1]),
                classes=classes,
                confidences=confidences,
                orig_shape=source_shape,
            )
    if len(predictions) != len(image_paths):
        raise RuntimeError(
            f"Horizontal-flip pass returned {len(predictions)} of {len(image_paths)} images"
        )
    return predictions


def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    top_left = np.maximum(box[:2], boxes[:, :2])
    bottom_right = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.prod(np.maximum(bottom_right - top_left, 0.0), axis=1)
    box_area = max(float((box[2] - box[0]) * (box[3] - box[1])), 0.0)
    box_areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(
        boxes[:, 3] - boxes[:, 1], 0.0
    )
    return intersection / np.maximum(box_area + box_areas - intersection, 1e-12)


def _box_vote(
    boxes: np.ndarray,
    classes: np.ndarray,
    confidences: np.ndarray,
    *,
    iou_threshold: float | Mapping[int, float],
    max_det: int,
    source_ids: np.ndarray | None = None,
    support_gain: float = 0.0,
    total_sources: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse same-class boxes with confidence-weighted coordinate voting.

    ``source_ids`` prevents two detections from the same scale from entering one
    cluster. This retains each scale's own NMS decisions while allowing one
    checkpoint's repeated multi-scale observations to tighten coordinates.

    ``support_gain`` optionally adds a small ranking bonus for confidence mass
    contributed by sources beyond the cluster's strongest source. The bonus is
    normalized by ``total_sources`` and leaves the legacy max-confidence score
    byte-for-byte unchanged when it is zero.
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    classes = np.asarray(classes, dtype=np.float32)
    confidences = np.asarray(confidences, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1:] != (4,):
        raise ValueError(f"Expected boxes with shape N x 4, got {boxes.shape}")
    if len(classes) != len(boxes) or len(confidences) != len(boxes):
        raise ValueError("Box, class, and confidence arrays must have equal lengths")
    if isinstance(iou_threshold, Mapping):
        class_iou_thresholds = {
            int(class_id): float(threshold)
            for class_id, threshold in iou_threshold.items()
        }
        if any(
            not np.isfinite(threshold) or not 0.0 < threshold <= 1.0
            for threshold in class_iou_thresholds.values()
        ):
            raise ValueError("Every class fusion IoU must be finite and in (0, 1]")
    else:
        if not np.isfinite(iou_threshold) or not 0.0 < iou_threshold <= 1.0:
            raise ValueError("Fusion IoU must be finite and in (0, 1]")
        class_iou_thresholds = None
    if max_det <= 0:
        raise ValueError("max_det must be positive")
    if not np.isfinite(support_gain) or support_gain < 0.0:
        raise ValueError("support_gain must be finite and non-negative")
    if source_ids is None:
        source_ids = np.arange(len(boxes), dtype=np.int64)
    else:
        source_ids = np.asarray(source_ids, dtype=np.int64)
        if len(source_ids) != len(boxes):
            raise ValueError("source_ids must have one entry per box")
    if np.any(source_ids < 0):
        raise ValueError("source_ids must be non-negative")

    valid = (
        np.isfinite(boxes).all(axis=1)
        & np.isfinite(classes)
        & np.isfinite(confidences)
        & (boxes[:, 2] > boxes[:, 0])
        & (boxes[:, 3] > boxes[:, 1])
        & (confidences >= 0.0)
    )
    boxes = boxes[valid]
    classes = classes[valid]
    confidences = confidences[valid]
    source_ids = source_ids[valid]
    if not len(boxes):
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )
    observed_sources = len(np.unique(source_ids))
    if total_sources is None:
        total_sources = observed_sources
    if total_sources < observed_sources or total_sources <= 0:
        raise ValueError("total_sources must cover every observed source")

    fused_boxes: list[np.ndarray] = []
    fused_classes: list[float] = []
    fused_confidences: list[float] = []
    for class_id in np.unique(classes):
        if class_iou_thresholds is None:
            class_iou_threshold = float(iou_threshold)
        else:
            integer_class_id = int(class_id)
            if integer_class_id not in class_iou_thresholds:
                raise ValueError(
                    f"Missing fusion IoU for observed class {integer_class_id}"
                )
            class_iou_threshold = class_iou_thresholds[integer_class_id]
        class_indices = np.flatnonzero(classes == class_id)
        class_indices = class_indices[np.argsort(confidences[class_indices])[::-1]]
        cluster_boxes: list[np.ndarray] = []
        weighted_sums: list[np.ndarray] = []
        weight_sums: list[float] = []
        cluster_confidences: list[float] = []
        cluster_confidence_sums: list[float] = []
        cluster_sources: list[set[int]] = []
        for index in class_indices:
            box = boxes[index]
            confidence = float(confidences[index])
            source_id = int(source_ids[index])
            eligible = [
                cluster_index
                for cluster_index, sources in enumerate(cluster_sources)
                if source_id not in sources
            ]
            best_cluster: int | None = None
            if eligible:
                eligible_boxes = np.stack([cluster_boxes[i] for i in eligible])
                ious = _iou_one_to_many(box, eligible_boxes)
                best_position = int(np.argmax(ious))
                if float(ious[best_position]) >= class_iou_threshold:
                    best_cluster = eligible[best_position]

            weight = max(confidence, 1e-8)
            if best_cluster is None:
                cluster_boxes.append(box.copy())
                weighted_sums.append(box.astype(np.float64) * weight)
                weight_sums.append(weight)
                cluster_confidences.append(confidence)
                cluster_confidence_sums.append(confidence)
                cluster_sources.append({source_id})
            else:
                weighted_sums[best_cluster] += box * weight
                weight_sums[best_cluster] += weight
                cluster_boxes[best_cluster] = (
                    weighted_sums[best_cluster] / weight_sums[best_cluster]
                ).astype(np.float32)
                cluster_confidences[best_cluster] = max(
                    cluster_confidences[best_cluster], confidence
                )
                cluster_confidence_sums[best_cluster] += confidence
                cluster_sources[best_cluster].add(source_id)

        fused_boxes.extend(cluster_boxes)
        fused_classes.extend([float(class_id)] * len(cluster_boxes))
        if support_gain == 0.0:
            fused_confidences.extend(cluster_confidences)
        else:
            fused_confidences.extend(
                min(
                    1.0,
                    maximum
                    + support_gain * max(total - maximum, 0.0) / total_sources,
                )
                for maximum, total in zip(
                    cluster_confidences, cluster_confidence_sums, strict=True
                )
            )

    output_boxes = np.asarray(fused_boxes, dtype=np.float32)
    output_classes = np.asarray(fused_classes, dtype=np.float32)
    output_confidences = np.asarray(fused_confidences, dtype=np.float32)
    order = np.argsort(output_confidences)[::-1][:max_det]
    return output_boxes[order], output_classes[order], output_confidences[order]


def _normalize_multiscale(scales: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(dict.fromkeys(int(scale) for scale in scales))
    if len(normalized) < 2:
        raise ValueError("--multi-scale requires at least two distinct image sizes")
    if any(scale <= 0 for scale in normalized):
        raise ValueError("Every --multi-scale image size must be positive")
    return normalized


def _collect_multiscale_predictions(
    model: YOLO,
    image_paths: Sequence[Path],
    *,
    image_directory: Path,
    input_format: str,
    channels: int,
    batch_size: int,
    scales: Sequence[int],
    predict_kwargs: dict[str, object],
) -> dict[int, dict[Path, PredictionArrays]]:
    predictions_by_scale: dict[int, dict[Path, PredictionArrays]] = {}
    expected_paths = set(image_paths)
    for scale in _normalize_multiscale(scales):
        scale_kwargs = {**predict_kwargs, "imgsz": scale, "augment": False}
        records = {
            record.path: record
            for record in _prediction_arrays(
                model,
                image_paths,
                image_directory=image_directory,
                input_format=input_format,
                channels=channels,
                batch_size=batch_size,
                predict_kwargs=scale_kwargs,
            )
        }
        if set(records) != expected_paths:
            raise RuntimeError(
                f"Multi-scale pass {scale} returned {len(records)} of {len(expected_paths)} images"
            )
        predictions_by_scale[scale] = records
    return predictions_by_scale


def _collect_tiled_predictions(
    model: YOLO,
    image_paths: Sequence[Path],
    *,
    channels: int,
    batch_size: int,
    tile_height: int,
    tile_width: int,
    stride_height: int,
    stride_width: int,
    tile_imgsz: int,
    predict_kwargs: dict[str, object],
) -> dict[Path, list[PredictionArrays]]:
    """Run one checkpoint on overlapping HSI tiles and map core-owned boxes to full images."""
    if batch_size <= 0:
        raise ValueError("--batch must be positive")
    predictions: dict[Path, list[PredictionArrays]] = {}
    tile_kwargs = {**predict_kwargs, "imgsz": tile_imgsz, "augment": False}
    legacy_half = tile_kwargs.pop("half", None)
    if legacy_half and "quantize" not in tile_kwargs:
        tile_kwargs["quantize"] = 16
    for image_index, path in enumerate(image_paths, start=1):
        array = np.load(path, allow_pickle=False)
        if array.ndim != 3 or array.shape[2] != channels:
            raise ValueError(
                f"Expected H x W x {channels} input at {path.resolve()}, got {array.shape}"
            )
        if array.dtype != np.uint8:
            raise ValueError(f"Expected uint8 input at {path.resolve()}, got {array.dtype}")
        image_height, image_width = map(int, array.shape[:2])
        windows = tile_windows(
            image_height,
            image_width,
            tile_height=tile_height,
            tile_width=tile_width,
            stride_height=stride_height,
            stride_width=stride_width,
        )
        records: list[PredictionArrays] = []
        for window_batch in _batches(windows, batch_size):
            arrays = [
                np.ascontiguousarray(array[window.y0 : window.y1, window.x0 : window.x1])
                for window in window_batch
            ]
            results = model.predict(source=arrays, stream=False, **tile_kwargs)
            if len(results) != len(window_batch):
                raise RuntimeError(
                    f"Expected {len(window_batch)} tile results for {path}, received {len(results)}"
                )
            for window, result in zip(window_batch, results, strict=True):
                result_shape = tuple(map(int, result.orig_shape))
                if result_shape != (window.height, window.width):
                    raise RuntimeError(
                        f"Tile result shape {result_shape} does not match "
                        f"{(window.height, window.width)} for {path}"
                    )
                if result.boxes is None:
                    local_boxes = np.empty((0, 4), dtype=np.float32)
                    classes = np.empty(0, dtype=np.float32)
                    confidences = np.empty(0, dtype=np.float32)
                else:
                    local_boxes = result.boxes.xyxy.detach().cpu().numpy()
                    classes = result.boxes.cls.detach().cpu().numpy()
                    confidences = result.boxes.conf.detach().cpu().numpy()
                mapped_boxes, keep = map_tile_boxes_to_image(local_boxes, window)
                records.append(
                    PredictionArrays(
                        path=path,
                        boxes=mapped_boxes[keep],
                        classes=classes[keep],
                        confidences=confidences[keep],
                        orig_shape=(image_height, image_width),
                    )
                )
        predictions[path] = records
        if image_index % 25 == 0 or image_index == len(image_paths):
            print(
                f"tile inference: {image_index}/{len(image_paths)} images, "
                f"{sum(len(items) for items in predictions.values())} tile sources",
                flush=True,
            )
    return predictions


def _fused_full_and_tiled_predictions(
    predictions_by_scale: dict[int, dict[Path, PredictionArrays]],
    tiled_predictions: dict[Path, list[PredictionArrays]],
    image_paths: Sequence[Path],
    *,
    fusion_iou: float,
    max_det: int,
    support_gain: float = 0.0,
) -> Iterator[PredictionArrays]:
    """Fuse full-image scales and core-filtered tiles from the same checkpoint."""
    scales = tuple(predictions_by_scale)
    if not scales:
        raise ValueError("At least one full-image prediction pass is required")
    for path in image_paths:
        full_records = [predictions_by_scale[scale][path] for scale in scales]
        tile_records = tiled_predictions[path]
        records = [*full_records, *tile_records]
        if not tile_records:
            raise RuntimeError(f"No tile predictions were produced for {path}")
        if len({record.orig_shape for record in records}) != 1:
            raise RuntimeError(f"Inconsistent original shapes across full/tile passes for {path}")
        boxes = np.concatenate([record.boxes for record in records], axis=0)
        classes = np.concatenate([record.classes for record in records], axis=0)
        confidences = np.concatenate([record.confidences for record in records], axis=0)
        source_ids = np.concatenate(
            [
                np.full(len(record.boxes), source_index, dtype=np.int64)
                for source_index, record in enumerate(records)
            ]
        )
        fused_boxes, fused_classes, fused_confidences = _box_vote(
            boxes,
            classes,
            confidences,
            iou_threshold=fusion_iou,
            max_det=max_det,
            source_ids=source_ids,
            support_gain=support_gain,
            total_sources=len(records),
        )
        yield PredictionArrays(
            path=path,
            boxes=fused_boxes,
            classes=fused_classes,
            confidences=fused_confidences,
            orig_shape=records[0].orig_shape,
        )


def _fused_prediction_sources(
    prediction_sources: Sequence[dict[Path, PredictionArrays]],
    image_paths: Sequence[Path],
    *,
    fusion_iou: float,
    max_det: int,
    support_gain: float = 0.0,
) -> Iterator[PredictionArrays]:
    """Fuse two or more transformed views produced by one checkpoint."""
    if len(prediction_sources) < 2:
        raise ValueError("At least two prediction sources are required for fusion")
    for path in image_paths:
        try:
            records = [source[path] for source in prediction_sources]
        except KeyError as error:
            raise RuntimeError(f"Prediction source is missing {path}") from error
        if len({record.orig_shape for record in records}) != 1:
            raise RuntimeError(f"Inconsistent original shapes across prediction sources for {path}")
        boxes = np.concatenate([record.boxes for record in records], axis=0)
        classes = np.concatenate([record.classes for record in records], axis=0)
        confidences = np.concatenate([record.confidences for record in records], axis=0)
        source_ids = np.concatenate(
            [
                np.full(len(record.boxes), source_index, dtype=np.int64)
                for source_index, record in enumerate(records)
            ]
        )
        fused_boxes, fused_classes, fused_confidences = _box_vote(
            boxes,
            classes,
            confidences,
            iou_threshold=fusion_iou,
            max_det=max_det,
            source_ids=source_ids,
            support_gain=support_gain,
            total_sources=len(records),
        )
        yield PredictionArrays(
            path=path,
            boxes=fused_boxes,
            classes=fused_classes,
            confidences=fused_confidences,
            orig_shape=records[0].orig_shape,
        )


def _fused_multiscale_predictions(
    predictions_by_scale: dict[int, dict[Path, PredictionArrays]],
    image_paths: Sequence[Path],
    *,
    fusion_iou: float,
    max_det: int,
    support_gain: float = 0.0,
) -> Iterator[PredictionArrays]:
    scales = tuple(predictions_by_scale)
    if len(scales) < 2:
        raise ValueError("At least two prediction scales are required for fusion")
    yield from _fused_prediction_sources(
        [predictions_by_scale[scale] for scale in scales],
        image_paths,
        fusion_iou=fusion_iou,
        max_det=max_det,
        support_gain=support_gain,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference and create submission.csv")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=Path("data/processed/pseudo_rgb/images/test"))
    parser.add_argument("--output", type=Path, default=Path("submission_baseline.csv"))
    parser.add_argument(
        "--input-format",
        choices=("auto", "png", "npy"),
        default="auto",
        help=(
            "Input representation. 'auto' uses NPY for checkpoints with more than three "
            "input channels and PNG otherwise."
        ),
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use FP16 inference on supported GPUs to reduce memory use.",
    )
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable single-model test-time augmentation. Use only after held-out validation.",
    )
    parser.add_argument(
        "--multi-scale",
        nargs="+",
        type=int,
        help=(
            "Run the same checkpoint at two or more image sizes and fuse its same-class boxes. "
            "This is mutually exclusive with --augment."
        ),
    )
    parser.add_argument(
        "--fusion-iou",
        type=float,
        default=0.65,
        help="IoU threshold for confidence-weighted coordinate voting across scales.",
    )
    parser.add_argument(
        "--support-gain",
        type=float,
        default=0.0,
        help=(
            "Add a normalized confidence-mass bonus from agreeing scales when ranking "
            "fused boxes. Zero preserves the legacy max-confidence score."
        ),
    )
    parser.add_argument(
        "--tile-size",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        help=(
            "Run the same checkpoint on overlapping NPY tiles in addition to the full image. "
            "Boxes are mapped back to the full image and assigned by non-overlapping center cores."
        ),
    )
    parser.add_argument(
        "--tile-stride",
        nargs=2,
        type=int,
        default=(96, 192),
        metavar=("HEIGHT", "WIDTH"),
        help="Tile stride in source-image pixels (default: 96 192).",
    )
    parser.add_argument(
        "--tile-imgsz",
        type=int,
        default=1024,
        help="Model input size for each tile (default: 1024).",
    )
    args = parser.parse_args()
    if args.augment and args.multi_scale:
        parser.error("--augment and --multi-scale are mutually exclusive")
    if args.augment and args.tile_size:
        parser.error("--augment and --tile-size are mutually exclusive")
    if args.tile_size:
        if any(value <= 0 for value in (*args.tile_size, *args.tile_stride, args.tile_imgsz)):
            parser.error("tile size, stride, and model input size must be positive")
        if any(stride > size for stride, size in zip(args.tile_stride, args.tile_size, strict=True)):
            parser.error("each tile stride must not exceed its tile size")

    model = YOLO(str(args.weights.resolve()))
    channels = _checkpoint_channels(model)
    input_format = args.input_format
    if input_format == "auto":
        input_format = "npy" if channels > 3 else "png"
    suffix = f".{input_format}"
    image_paths = _numeric_paths(args.images, suffix)
    if not image_paths:
        raise FileNotFoundError(
            f"No {suffix} images found in {args.images.resolve()} for a {channels}-channel checkpoint"
        )
    if input_format == "png" and channels not in (1, 3):
        raise ValueError(
            f"A {channels}-channel checkpoint requires NPY inputs; use --input-format npy "
            "and the multispectral test directory"
        )
    if args.tile_size and input_format != "npy":
        raise ValueError("--tile-size currently requires NPY inputs so spectral channels stay exact")

    predict_kwargs: dict[str, object] = dict(
        batch=args.batch,
        device=args.device,
        half=args.half,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        verbose=False,
    )
    if args.multi_scale:
        scales = _normalize_multiscale(args.multi_scale)
        predictions_by_scale = _collect_multiscale_predictions(
            model,
            image_paths,
            image_directory=args.images,
            input_format=input_format,
            channels=channels,
            batch_size=args.batch,
            scales=scales,
            predict_kwargs=predict_kwargs,
        )
    elif args.tile_size:
        single_scale_kwargs = {
            **predict_kwargs,
            "imgsz": args.imgsz,
            "augment": False,
        }
        records = {
            record.path: record
            for record in _prediction_arrays(
                model,
                image_paths,
                image_directory=args.images,
                input_format=input_format,
                channels=channels,
                batch_size=args.batch,
                predict_kwargs=single_scale_kwargs,
            )
        }
        if set(records) != set(image_paths):
            raise RuntimeError(
                f"Full-image pass returned {len(records)} of {len(image_paths)} images"
            )
        scales = (args.imgsz,)
        predictions_by_scale = {args.imgsz: records}
    else:
        single_scale_kwargs = {
            **predict_kwargs,
            "imgsz": args.imgsz,
            "augment": args.augment,
        }
        prediction_items = _prediction_arrays(
            model,
            image_paths,
            image_directory=args.images,
            input_format=input_format,
            channels=channels,
            batch_size=args.batch,
            predict_kwargs=single_scale_kwargs,
        )
        inference_description = f"imgsz {args.imgsz}" + (", TTA" if args.augment else "")

    if args.tile_size:
        tile_height, tile_width = args.tile_size
        stride_height, stride_width = args.tile_stride
        tiled_predictions = _collect_tiled_predictions(
            model,
            image_paths,
            channels=channels,
            batch_size=args.batch,
            tile_height=tile_height,
            tile_width=tile_width,
            stride_height=stride_height,
            stride_width=stride_width,
            tile_imgsz=args.tile_imgsz,
            predict_kwargs=predict_kwargs,
        )
        prediction_items = _fused_full_and_tiled_predictions(
            predictions_by_scale,
            tiled_predictions,
            image_paths,
            fusion_iou=args.fusion_iou,
            max_det=args.max_det,
            support_gain=args.support_gain,
        )
        inference_description = (
            "full "
            + "/".join(str(scale) for scale in scales)
            + f" + tiles {tile_height}x{tile_width} stride {stride_height}x{stride_width}"
            + f" at {args.tile_imgsz}, vote IoU {args.fusion_iou:g}"
        )
        if args.support_gain:
            inference_description += f", support gain {args.support_gain:g}"
    elif args.multi_scale:
        prediction_items = _fused_multiscale_predictions(
            predictions_by_scale,
            image_paths,
            fusion_iou=args.fusion_iou,
            max_det=args.max_det,
            support_gain=args.support_gain,
        )
        inference_description = (
            "multi-scale "
            + "/".join(str(scale) for scale in scales)
            + f", vote IoU {args.fusion_iou:g}"
        )
        if args.support_gain:
            inference_description += f", support gain {args.support_gain:g}"

    rows: list[dict[str, int | float]] = []
    next_id = 0
    dropped_invalid = 0
    for prediction in prediction_items:
        image_path = prediction.path
        boxes = prediction.boxes
        classes = prediction.classes
        confidences = prediction.confidences
        height, width = prediction.orig_shape
        for box, class_id, confidence in zip(boxes, classes, confidences, strict=True):
            clipped_box = clip_xyxy(box.tolist(), width=width, height=height)
            if clipped_box is None:
                dropped_invalid += 1
                continue
            x1, y1, x2, y2 = clipped_box
            rows.append(
                {
                    "id": next_id,
                    "image_id": int(image_path.stem),
                    "class_id": int(class_id),
                    "confidence": float(confidence),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
            next_id += 1

    columns = ["id", "image_id", "class_id", "confidence", "x1", "y1", "x2", "y2"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(args.output, index=False)
    print(f"wrote {len(rows)} detections to {args.output.resolve()}")
    print(f"dropped {dropped_invalid} zero-area or non-finite predictions")
    print(f"input: {len(image_paths)} {input_format.upper()} images, {channels} channels")
    print(f"inference: {inference_description}")


if __name__ == "__main__":
    main()
