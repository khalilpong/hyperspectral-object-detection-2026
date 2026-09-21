"""Evaluate geometry-only variants of single-checkpoint multi-scale box voting.

This script is intentionally cache-only: it accepts exactly one validation
prediction cache and never loads a model or checkpoint.  Every source in the
cache is one input scale from the same checkpoint.  Candidate variants either
change how a box is assigned to an existing cluster or how a cluster's final
coordinates are estimated; confidence/support scoring remains unchanged.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Literal

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_inference_modes import _dataset_details, _evaluate_predictions  # noqa: E402
from predict_submission import (  # noqa: E402
    PredictionArrays,
    _box_vote,
    _iou_one_to_many,
    _numeric_paths,
)


ClusterMode = Literal["centroid", "anchor", "medoid"]
CoordinateMode = Literal[
    "mean",
    "uniform",
    "sqrt_conf",
    "square_conf",
    "top2",
    "top3",
    "top4",
    "iou_anchor",
    "trimmed1",
]

VARIANTS: dict[str, tuple[ClusterMode, CoordinateMode]] = {
    "anchor_mean": ("anchor", "mean"),
    "medoid_mean": ("medoid", "mean"),
    "centroid_uniform": ("centroid", "uniform"),
    "centroid_sqrt_conf": ("centroid", "sqrt_conf"),
    "centroid_square_conf": ("centroid", "square_conf"),
    "centroid_top2": ("centroid", "top2"),
    "centroid_top3": ("centroid", "top3"),
    "centroid_top4": ("centroid", "top4"),
    "centroid_iou_anchor": ("centroid", "iou_anchor"),
    "centroid_trimmed1": ("centroid", "trimmed1"),
}


def _load_cache(
    cache_path: Path,
) -> tuple[tuple[int, ...], dict[int, dict[str, PredictionArrays]]]:
    if not cache_path.is_file():
        raise FileNotFoundError(f"Validation cache not found: {cache_path.resolve()}")
    with cache_path.open("rb") as handle:
        stored = pickle.load(handle)
    if not isinstance(stored, dict) or len(stored) < 2:
        raise ValueError("Expected a multi-scale prediction cache keyed by image size")
    scales = tuple(int(scale) for scale in stored)
    by_stem = {
        int(scale): {Path(path).stem: record for path, record in records.items()}
        for scale, records in stored.items()
    }
    return scales, by_stem


def _pairwise_iou(boxes: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return np.empty((0, 0), dtype=np.float32)
    top_left = np.maximum(boxes[:, None, :2], boxes[None, :, :2])
    bottom_right = np.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
    intersections = np.prod(np.maximum(bottom_right - top_left, 0.0), axis=2)
    areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(
        boxes[:, 3] - boxes[:, 1], 0.0
    )
    unions = areas[:, None] + areas[None, :] - intersections
    return intersections / np.maximum(unions, 1e-12)


def _weighted_mean(boxes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    safe_weights = np.maximum(np.asarray(weights, dtype=np.float64), 1e-8)
    return np.average(boxes.astype(np.float64), axis=0, weights=safe_weights).astype(
        np.float32
    )


def _coordinate_estimate(
    member_boxes: np.ndarray,
    member_confidences: np.ndarray,
    *,
    mode: CoordinateMode,
) -> np.ndarray:
    confidences = np.maximum(member_confidences.astype(np.float64), 1e-8)
    selected = np.arange(len(member_boxes))
    weights = confidences

    if mode == "uniform":
        weights = np.ones_like(confidences)
    elif mode == "sqrt_conf":
        weights = np.sqrt(confidences)
    elif mode == "square_conf":
        weights = np.square(confidences)
    elif mode.startswith("top"):
        keep = int(mode.removeprefix("top"))
        selected = np.argsort(confidences)[::-1][:keep]
        weights = confidences[selected]
    elif mode == "iou_anchor":
        anchor = member_boxes[int(np.argmax(confidences))]
        overlaps = _iou_one_to_many(anchor, member_boxes).astype(np.float64)
        weights = confidences * np.maximum(overlaps, 1e-8)
    elif mode == "trimmed1" and len(member_boxes) >= 3:
        provisional = _weighted_mean(member_boxes, confidences)
        overlaps = _iou_one_to_many(provisional, member_boxes)
        selected = np.delete(selected, int(np.argmin(overlaps)))
        weights = confidences[selected]
    elif mode != "mean" and mode != "trimmed1":
        raise ValueError(f"Unsupported coordinate mode: {mode}")

    return _weighted_mean(member_boxes[selected], weights)


def _variant_box_vote(
    boxes: np.ndarray,
    classes: np.ndarray,
    confidences: np.ndarray,
    *,
    iou_threshold: float,
    max_det: int,
    source_ids: np.ndarray,
    support_gain: float,
    total_sources: int,
    cluster_mode: ClusterMode,
    coordinate_mode: CoordinateMode,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = np.asarray(boxes, dtype=np.float32)
    classes = np.asarray(classes, dtype=np.float32)
    confidences = np.asarray(confidences, dtype=np.float32)
    source_ids = np.asarray(source_ids, dtype=np.int64)
    valid = (
        np.isfinite(boxes).all(axis=1)
        & np.isfinite(classes)
        & np.isfinite(confidences)
        & (boxes[:, 2] > boxes[:, 0])
        & (boxes[:, 3] > boxes[:, 1])
        & (confidences >= 0.0)
        & (source_ids >= 0)
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

    output_boxes: list[np.ndarray] = []
    output_classes: list[float] = []
    output_confidences: list[float] = []
    for class_id in np.unique(classes):
        class_indices = np.flatnonzero(classes == class_id)
        class_indices = class_indices[np.argsort(confidences[class_indices])[::-1]]
        cluster_members: list[list[int]] = []
        cluster_sources: list[set[int]] = []
        cluster_references: list[np.ndarray] = []

        for index in class_indices:
            box = boxes[index]
            source_id = int(source_ids[index])
            eligible = [
                cluster_index
                for cluster_index, sources in enumerate(cluster_sources)
                if source_id not in sources
            ]
            best_cluster: int | None = None
            if eligible:
                reference_boxes = np.stack(
                    [cluster_references[cluster_index] for cluster_index in eligible]
                )
                overlaps = _iou_one_to_many(box, reference_boxes)
                best_position = int(np.argmax(overlaps))
                if float(overlaps[best_position]) >= iou_threshold:
                    best_cluster = eligible[best_position]

            if best_cluster is None:
                cluster_members.append([int(index)])
                cluster_sources.append({source_id})
                cluster_references.append(box.copy())
                continue

            members = cluster_members[best_cluster]
            members.append(int(index))
            cluster_sources[best_cluster].add(source_id)
            member_indices = np.asarray(members, dtype=np.int64)
            member_boxes = boxes[member_indices]
            member_confidences = confidences[member_indices]
            if cluster_mode == "centroid":
                cluster_references[best_cluster] = _coordinate_estimate(
                    member_boxes, member_confidences, mode="mean"
                )
            elif cluster_mode == "medoid":
                medoid = int(np.argmax(_pairwise_iou(member_boxes).sum(axis=1)))
                cluster_references[best_cluster] = member_boxes[medoid].copy()
            elif cluster_mode != "anchor":
                raise ValueError(f"Unsupported cluster mode: {cluster_mode}")

        for members in cluster_members:
            member_indices = np.asarray(members, dtype=np.int64)
            member_boxes = boxes[member_indices]
            member_confidences = confidences[member_indices]
            maximum = float(member_confidences.max())
            total = float(member_confidences.sum(dtype=np.float64))
            output_boxes.append(
                _coordinate_estimate(
                    member_boxes, member_confidences, mode=coordinate_mode
                )
            )
            output_classes.append(float(class_id))
            output_confidences.append(
                min(
                    1.0,
                    maximum
                    + support_gain * max(total - maximum, 0.0) / total_sources,
                )
                if support_gain
                else maximum
            )

    output_boxes_array = np.asarray(output_boxes, dtype=np.float32)
    output_classes_array = np.asarray(output_classes, dtype=np.float32)
    output_confidences_array = np.asarray(output_confidences, dtype=np.float32)
    order = np.argsort(output_confidences_array)[::-1][:max_det]
    return (
        output_boxes_array[order],
        output_classes_array[order],
        output_confidences_array[order],
    )


def _fuse_cached(
    passes: dict[int, dict[str, PredictionArrays]],
    image_paths: list[Path],
    *,
    fusion_iou: float,
    support_gain: float,
    max_det: int,
    variant: str,
) -> list[PredictionArrays]:
    scales = tuple(passes)
    predictions: list[PredictionArrays] = []
    for path in image_paths:
        records = [passes[scale][path.stem] for scale in scales]
        boxes = np.concatenate([record.boxes for record in records], axis=0)
        classes = np.concatenate([record.classes for record in records], axis=0)
        confidences = np.concatenate([record.confidences for record in records], axis=0)
        source_ids = np.concatenate(
            [
                np.full(len(record.boxes), index, dtype=np.int64)
                for index, record in enumerate(records)
            ]
        )

        if variant == "legacy":
            fused = _box_vote(
                boxes,
                classes,
                confidences,
                iou_threshold=fusion_iou,
                max_det=max_det,
                source_ids=source_ids,
                support_gain=support_gain,
                total_sources=len(records),
            )
        else:
            cluster_mode, coordinate_mode = VARIANTS[variant]
            fused = _variant_box_vote(
                boxes,
                classes,
                confidences,
                iou_threshold=fusion_iou,
                max_det=max_det,
                source_ids=source_ids,
                support_gain=support_gain,
                total_sources=len(records),
                cluster_mode=cluster_mode,
                coordinate_mode=coordinate_mode,
            )
        predictions.append(
            PredictionArrays(path, fused[0], fused[1], fused[2], records[0].orig_shape)
        )
    return predictions


def _fold_indices(image_count: int, fold_index: int, fold_count: int) -> list[int]:
    return [index for index in range(image_count) if index % fold_count == fold_index]


def _metrics_for_indices(
    predictions: list[PredictionArrays], indices: list[int], names: dict[int, str]
) -> dict[str, object]:
    return _evaluate_predictions((predictions[index] for index in indices), names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANTS),
        default=tuple(VARIANTS),
    )
    parser.add_argument("--fusion-iou", type=float, default=0.74)
    parser.add_argument("--support-gain", type=float, default=0.125)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--expected-baseline", type=float, default=0.7057081826)
    parser.add_argument("--baseline-tolerance", type=float, default=0.000002)
    parser.add_argument("--minimum-full-gain", type=float, default=0.001)
    args = parser.parse_args()

    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if not 0.0 < args.fusion_iou <= 1.0:
        parser.error("--fusion-iou must be in (0, 1]")
    if args.support_gain < 0.0:
        parser.error("--support-gain must be non-negative")

    names, validation_directory = _dataset_details(args.data)
    image_paths = [
        path.resolve() for path in _numeric_paths(validation_directory, ".npy")
    ]
    scales, passes = _load_cache(args.cache_file)
    missing = {
        path.stem
        for path in image_paths
        if any(path.stem not in passes[scale] for scale in scales)
    }
    if missing:
        raise RuntimeError(
            f"Cache misses {len(missing)} validation images, e.g. {sorted(missing)[:3]}"
        )

    print(
        f"cache-only geometry sweep: {len(image_paths)} images, scales={scales}, "
        f"variants={tuple(args.variants)}",
        flush=True,
    )
    baseline_predictions = _fuse_cached(
        passes,
        image_paths,
        fusion_iou=args.fusion_iou,
        support_gain=args.support_gain,
        max_det=args.max_det,
        variant="legacy",
    )
    baseline_metrics = _evaluate_predictions(iter(baseline_predictions), names)
    observed_baseline = float(baseline_metrics["map50_95"])
    if abs(observed_baseline - args.expected_baseline) > args.baseline_tolerance:
        raise RuntimeError(
            f"Baseline mismatch: observed {observed_baseline:.10f}, "
            f"expected {args.expected_baseline:.10f} +/- {args.baseline_tolerance:g}"
        )

    baseline_folds = [
        _metrics_for_indices(
            baseline_predictions,
            _fold_indices(len(image_paths), fold, args.folds),
            names,
        )
        for fold in range(args.folds)
    ]
    records: list[dict[str, object]] = []
    for variant in args.variants:
        predictions = _fuse_cached(
            passes,
            image_paths,
            fusion_iou=args.fusion_iou,
            support_gain=args.support_gain,
            max_det=args.max_det,
            variant=variant,
        )
        metrics = _evaluate_predictions(iter(predictions), names)
        delta = float(metrics["map50_95"]) - observed_baseline
        fold_records: list[dict[str, object]] = []
        for fold in range(args.folds):
            indices = _fold_indices(len(image_paths), fold, args.folds)
            candidate = _metrics_for_indices(predictions, indices, names)
            fold_delta = float(candidate["map50_95"]) - float(
                baseline_folds[fold]["map50_95"]
            )
            fold_records.append(
                {
                    "fold": fold,
                    "baseline": baseline_folds[fold],
                    "candidate": candidate,
                    "delta_map50_95": fold_delta,
                }
            )
        fold_deltas = [float(record["delta_map50_95"]) for record in fold_records]
        passes_gate = (
            delta >= args.minimum_full_gain
            and sum(value > 0.0 for value in fold_deltas) >= 2
            and min(fold_deltas) >= -0.0005
        )
        records.append(
            {
                "variant": variant,
                "cluster_mode": VARIANTS[variant][0],
                "coordinate_mode": VARIANTS[variant][1],
                "metrics": metrics,
                "delta_map50_95": delta,
                "folds": fold_records,
                "passes_stability_gate": passes_gate,
            }
        )
        print(
            f"{variant:<24} mAP50-95={metrics['map50_95']:.8f} "
            f"delta={delta:+.8f} folds={tuple(round(value, 8) for value in fold_deltas)} "
            f"gate={passes_gate}",
            flush=True,
        )

    payload = {
        "contract": {
            "cache_only": True,
            "single_checkpoint_multiscale": True,
            "scales": scales,
            "images": len(image_paths),
            "fusion_iou": args.fusion_iou,
            "support_gain": args.support_gain,
            "fold_rule": "numeric validation order modulo fold count",
            "minimum_full_gain": args.minimum_full_gain,
        },
        "baseline": baseline_metrics,
        "variants": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
