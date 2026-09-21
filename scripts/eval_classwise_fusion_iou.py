"""Cross-validate class-specific IoU thresholds for one-model multi-scale voting.

The evaluator is cache-only and never instantiates a model.  It chooses a
fusion-IoU threshold for each class on two folds and evaluates that fixed choice
on the held-out fold.  A class keeps the global baseline threshold unless its
training-fold gain clears a fixed margin, which limits validation overfitting.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_inference_modes import _dataset_details, _evaluate_predictions  # noqa: E402
from predict_submission import PredictionArrays, _box_vote, _numeric_paths  # noqa: E402


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


def _fuse_cached(
    passes: dict[int, dict[str, PredictionArrays]],
    image_paths: list[Path],
    *,
    thresholds: dict[int, float],
    default_threshold: float,
    support_gain: float,
    max_det: int,
) -> list[PredictionArrays]:
    scales = tuple(passes)
    fused: list[PredictionArrays] = []
    for path in image_paths:
        records = [passes[scale][path.stem] for scale in scales]
        boxes = np.concatenate([record.boxes for record in records], axis=0)
        classes = np.concatenate([record.classes for record in records], axis=0)
        confidences = np.concatenate([record.confidences for record in records], axis=0)
        source_ids = np.concatenate(
            [
                np.full(len(record.boxes), source_index, dtype=np.int64)
                for source_index, record in enumerate(records)
            ]
        )

        iou_threshold: float | dict[int, float]
        iou_threshold = thresholds if thresholds else default_threshold
        output_boxes, output_classes, output_confidences = _box_vote(
            boxes,
            classes,
            confidences,
            iou_threshold=iou_threshold,
            max_det=max_det,
            source_ids=source_ids,
            support_gain=support_gain,
            total_sources=len(records),
        )
        fused.append(
            PredictionArrays(
                path,
                output_boxes,
                output_classes,
                output_confidences,
                records[0].orig_shape,
            )
        )
    return fused


def _fold_indices(image_count: int, fold_index: int, fold_count: int) -> list[int]:
    return [index for index in range(image_count) if index % fold_count == fold_index]


def _metrics_for_indices(
    predictions: list[PredictionArrays], indices: list[int], names: dict[int, str]
) -> dict[str, object]:
    return _evaluate_predictions((predictions[index] for index in indices), names)


def _choose_thresholds(
    metrics_by_threshold: dict[float, dict[str, object]],
    *,
    names: dict[int, str],
    baseline_threshold: float,
    minimum_class_gain: float,
) -> dict[int, float]:
    baseline_maps = metrics_by_threshold[baseline_threshold]["per_class_map50_95"]
    chosen: dict[int, float] = {}
    for class_id in names:
        baseline_map = float(baseline_maps[class_id])
        best_threshold = max(
            metrics_by_threshold,
            key=lambda threshold: (
                float(metrics_by_threshold[threshold]["per_class_map50_95"][class_id]),
                -abs(threshold - baseline_threshold),
            ),
        )
        best_map = float(
            metrics_by_threshold[best_threshold]["per_class_map50_95"][class_id]
        )
        chosen[class_id] = (
            best_threshold
            if best_map - baseline_map >= minimum_class_gain
            else baseline_threshold
        )
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fusion-ious",
        nargs="+",
        type=float,
        default=[0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80],
    )
    parser.add_argument("--baseline-iou", type=float, default=0.74)
    parser.add_argument("--support-gain", type=float, default=0.125)
    parser.add_argument("--minimum-class-gain", type=float, default=0.002)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--expected-baseline", type=float, default=0.7057081826)
    parser.add_argument("--baseline-tolerance", type=float, default=0.000002)
    args = parser.parse_args()

    thresholds = tuple(dict.fromkeys(args.fusion_ious))
    if args.baseline_iou not in thresholds:
        parser.error("--fusion-ious must include --baseline-iou")
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if args.minimum_class_gain < 0.0:
        parser.error("--minimum-class-gain must be non-negative")
    if any(not 0.0 < threshold <= 1.0 for threshold in thresholds):
        parser.error("Every fusion IoU must be in (0, 1]")

    names, val_directory = _dataset_details(args.data)
    image_paths = [path.resolve() for path in _numeric_paths(val_directory, ".npy")]
    scales, passes = _load_cache(args.cache_file)
    missing = {
        path.stem
        for path in image_paths
        if any(path.stem not in passes[scale] for scale in scales)
    }
    if missing:
        raise RuntimeError(f"Cache misses {len(missing)} validation images, e.g. {sorted(missing)[:3]}")

    print(
        f"cache-only classwise sweep: {len(image_paths)} images, scales={scales}, "
        f"thresholds={thresholds}, folds={args.folds}",
        flush=True,
    )
    uniform_predictions: dict[float, list[PredictionArrays]] = {}
    uniform_full_metrics: dict[float, dict[str, object]] = {}
    for threshold in thresholds:
        predictions = _fuse_cached(
            passes,
            image_paths,
            thresholds={},
            default_threshold=threshold,
            support_gain=args.support_gain,
            max_det=args.max_det,
        )
        metrics = _evaluate_predictions(iter(predictions), names)
        uniform_predictions[threshold] = predictions
        uniform_full_metrics[threshold] = metrics
        print(
            f"uniform f{threshold:.2f}: mAP50-95={metrics['map50_95']:.8f} "
            f"mAP75={metrics['map75']:.8f}",
            flush=True,
        )

    observed_baseline = float(uniform_full_metrics[args.baseline_iou]["map50_95"])
    if abs(observed_baseline - args.expected_baseline) > args.baseline_tolerance:
        raise RuntimeError(
            f"Baseline mismatch: observed {observed_baseline:.10f}, "
            f"expected {args.expected_baseline:.10f} +/- {args.baseline_tolerance:g}"
        )

    oof_predictions: list[PredictionArrays | None] = [None] * len(image_paths)
    fold_records: list[dict[str, object]] = []
    for fold_index in range(args.folds):
        held_out = _fold_indices(len(image_paths), fold_index, args.folds)
        held_out_set = set(held_out)
        training = [index for index in range(len(image_paths)) if index not in held_out_set]
        training_metrics = {
            threshold: _metrics_for_indices(predictions, training, names)
            for threshold, predictions in uniform_predictions.items()
        }
        chosen = _choose_thresholds(
            training_metrics,
            names=names,
            baseline_threshold=args.baseline_iou,
            minimum_class_gain=args.minimum_class_gain,
        )
        fold_paths = [image_paths[index] for index in held_out]
        fold_predictions = _fuse_cached(
            passes,
            fold_paths,
            thresholds=chosen,
            default_threshold=args.baseline_iou,
            support_gain=args.support_gain,
            max_det=args.max_det,
        )
        for index, prediction in zip(held_out, fold_predictions, strict=True):
            oof_predictions[index] = prediction
        mixed_metrics = _evaluate_predictions(iter(fold_predictions), names)
        baseline_metrics = _metrics_for_indices(
            uniform_predictions[args.baseline_iou], held_out, names
        )
        delta = float(mixed_metrics["map50_95"]) - float(baseline_metrics["map50_95"])
        fold_records.append(
            {
                "fold": fold_index,
                "training_images": len(training),
                "held_out_images": len(held_out),
                "chosen_thresholds": {str(key): value for key, value in chosen.items()},
                "baseline": baseline_metrics,
                "mixed": mixed_metrics,
                "delta_map50_95": delta,
            }
        )
        print(
            f"fold {fold_index}: baseline={baseline_metrics['map50_95']:.8f} "
            f"mixed={mixed_metrics['map50_95']:.8f} delta={delta:+.8f}",
            flush=True,
        )

    if any(prediction is None for prediction in oof_predictions):
        raise RuntimeError("Internal error: out-of-fold predictions are incomplete")
    complete_oof = [prediction for prediction in oof_predictions if prediction is not None]
    oof_metrics = _evaluate_predictions(iter(complete_oof), names)
    oof_delta = float(oof_metrics["map50_95"]) - observed_baseline

    full_chosen = _choose_thresholds(
        uniform_full_metrics,
        names=names,
        baseline_threshold=args.baseline_iou,
        minimum_class_gain=args.minimum_class_gain,
    )
    full_mixed_predictions = _fuse_cached(
        passes,
        image_paths,
        thresholds=full_chosen,
        default_threshold=args.baseline_iou,
        support_gain=args.support_gain,
        max_det=args.max_det,
    )
    full_mixed_metrics = _evaluate_predictions(iter(full_mixed_predictions), names)
    full_delta = float(full_mixed_metrics["map50_95"]) - observed_baseline
    fold_deltas = [float(record["delta_map50_95"]) for record in fold_records]
    passes_stability_gate = (
        oof_delta >= 0.001
        and sum(delta > 0.0 for delta in fold_deltas) >= 2
        and min(fold_deltas) >= -0.0005
    )

    payload = {
        "contract": {
            "cache_only": True,
            "single_checkpoint_multiscale": True,
            "scales": scales,
            "images": len(image_paths),
            "fusion_ious": thresholds,
            "baseline_iou": args.baseline_iou,
            "support_gain": args.support_gain,
            "minimum_class_gain": args.minimum_class_gain,
            "fold_rule": "numeric validation order modulo fold count",
        },
        "uniform_full_metrics": {
            str(threshold): metrics
            for threshold, metrics in uniform_full_metrics.items()
        },
        "folds": fold_records,
        "oof": {
            "metrics": oof_metrics,
            "delta_map50_95": oof_delta,
        },
        "full_selection": {
            "chosen_thresholds": {str(key): value for key, value in full_chosen.items()},
            "metrics": full_mixed_metrics,
            "delta_map50_95": full_delta,
        },
        "passes_stability_gate": passes_stability_gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"OOF mAP50-95={oof_metrics['map50_95']:.8f} delta={oof_delta:+.8f}; "
        f"full-selected delta={full_delta:+.8f}; gate={passes_stability_gate}; "
        f"wrote {args.output.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
