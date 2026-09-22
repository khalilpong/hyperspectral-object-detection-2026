"""Cross-validate distinct-scale support-count scoring for one checkpoint.

The production incumbent uses one checkpoint at seven image scales, fuses boxes
with ``fusion_iou=0.74``, ranks them with confidence-mass
``support_gain=0.125``, and applies the already accepted global box calibration
``width_scale=height_scale=1.01``.  This cache-only audit keeps every part of
that geometry contract fixed and asks one narrow question: can a score based
only on the number of distinct supporting scales beat the incumbent?

Each fold selects a non-zero count gain on two folds only when it clears the
fit gate relative to the incumbent.  The selected rule is then measured on the
held-out fold.  The script never loads a model or creates a submission.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_inference_modes import _dataset_details, _evaluate_predictions  # noqa: E402
from eval_box_calibration import _calibrate_predictions  # noqa: E402
from eval_box_vote_geometry import (  # noqa: E402
    _fold_indices,
    _fuse_cached,
    _load_cache,
    _metrics_for_indices,
)
from hsi_detection.box_calibration import BoxCalibration  # noqa: E402
from predict_submission import (  # noqa: E402
    PredictionArrays,
    _iou_one_to_many,
    _numeric_paths,
)


INCUMBENT_NAME = "support_mass_gain_0.125"
GLOBAL_CALIBRATION = BoxCalibration(width_scale=1.01, height_scale=1.01)


def _count_candidate_name(gain: float) -> str:
    return f"support_count_gain_{gain:.3f}".replace(".", "p")


def _support_count_box_vote(
    boxes: np.ndarray,
    classes: np.ndarray,
    confidences: np.ndarray,
    *,
    iou_threshold: float,
    max_det: int,
    source_ids: np.ndarray,
    count_gain: float,
    total_sources: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use production-equivalent clustering with a distinct-source score.

    The final score is ``max_conf + count_gain * (n_sources - 1) /
    (total_sources - 1)`` clipped to one.  ``count_gain=0`` therefore preserves
    max-confidence ranking while coordinates remain the same confidence-weighted
    centroids used by production box voting.
    """

    boxes = np.asarray(boxes, dtype=np.float32)
    classes = np.asarray(classes, dtype=np.float32)
    confidences = np.asarray(confidences, dtype=np.float32)
    source_ids = np.asarray(source_ids, dtype=np.int64)
    if boxes.ndim != 2 or boxes.shape[1:] != (4,):
        raise ValueError(f"Expected boxes with shape N x 4, got {boxes.shape}")
    if len(classes) != len(boxes) or len(confidences) != len(boxes):
        raise ValueError("Box, class, and confidence arrays must have equal lengths")
    if len(source_ids) != len(boxes):
        raise ValueError("source_ids must have one entry per box")
    if not np.isfinite(iou_threshold) or not 0.0 < iou_threshold <= 1.0:
        raise ValueError("Fusion IoU must be finite and in (0, 1]")
    if max_det <= 0:
        raise ValueError("max_det must be positive")
    if not np.isfinite(count_gain) or count_gain < 0.0:
        raise ValueError("count_gain must be finite and non-negative")
    if total_sources < 2:
        raise ValueError("total_sources must be at least two")
    if np.any(source_ids < 0) or (
        len(source_ids) and int(source_ids.max()) >= total_sources
    ):
        raise ValueError("source_ids must be covered by total_sources")

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

    fused_boxes: list[np.ndarray] = []
    fused_classes: list[float] = []
    fused_scores: list[float] = []
    denominator = float(total_sources - 1)
    for class_id in np.unique(classes):
        class_indices = np.flatnonzero(classes == class_id)
        class_indices = class_indices[np.argsort(confidences[class_indices])[::-1]]
        cluster_boxes: list[np.ndarray] = []
        weighted_sums: list[np.ndarray] = []
        weight_sums: list[float] = []
        cluster_confidences: list[float] = []
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
                eligible_boxes = np.stack([cluster_boxes[item] for item in eligible])
                overlaps = _iou_one_to_many(box, eligible_boxes)
                best_position = int(np.argmax(overlaps))
                if float(overlaps[best_position]) >= iou_threshold:
                    best_cluster = eligible[best_position]

            weight = max(confidence, 1e-8)
            if best_cluster is None:
                cluster_boxes.append(box.copy())
                weighted_sums.append(box.astype(np.float64) * weight)
                weight_sums.append(weight)
                cluster_confidences.append(confidence)
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
                cluster_sources[best_cluster].add(source_id)

        fused_boxes.extend(cluster_boxes)
        fused_classes.extend([float(class_id)] * len(cluster_boxes))
        fused_scores.extend(
            min(
                1.0,
                maximum
                + count_gain * (len(sources) - 1) / denominator,
            )
            for maximum, sources in zip(
                cluster_confidences, cluster_sources, strict=True
            )
        )

    output_boxes = np.asarray(fused_boxes, dtype=np.float32)
    output_classes = np.asarray(fused_classes, dtype=np.float32)
    output_scores = np.asarray(fused_scores, dtype=np.float32)
    order = np.argsort(output_scores)[::-1][:max_det]
    return output_boxes[order], output_classes[order], output_scores[order]


def _fuse_count_scoring(
    passes: dict[int, dict[str, PredictionArrays]],
    image_paths: list[Path],
    *,
    fusion_iou: float,
    count_gain: float,
    max_det: int,
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
        fused = _support_count_box_vote(
            boxes,
            classes,
            confidences,
            iou_threshold=fusion_iou,
            max_det=max_det,
            source_ids=source_ids,
            count_gain=count_gain,
            total_sources=len(records),
        )
        predictions.append(
            PredictionArrays(path, fused[0], fused[1], fused[2], records[0].orig_shape)
        )
    return predictions


def _select_candidate(
    metrics_by_candidate: dict[str, dict[str, object]],
    *,
    incumbent_name: str,
    count_gains: tuple[float, ...],
    minimum_fit_gain: float,
) -> str:
    count_names = [_count_candidate_name(gain) for gain in count_gains]
    best_count = max(
        count_names,
        key=lambda name: (
            float(metrics_by_candidate[name]["map50_95"]),
            -count_gains[count_names.index(name)],
        ),
    )
    gain = float(metrics_by_candidate[best_count]["map50_95"]) - float(
        metrics_by_candidate[incumbent_name]["map50_95"]
    )
    return best_count if gain >= minimum_fit_gain else incumbent_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--count-gains",
        nargs="+",
        type=float,
        default=[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08],
    )
    parser.add_argument("--fusion-iou", type=float, default=0.74)
    parser.add_argument("--incumbent-support-gain", type=float, default=0.125)
    parser.add_argument("--box-scale", type=float, default=1.01)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--expected-incumbent", type=float, default=0.7075064748)
    parser.add_argument("--baseline-tolerance", type=float, default=0.000002)
    parser.add_argument("--minimum-fit-gain", type=float, default=0.001)
    parser.add_argument("--minimum-oof-gain", type=float, default=0.001)
    args = parser.parse_args()

    count_gains = tuple(dict.fromkeys(float(value) for value in args.count_gains))
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if any(not np.isfinite(value) or value < 0.0 for value in count_gains):
        parser.error("Every count gain must be finite and non-negative")
    if 0.0 not in count_gains:
        parser.error("--count-gains must include the zero control")
    if not 0.0 < args.fusion_iou <= 1.0:
        parser.error("--fusion-iou must be in (0, 1]")
    if args.incumbent_support_gain < 0.0:
        parser.error("--incumbent-support-gain must be non-negative")
    if not np.isfinite(args.box_scale) or args.box_scale <= 0.0:
        parser.error("--box-scale must be finite and positive")
    if args.minimum_fit_gain < 0.0 or args.minimum_oof_gain < 0.0:
        parser.error("Gain thresholds must be non-negative")

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

    calibration = BoxCalibration(
        width_scale=args.box_scale, height_scale=args.box_scale
    )
    incumbent = _calibrate_predictions(
        _fuse_cached(
            passes,
            image_paths,
            fusion_iou=args.fusion_iou,
            support_gain=args.incumbent_support_gain,
            max_det=args.max_det,
            variant="legacy",
        ),
        calibration,
    )
    predictions_by_candidate = {INCUMBENT_NAME: incumbent}
    for gain in count_gains:
        predictions_by_candidate[_count_candidate_name(gain)] = _calibrate_predictions(
            _fuse_count_scoring(
                passes,
                image_paths,
                fusion_iou=args.fusion_iou,
                count_gain=gain,
                max_det=args.max_det,
            ),
            calibration,
        )

    print(
        f"cache-only support-count audit: {len(image_paths)} images, scales={scales}, "
        f"candidates={len(count_gains)}, folds={args.folds}",
        flush=True,
    )
    full_metrics = {
        candidate: _evaluate_predictions(iter(predictions), names)
        for candidate, predictions in predictions_by_candidate.items()
    }
    observed_incumbent = float(full_metrics[INCUMBENT_NAME]["map50_95"])
    if abs(observed_incumbent - args.expected_incumbent) > args.baseline_tolerance:
        raise RuntimeError(
            f"Incumbent mismatch: observed {observed_incumbent:.10f}, "
            f"expected {args.expected_incumbent:.10f} +/- {args.baseline_tolerance:g}"
        )

    oof_predictions: list[PredictionArrays | None] = [None] * len(image_paths)
    fold_records: list[dict[str, object]] = []
    for fold_index in range(args.folds):
        held_out = _fold_indices(len(image_paths), fold_index, args.folds)
        held_out_set = set(held_out)
        training = [
            index for index in range(len(image_paths)) if index not in held_out_set
        ]
        training_metrics = {
            candidate: _metrics_for_indices(predictions, training, names)
            for candidate, predictions in predictions_by_candidate.items()
        }
        chosen = _select_candidate(
            training_metrics,
            incumbent_name=INCUMBENT_NAME,
            count_gains=count_gains,
            minimum_fit_gain=args.minimum_fit_gain,
        )
        for index in held_out:
            oof_predictions[index] = predictions_by_candidate[chosen][index]

        incumbent_held_out = _metrics_for_indices(incumbent, held_out, names)
        chosen_held_out = _metrics_for_indices(
            predictions_by_candidate[chosen], held_out, names
        )
        training_gain = float(training_metrics[chosen]["map50_95"]) - float(
            training_metrics[INCUMBENT_NAME]["map50_95"]
        )
        held_out_delta = float(chosen_held_out["map50_95"]) - float(
            incumbent_held_out["map50_95"]
        )
        fold_records.append(
            {
                "fold": fold_index,
                "training_images": len(training),
                "held_out_images": len(held_out),
                "chosen": chosen,
                "training_incumbent": training_metrics[INCUMBENT_NAME],
                "training_chosen": training_metrics[chosen],
                "training_gain_map50_95": training_gain,
                "held_out_incumbent": incumbent_held_out,
                "held_out_chosen": chosen_held_out,
                "delta_map50_95": held_out_delta,
            }
        )
        print(
            f"fold {fold_index}: {chosen} train_gain={training_gain:+.8f} "
            f"held_out_delta={held_out_delta:+.8f}",
            flush=True,
        )

    if any(prediction is None for prediction in oof_predictions):
        raise RuntimeError("Internal error: out-of-fold predictions are incomplete")
    complete_oof = [prediction for prediction in oof_predictions if prediction is not None]
    oof_metrics = _evaluate_predictions(iter(complete_oof), names)
    oof_delta = float(oof_metrics["map50_95"]) - observed_incumbent
    fold_deltas = [float(record["delta_map50_95"]) for record in fold_records]
    full_chosen = _select_candidate(
        full_metrics,
        incumbent_name=INCUMBENT_NAME,
        count_gains=count_gains,
        minimum_fit_gain=args.minimum_fit_gain,
    )
    full_delta = float(full_metrics[full_chosen]["map50_95"]) - observed_incumbent
    fold_choices = [str(record["chosen"]) for record in fold_records]
    selection_stability = {
        "unanimous_fold_choice": len(set(fold_choices)) == 1,
        "full_choice_matches_every_fold": all(
            choice == full_chosen for choice in fold_choices
        ),
        "fold_choices": fold_choices,
    }
    passes_stability_gate = (
        full_chosen != INCUMBENT_NAME
        and oof_delta >= args.minimum_oof_gain
        and sum(delta > 0.0 for delta in fold_deltas) >= 2
        and min(fold_deltas) >= -0.0005
        and selection_stability["unanimous_fold_choice"]
        and selection_stability["full_choice_matches_every_fold"]
    )

    candidate_records = [
        {
            "name": name,
            "count_gain": gain,
            "metrics": full_metrics[name],
            "delta_vs_incumbent_map50_95": float(full_metrics[name]["map50_95"])
            - observed_incumbent,
        }
        for gain in count_gains
        for name in [_count_candidate_name(gain)]
    ]
    candidate_records.sort(
        key=lambda record: float(record["metrics"]["map50_95"]), reverse=True
    )
    payload = {
        "contract": {
            "cache_only": True,
            "single_checkpoint_multiscale": True,
            "scales": scales,
            "images": len(image_paths),
            "fusion_iou": args.fusion_iou,
            "incumbent_support_gain": args.incumbent_support_gain,
            "fixed_global_box_scale": args.box_scale,
            "count_gains": count_gains,
            "count_score_formula": (
                "min(1,max_conf+count_gain*(distinct_sources-1)/(total_sources-1))"
            ),
            "fold_rule": "numeric validation order modulo fold count",
            "minimum_fit_gain_vs_incumbent": args.minimum_fit_gain,
            "minimum_oof_gain_vs_incumbent": args.minimum_oof_gain,
            "stability_gate": (
                "non-incumbent full choice, OOF gain >= minimum, >=2 positive folds, "
                "worst fold >= -0.0005, and one unanimous fold/full selection"
            ),
        },
        "incumbent": {
            "name": INCUMBENT_NAME,
            "metrics": full_metrics[INCUMBENT_NAME],
        },
        "folds": fold_records,
        "oof": {
            "metrics": oof_metrics,
            "delta_vs_incumbent_map50_95": oof_delta,
        },
        "full_selection": {
            "chosen": full_chosen,
            "metrics": full_metrics[full_chosen],
            "delta_vs_incumbent_map50_95": full_delta,
        },
        "selection_stability": selection_stability,
        "passes_stability_gate": passes_stability_gate,
        "candidates": candidate_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"OOF mAP50-95={oof_metrics['map50_95']:.8f} delta={oof_delta:+.8f}; "
        f"full={full_chosen} delta={full_delta:+.8f}; "
        f"gate={passes_stability_gate}; wrote {args.output.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
