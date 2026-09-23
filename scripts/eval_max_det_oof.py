"""Cross-validate the final detection cap for one checkpoint's cached scales.

The production incumbent uses one YOLO26m checkpoint at seven image scales,
confidence-mass box voting, and the accepted global 1.01 box calibration.  This
cache-only audit changes only the final number of fused detections retained per
image.  It never loads a model and never creates a competition submission.
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
from predict_submission import PredictionArrays, _numeric_paths  # noqa: E402


INCUMBENT_MAX_DET = 300


def _candidate_name(max_det: int) -> str:
    return f"max_det_{max_det}"


def _select_max_det(
    metrics_by_candidate: dict[int, dict[str, object]],
    *,
    incumbent_max_det: int,
    minimum_fit_gain: float,
) -> int:
    alternatives = [
        max_det for max_det in metrics_by_candidate if max_det != incumbent_max_det
    ]
    if not alternatives:
        return incumbent_max_det
    best = max(
        alternatives,
        key=lambda max_det: (
            float(metrics_by_candidate[max_det]["map50_95"]),
            -abs(max_det - incumbent_max_det),
            -max_det,
        ),
    )
    gain = float(metrics_by_candidate[best]["map50_95"]) - float(
        metrics_by_candidate[incumbent_max_det]["map50_95"]
    )
    return best if gain >= minimum_fit_gain else incumbent_max_det


def _prediction_count_summary(
    predictions: list[PredictionArrays], max_det: int
) -> dict[str, object]:
    counts = np.asarray([len(prediction.boxes) for prediction in predictions])
    return {
        "total_predictions": int(counts.sum()),
        "minimum_per_image": int(counts.min()) if len(counts) else 0,
        "median_per_image": float(np.median(counts)) if len(counts) else 0.0,
        "maximum_per_image": int(counts.max()) if len(counts) else 0,
        "images_at_limit": int(np.count_nonzero(counts == max_det)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-dets", nargs="+", type=int, default=[200, 300, 400, 500]
    )
    parser.add_argument("--fusion-iou", type=float, default=0.74)
    parser.add_argument("--support-gain", type=float, default=0.125)
    parser.add_argument("--box-scale", type=float, default=1.01)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--expected-incumbent", type=float, default=0.7075064748)
    parser.add_argument("--baseline-tolerance", type=float, default=0.000002)
    parser.add_argument("--minimum-fit-gain", type=float, default=0.001)
    parser.add_argument("--minimum-oof-gain", type=float, default=0.001)
    args = parser.parse_args()

    max_dets = tuple(dict.fromkeys(int(value) for value in args.max_dets))
    if INCUMBENT_MAX_DET not in max_dets:
        parser.error(f"--max-dets must include the {INCUMBENT_MAX_DET} incumbent")
    if any(value <= 0 for value in max_dets):
        parser.error("Every max-det value must be positive")
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if not 0.0 < args.fusion_iou <= 1.0:
        parser.error("--fusion-iou must be in (0, 1]")
    if args.support_gain < 0.0:
        parser.error("--support-gain must be non-negative")
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
    predictions_by_candidate = {
        max_det: _calibrate_predictions(
            _fuse_cached(
                passes,
                image_paths,
                fusion_iou=args.fusion_iou,
                support_gain=args.support_gain,
                max_det=max_det,
                variant="legacy",
            ),
            calibration,
        )
        for max_det in max_dets
    }
    print(
        f"cache-only max-det audit: {len(image_paths)} images, scales={scales}, "
        f"candidates={max_dets}, folds={args.folds}",
        flush=True,
    )

    full_metrics = {
        max_det: _evaluate_predictions(iter(predictions), names)
        for max_det, predictions in predictions_by_candidate.items()
    }
    observed_incumbent = float(full_metrics[INCUMBENT_MAX_DET]["map50_95"])
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
            max_det: _metrics_for_indices(predictions, training, names)
            for max_det, predictions in predictions_by_candidate.items()
        }
        chosen = _select_max_det(
            training_metrics,
            incumbent_max_det=INCUMBENT_MAX_DET,
            minimum_fit_gain=args.minimum_fit_gain,
        )
        for index in held_out:
            oof_predictions[index] = predictions_by_candidate[chosen][index]

        incumbent_held_out = _metrics_for_indices(
            predictions_by_candidate[INCUMBENT_MAX_DET], held_out, names
        )
        chosen_held_out = _metrics_for_indices(
            predictions_by_candidate[chosen], held_out, names
        )
        training_gain = float(training_metrics[chosen]["map50_95"]) - float(
            training_metrics[INCUMBENT_MAX_DET]["map50_95"]
        )
        held_out_delta = float(chosen_held_out["map50_95"]) - float(
            incumbent_held_out["map50_95"]
        )
        fold_records.append(
            {
                "fold": fold_index,
                "training_images": len(training),
                "held_out_images": len(held_out),
                "chosen_max_det": chosen,
                "training_incumbent": training_metrics[INCUMBENT_MAX_DET],
                "training_chosen": training_metrics[chosen],
                "training_gain_map50_95": training_gain,
                "held_out_incumbent": incumbent_held_out,
                "held_out_chosen": chosen_held_out,
                "delta_map50_95": held_out_delta,
            }
        )
        print(
            f"fold {fold_index}: max_det={chosen} train_gain={training_gain:+.8f} "
            f"held_out_delta={held_out_delta:+.8f}",
            flush=True,
        )

    if any(prediction is None for prediction in oof_predictions):
        raise RuntimeError("Internal error: out-of-fold predictions are incomplete")
    complete_oof = [
        prediction for prediction in oof_predictions if prediction is not None
    ]
    oof_metrics = _evaluate_predictions(iter(complete_oof), names)
    oof_delta = float(oof_metrics["map50_95"]) - observed_incumbent
    fold_deltas = [float(record["delta_map50_95"]) for record in fold_records]
    full_chosen = _select_max_det(
        full_metrics,
        incumbent_max_det=INCUMBENT_MAX_DET,
        minimum_fit_gain=args.minimum_fit_gain,
    )
    full_delta = (
        float(full_metrics[full_chosen]["map50_95"]) - observed_incumbent
    )
    fold_choices = [int(record["chosen_max_det"]) for record in fold_records]
    selection_stability = {
        "unanimous_fold_choice": len(set(fold_choices)) == 1,
        "full_choice_matches_every_fold": all(
            choice == full_chosen for choice in fold_choices
        ),
        "fold_choices": fold_choices,
    }
    passes_stability_gate = (
        full_chosen != INCUMBENT_MAX_DET
        and oof_delta >= args.minimum_oof_gain
        and sum(delta > 0.0 for delta in fold_deltas) >= 2
        and min(fold_deltas) >= -0.0005
        and selection_stability["unanimous_fold_choice"]
        and selection_stability["full_choice_matches_every_fold"]
    )

    candidate_records = [
        {
            "name": _candidate_name(max_det),
            "max_det": max_det,
            "metrics": full_metrics[max_det],
            "delta_vs_incumbent_map50_95": float(
                full_metrics[max_det]["map50_95"]
            )
            - observed_incumbent,
            "prediction_counts": _prediction_count_summary(
                predictions_by_candidate[max_det], max_det
            ),
        }
        for max_det in max_dets
    ]
    candidate_records.sort(
        key=lambda record: float(record["metrics"]["map50_95"]), reverse=True
    )
    payload = {
        "contract": {
            "cache_only": True,
            "single_checkpoint_multiscale": True,
            "only_variable": "final fused detections retained per image",
            "scales": scales,
            "images": len(image_paths),
            "fusion_iou": args.fusion_iou,
            "support_gain": args.support_gain,
            "fixed_global_box_scale": args.box_scale,
            "max_dets": max_dets,
            "incumbent_max_det": INCUMBENT_MAX_DET,
            "fold_rule": "numeric validation order modulo fold count",
            "minimum_fit_gain_vs_incumbent": args.minimum_fit_gain,
            "minimum_oof_gain_vs_incumbent": args.minimum_oof_gain,
            "stability_gate": (
                "non-incumbent full choice, OOF gain >= minimum, >=2 positive "
                "folds, worst fold >= -0.0005, and one unanimous fold/full selection"
            ),
        },
        "incumbent": {
            "name": _candidate_name(INCUMBENT_MAX_DET),
            "metrics": full_metrics[INCUMBENT_MAX_DET],
        },
        "folds": fold_records,
        "oof": {
            "metrics": oof_metrics,
            "delta_vs_incumbent_map50_95": oof_delta,
        },
        "full_selection": {
            "chosen_max_det": full_chosen,
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
        f"full=max_det_{full_chosen} delta={full_delta:+.8f}; "
        f"gate={passes_stability_gate}; wrote {args.output.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
