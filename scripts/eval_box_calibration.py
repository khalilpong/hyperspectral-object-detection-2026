"""Cross-validate global box-geometry calibration for one checkpoint.

The input is a cache of multi-scale predictions produced by one detection
checkpoint.  The cache is fused once with the production box-vote settings,
then every candidate applies one global transform to the fused boxes.  A
three-fold out-of-fold procedure selects a transform on two folds and measures
it only on the held-out fold, which prevents a full-validation grid winner from
being mistaken for evidence that should be used on the competition test set.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_inference_modes import _dataset_details, _evaluate_predictions  # noqa: E402
from eval_box_vote_geometry import (  # noqa: E402
    _fold_indices,
    _fuse_cached,
    _load_cache,
    _metrics_for_indices,
)
from hsi_detection.box_calibration import (  # noqa: E402
    IDENTITY_BOX_CALIBRATION as IDENTITY,
    BoxCalibration as Calibration,
    box_calibration_name,
    calibrate_xyxy,
)
from predict_submission import PredictionArrays, _numeric_paths  # noqa: E402


def _candidate_calibrations(
    scale_values: tuple[float, ...], shift_values: tuple[float, ...]
) -> tuple[Calibration, ...]:
    """Build two bounded families without fitting scale and shift jointly."""

    candidates = {IDENTITY}
    candidates.update(
        Calibration(width_scale=width_scale, height_scale=height_scale)
        for width_scale in scale_values
        for height_scale in scale_values
    )
    candidates.update(
        Calibration(center_x_shift=center_x_shift, center_y_shift=center_y_shift)
        for center_x_shift in shift_values
        for center_y_shift in shift_values
    )
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                candidate.distance_from_identity,
                candidate.width_scale,
                candidate.height_scale,
                candidate.center_x_shift,
                candidate.center_y_shift,
            ),
        )
    )


def _calibrate_prediction(
    prediction: PredictionArrays, calibration: Calibration
) -> PredictionArrays:
    boxes = np.asarray(prediction.boxes, dtype=np.float32)
    if not len(boxes):
        return PredictionArrays(
            prediction.path,
            boxes.copy(),
            prediction.classes.copy(),
            prediction.confidences.copy(),
            prediction.orig_shape,
        )

    image_height, image_width = prediction.orig_shape
    calibrated, valid = calibrate_xyxy(
        boxes,
        image_width=image_width,
        image_height=image_height,
        calibration=calibration,
    )
    return PredictionArrays(
        prediction.path,
        calibrated[valid],
        prediction.classes[valid].copy(),
        prediction.confidences[valid].copy(),
        prediction.orig_shape,
    )


def _calibrate_predictions(
    predictions: list[PredictionArrays], calibration: Calibration
) -> list[PredictionArrays]:
    return [_calibrate_prediction(prediction, calibration) for prediction in predictions]


def _select_calibration(
    metrics_by_candidate: dict[Calibration, dict[str, object]],
    *,
    identity_map: float,
    minimum_fit_gain: float,
) -> Calibration:
    best = max(
        metrics_by_candidate,
        key=lambda candidate: (
            float(metrics_by_candidate[candidate]["map50_95"]),
            -candidate.distance_from_identity,
            box_calibration_name(candidate),
        ),
    )
    gain = float(metrics_by_candidate[best]["map50_95"]) - identity_map
    return best if gain >= minimum_fit_gain else IDENTITY


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scale-values",
        nargs="+",
        type=float,
        default=[0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03],
    )
    parser.add_argument(
        "--shift-values",
        nargs="+",
        type=float,
        default=[-0.02, -0.01, 0.0, 0.01, 0.02],
    )
    parser.add_argument("--fusion-iou", type=float, default=0.74)
    parser.add_argument("--support-gain", type=float, default=0.125)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--expected-baseline", type=float, default=0.7057081826)
    parser.add_argument("--baseline-tolerance", type=float, default=0.000002)
    parser.add_argument("--minimum-fit-gain", type=float, default=0.001)
    parser.add_argument("--minimum-oof-gain", type=float, default=0.001)
    args = parser.parse_args()

    scale_values = tuple(dict.fromkeys(float(value) for value in args.scale_values))
    shift_values = tuple(dict.fromkeys(float(value) for value in args.shift_values))
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if any(not np.isfinite(value) or value <= 0.0 for value in scale_values):
        parser.error("Every scale value must be finite and positive")
    if any(not np.isfinite(value) or abs(value) > 0.1 for value in shift_values):
        parser.error("Every shift value must be finite and within [-0.1, 0.1]")
    if 1.0 not in scale_values or 0.0 not in shift_values:
        parser.error("The candidate grids must include scale 1.0 and shift 0.0")
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

    candidates = _candidate_calibrations(scale_values, shift_values)
    print(
        f"cache-only box calibration: {len(image_paths)} images, scales={scales}, "
        f"candidates={len(candidates)}, folds={args.folds}",
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
    predictions_by_candidate = {
        candidate: _calibrate_predictions(baseline_predictions, candidate)
        for candidate in candidates
    }
    full_metrics = {
        candidate: _evaluate_predictions(iter(predictions), names)
        for candidate, predictions in predictions_by_candidate.items()
    }
    baseline_metrics = full_metrics[IDENTITY]
    observed_baseline = float(baseline_metrics["map50_95"])
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
        training = [
            index for index in range(len(image_paths)) if index not in held_out_set
        ]
        training_metrics = {
            candidate: _metrics_for_indices(predictions, training, names)
            for candidate, predictions in predictions_by_candidate.items()
        }
        training_identity_map = float(training_metrics[IDENTITY]["map50_95"])
        chosen = _select_calibration(
            training_metrics,
            identity_map=training_identity_map,
            minimum_fit_gain=args.minimum_fit_gain,
        )
        for index in held_out:
            oof_predictions[index] = predictions_by_candidate[chosen][index]

        baseline_held_out = _metrics_for_indices(
            predictions_by_candidate[IDENTITY], held_out, names
        )
        chosen_held_out = _metrics_for_indices(
            predictions_by_candidate[chosen], held_out, names
        )
        fold_delta = float(chosen_held_out["map50_95"]) - float(
            baseline_held_out["map50_95"]
        )
        fold_records.append(
            {
                "fold": fold_index,
                "training_images": len(training),
                "held_out_images": len(held_out),
                "chosen": asdict(chosen),
                "chosen_name": box_calibration_name(chosen),
                "training_baseline": training_metrics[IDENTITY],
                "training_chosen": training_metrics[chosen],
                "training_gain_map50_95": float(
                    training_metrics[chosen]["map50_95"]
                )
                - training_identity_map,
                "held_out_baseline": baseline_held_out,
                "held_out_chosen": chosen_held_out,
                "delta_map50_95": fold_delta,
            }
        )
        print(
            f"fold {fold_index}: {box_calibration_name(chosen)} "
            f"train_gain={fold_records[-1]['training_gain_map50_95']:+.8f} "
            f"held_out_delta={fold_delta:+.8f}",
            flush=True,
        )

    if any(prediction is None for prediction in oof_predictions):
        raise RuntimeError("Internal error: out-of-fold predictions are incomplete")
    complete_oof = [prediction for prediction in oof_predictions if prediction is not None]
    oof_metrics = _evaluate_predictions(iter(complete_oof), names)
    oof_delta = float(oof_metrics["map50_95"]) - observed_baseline
    fold_deltas = [float(record["delta_map50_95"]) for record in fold_records]
    full_chosen = _select_calibration(
        full_metrics,
        identity_map=observed_baseline,
        minimum_fit_gain=args.minimum_fit_gain,
    )
    full_delta = float(full_metrics[full_chosen]["map50_95"]) - observed_baseline
    fold_chosen_names = [str(record["chosen_name"]) for record in fold_records]
    full_chosen_name = box_calibration_name(full_chosen)
    selection_stability = {
        "unanimous_fold_choice": len(set(fold_chosen_names)) == 1,
        "full_choice_matches_every_fold": all(
            chosen_name == full_chosen_name for chosen_name in fold_chosen_names
        ),
        "fold_chosen_names": fold_chosen_names,
    }
    passes_stability_gate = (
        oof_delta >= args.minimum_oof_gain
        and sum(delta > 0.0 for delta in fold_deltas) >= 2
        and min(fold_deltas) >= -0.0005
        and selection_stability["unanimous_fold_choice"]
        and selection_stability["full_choice_matches_every_fold"]
    )
    candidate_records = [
        {
            "name": box_calibration_name(candidate),
            "calibration": asdict(candidate),
            "metrics": full_metrics[candidate],
            "delta_map50_95": float(full_metrics[candidate]["map50_95"])
            - observed_baseline,
        }
        for candidate in sorted(
            candidates,
            key=lambda item: float(full_metrics[item]["map50_95"]),
            reverse=True,
        )
    ]
    payload = {
        "contract": {
            "cache_only": True,
            "single_checkpoint_multiscale": True,
            "global_calibration_only": True,
            "joint_scale_and_shift_search": False,
            "scales": scales,
            "images": len(image_paths),
            "fusion_iou": args.fusion_iou,
            "support_gain": args.support_gain,
            "scale_values": scale_values,
            "shift_values": shift_values,
            "fold_rule": "numeric validation order modulo fold count",
            "minimum_fit_gain": args.minimum_fit_gain,
            "minimum_oof_gain": args.minimum_oof_gain,
            "stability_gate": (
                "OOF gain >= minimum, >=2 positive folds, worst fold >= -0.0005, "
                "and one unanimous fold/full selection"
            ),
        },
        "baseline": baseline_metrics,
        "folds": fold_records,
        "oof": {
            "metrics": oof_metrics,
            "delta_map50_95": oof_delta,
        },
        "full_selection": {
            "chosen": asdict(full_chosen),
            "chosen_name": full_chosen_name,
            "metrics": full_metrics[full_chosen],
            "delta_map50_95": full_delta,
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
        f"full={box_calibration_name(full_chosen)} delta={full_delta:+.8f}; "
        f"gate={passes_stability_gate}; wrote {args.output.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
