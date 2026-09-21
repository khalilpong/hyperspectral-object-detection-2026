"""Cache-only sweep of source subsets and fixed scale weights for one checkpoint.

The experiment is deliberately pre-registered and narrow.  It accepts the
existing seven-scale validation cache, never loads a model, and evaluates only:

* leave-one-scale-out subsets;
* fixed central three/five-scale subsets; and
* two fixed, symmetric confidence-weight templates.

All candidates retain the production box-vote implementation.  The full
validation gain and deterministic three-fold deltas are reported against the
unchanged seven-scale control.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_inference_modes import _dataset_details, _evaluate_predictions  # noqa: E402
from eval_box_vote_geometry import _fold_indices, _load_cache, _metrics_for_indices  # noqa: E402
from predict_submission import PredictionArrays, _box_vote, _numeric_paths  # noqa: E402


EXPECTED_SCALES = (832, 896, 960, 1024, 1088, 1152, 1216)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    scales: tuple[int, ...]
    confidence_weights: tuple[float, ...]


def _source_specs(scales: tuple[int, ...]) -> tuple[SourceSpec, ...]:
    if scales != EXPECTED_SCALES:
        raise ValueError(
            f"Expected exact seven-scale cache {EXPECTED_SCALES}, got {scales}"
        )

    uniform = tuple(1.0 for _ in scales)
    specs = [SourceSpec("all7_control", scales, uniform)]
    specs.extend(
        SourceSpec(
            f"drop_{dropped}",
            tuple(scale for scale in scales if scale != dropped),
            tuple(1.0 for scale in scales if scale != dropped),
        )
        for dropped in scales
    )
    specs.extend(
        (
            SourceSpec("center3", (960, 1024, 1088), (1.0, 1.0, 1.0)),
            SourceSpec(
                "center5",
                (896, 960, 1024, 1088, 1152),
                (1.0, 1.0, 1.0, 1.0, 1.0),
            ),
            SourceSpec(
                "center_weighted",
                scales,
                (0.9, 0.95, 1.0, 1.1, 1.0, 0.95, 0.9),
            ),
            SourceSpec(
                "ends_weighted",
                scales,
                (1.1, 1.0, 0.95, 0.9, 0.95, 1.0, 1.1),
            ),
        )
    )
    return tuple(specs)


def _fuse_cached_sources(
    passes: dict[int, dict[str, PredictionArrays]],
    image_paths: list[Path],
    *,
    spec: SourceSpec,
    fusion_iou: float,
    support_gain: float,
    max_det: int,
) -> list[PredictionArrays]:
    if len(spec.scales) != len(spec.confidence_weights):
        raise ValueError("Every selected scale requires one confidence weight")
    if not spec.scales:
        raise ValueError("At least one scale is required")
    if any(scale not in passes for scale in spec.scales):
        raise ValueError("Source spec references a scale absent from the cache")
    if any(not np.isfinite(weight) or weight <= 0.0 for weight in spec.confidence_weights):
        raise ValueError("Every scale confidence weight must be finite and positive")

    predictions: list[PredictionArrays] = []
    for path in image_paths:
        records = [passes[scale][path.stem] for scale in spec.scales]
        boxes = np.concatenate([record.boxes for record in records], axis=0)
        classes = np.concatenate([record.classes for record in records], axis=0)
        confidences = np.concatenate(
            [
                record.confidences * weight
                for record, weight in zip(
                    records, spec.confidence_weights, strict=True
                )
            ],
            axis=0,
        )
        source_ids = np.concatenate(
            [
                np.full(len(record.boxes), index, dtype=np.int64)
                for index, record in enumerate(records)
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
        predictions.append(
            PredictionArrays(
                path,
                fused_boxes,
                fused_classes,
                fused_confidences,
                records[0].orig_shape,
            )
        )
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fusion-iou", type=float, default=0.74)
    parser.add_argument("--support-gain", type=float, default=0.125)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--expected-baseline", type=float, default=0.7057081826)
    parser.add_argument("--baseline-tolerance", type=float, default=0.000002)
    parser.add_argument("--minimum-full-gain", type=float, default=0.001)
    parser.add_argument("--minimum-worst-fold-gain", type=float, default=-0.0005)
    args = parser.parse_args()

    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if not np.isfinite(args.fusion_iou) or not 0.0 < args.fusion_iou <= 1.0:
        parser.error("--fusion-iou must be finite and in (0, 1]")
    if not np.isfinite(args.support_gain) or args.support_gain < 0.0:
        parser.error("--support-gain must be finite and non-negative")

    names, validation_directory = _dataset_details(args.data)
    image_paths = [
        path.resolve() for path in _numeric_paths(validation_directory, ".npy")
    ]
    scales, passes = _load_cache(args.cache_file)
    specs = _source_specs(scales)
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
        f"cache-only scale-source sweep: {len(image_paths)} images, "
        f"scales={scales}, candidates={len(specs) - 1}",
        flush=True,
    )
    control_spec = specs[0]
    control_predictions = _fuse_cached_sources(
        passes,
        image_paths,
        spec=control_spec,
        fusion_iou=args.fusion_iou,
        support_gain=args.support_gain,
        max_det=args.max_det,
    )
    control_metrics = _evaluate_predictions(iter(control_predictions), names)
    observed_baseline = float(control_metrics["map50_95"])
    if abs(observed_baseline - args.expected_baseline) > args.baseline_tolerance:
        raise RuntimeError(
            f"Baseline mismatch: observed {observed_baseline:.10f}, "
            f"expected {args.expected_baseline:.10f} +/- {args.baseline_tolerance:g}"
        )
    control_folds = [
        _metrics_for_indices(
            control_predictions,
            _fold_indices(len(image_paths), fold, args.folds),
            names,
        )
        for fold in range(args.folds)
    ]

    records: list[dict[str, object]] = []
    for spec in specs[1:]:
        predictions = _fuse_cached_sources(
            passes,
            image_paths,
            spec=spec,
            fusion_iou=args.fusion_iou,
            support_gain=args.support_gain,
            max_det=args.max_det,
        )
        metrics = _evaluate_predictions(iter(predictions), names)
        delta = float(metrics["map50_95"]) - observed_baseline
        fold_records: list[dict[str, object]] = []
        for fold in range(args.folds):
            indices = _fold_indices(len(image_paths), fold, args.folds)
            candidate = _metrics_for_indices(predictions, indices, names)
            fold_delta = float(candidate["map50_95"]) - float(
                control_folds[fold]["map50_95"]
            )
            fold_records.append(
                {
                    "fold": fold,
                    "control": control_folds[fold],
                    "candidate": candidate,
                    "delta_map50_95": fold_delta,
                }
            )
        fold_deltas = [float(record["delta_map50_95"]) for record in fold_records]
        passes_gate = (
            delta >= args.minimum_full_gain
            and min(fold_deltas) >= args.minimum_worst_fold_gain
            and sum(value > 0.0 for value in fold_deltas) >= 2
        )
        records.append(
            {
                "name": spec.name,
                "scales": spec.scales,
                "confidence_weights": spec.confidence_weights,
                "metrics": metrics,
                "delta_map50_95": delta,
                "folds": fold_records,
                "passes_stability_gate": passes_gate,
            }
        )
        print(
            f"{spec.name:<18} mAP50-95={metrics['map50_95']:.8f} "
            f"delta={delta:+.8f} folds={tuple(round(value, 8) for value in fold_deltas)} "
            f"gate={passes_gate}",
            flush=True,
        )

    payload = {
        "contract": {
            "cache_only": True,
            "single_checkpoint_multiscale": True,
            "pre_registered_candidates": True,
            "images": len(image_paths),
            "control_scales": scales,
            "fusion_iou": args.fusion_iou,
            "support_gain": args.support_gain,
            "fold_rule": "numeric validation order modulo fold count",
            "minimum_full_gain": args.minimum_full_gain,
            "minimum_worst_fold_gain": args.minimum_worst_fold_gain,
            "minimum_positive_folds": 2,
        },
        "control": {
            "name": control_spec.name,
            "scales": control_spec.scales,
            "confidence_weights": control_spec.confidence_weights,
            "metrics": control_metrics,
            "folds": control_folds,
        },
        "candidates": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
