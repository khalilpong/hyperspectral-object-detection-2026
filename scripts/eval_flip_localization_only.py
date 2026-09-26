"""Test one preregistered, cache-only role for the existing horizontal flip.

Keep every seven-identity-scale detection, class and confidence. Match the flip
one-to-one by class and IoU, and use it only as one additional coordinate vote.
No new detections, confidence re-ranking, checkpoint loading or model inference.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibrate_submission_boxes import load_audited_calibration
from compare_inference_modes import _dataset_details, _evaluate_predictions
from eval_box_calibration import _calibrate_predictions
from eval_flip_tta import _remap_flip_cache
from eval_phase2_cached_controls import (
    AUDIT, DATA, FLIP_CACHE, FULL_CACHE, PINNED, ROOT, SCALES,
    group_indices, require_control, sha256,
)
from eval_tiled_inference import _remap_full_cache
from predict_submission import PredictionArrays, _fused_prediction_sources, _iou_one_to_many, _numeric_paths


def refine_identity_geometry(
    identity: PredictionArrays, flip: PredictionArrays,
    *, identity_sources: int = 7, match_iou: float = 0.65,
) -> tuple[PredictionArrays, int]:
    """Greedy maximum-IoU one-to-one matching; all non-coordinate fields survive."""
    if identity.path != flip.path or identity.orig_shape != flip.orig_shape:
        raise ValueError("Identity and flip image/shape must match")
    if identity_sources <= 0 or not 0 < match_iou <= 1:
        raise ValueError("Invalid source count or match threshold")
    boxes = identity.boxes.astype(np.float32, copy=True)
    edges = []
    for index, box in enumerate(boxes):
        same = np.flatnonzero(flip.classes == identity.classes[index])
        if not len(same):
            continue
        overlaps = _iou_one_to_many(box, flip.boxes[same])
        edges.extend((float(overlap), index, int(other))
                     for overlap, other in zip(overlaps, same, strict=True)
                     if overlap >= match_iou)
    used_identity, used_flip = set(), set()
    for _, index, other in sorted(edges, key=lambda edge: (-edge[0], edge[1], edge[2])):
        if index in used_identity or other in used_flip:
            continue
        # Seven identity sources supply one aggregate; the one flip supplies one vote.
        base_weight = identity_sources * float(identity.confidences[index])
        flip_weight = float(flip.confidences[other])
        if base_weight + flip_weight <= 0:
            continue
        boxes[index] = (
            base_weight * identity.boxes[index] + flip_weight * flip.boxes[other]
        ) / (base_weight + flip_weight)
        used_identity.add(index)
        used_flip.add(other)
    return PredictionArrays(identity.path, boxes, identity.classes.copy(),
                            identity.confidences.copy(), identity.orig_shape), len(used_identity)


def main() -> None:
    for path, expected in PINNED.items():
        if sha256(path) != expected:
            raise RuntimeError(f"Frozen evidence changed: {path}")
    controls_path = ROOT / "artifacts/phase2_last_day_20260927/cached_controls.json"
    controls = json.loads(controls_path.read_text())
    names, directory = _dataset_details(DATA)
    paths = _numeric_paths(directory, ".npy")
    if len(paths) != 600:
        raise RuntimeError("Expected the original 600 validation images")
    identity = _remap_full_cache(FULL_CACHE, paths)
    with FLIP_CACHE.open("rb") as handle:
        flip = _remap_flip_cache(pickle.load(handle), paths)
    calibration, _ = load_audited_calibration(AUDIT)
    baseline = list(_fused_prediction_sources(
        [identity[scale] for scale in SCALES], paths,
        fusion_iou=0.74, max_det=300, support_gain=0.125,
    ))
    actual = _evaluate_predictions(iter(baseline), names)["map50_95"]
    require_control(actual, 0.705708182597841)
    refined, counts = [], []
    for prediction in baseline:
        result, matches = refine_identity_geometry(prediction, flip[prediction.path])
        assert np.array_equal(result.classes, prediction.classes)
        assert np.array_equal(result.confidences, prediction.confidences)
        assert len(result.boxes) == len(prediction.boxes)
        refined.append(result)
        counts.append(matches)
    calibrated = _calibrate_predictions(refined, calibration)
    metrics = _evaluate_predictions(iter(calibrated), names)
    groups = [_evaluate_predictions((calibrated[index] for index in group), names)
              for group in group_indices(len(paths))]
    comparisons = {}
    for label in ("baseline", "A", "B"):
        reference = controls["controls"][label]
        comparisons[label] = {
            "delta_map50_95": metrics["map50_95"] - reference["calibrated"]["map50_95"],
            "group_deltas": [item["map50_95"] - ref["map50_95"]
                             for item, ref in zip(groups, reference["calibrated_groups"], strict=True)],
        }
    a = comparisons["A"]
    passes_gate = (a["delta_map50_95"] >= 0.001
                   and sum(value > 0 for value in a["group_deltas"]) >= 2
                   and min(a["group_deltas"]) >= -0.0005)
    report = {
        "schema_version": 1, "hypothesis": "horizontal_flip_localization_only",
        "cache_only": True, "checkpoint_loaded": False, "new_inference_images": 0,
        "settings": {"identity_sources": 7, "identity_fusion_iou": 0.74,
                     "flip_match_iou": 0.65, "support_gain": 0.125, "max_det": 300,
                     "boxscale": 1.01, "new_detections": False, "confidence_changed": False},
        "source_hashes": {str(path.relative_to(ROOT)): expected for path, expected in PINNED.items()},
        "controls_sha256": sha256(controls_path), "script_sha256": sha256(Path(__file__)),
        "command": [sys.executable, *sys.argv],
        "matched_boxes": sum(counts), "metrics": metrics, "groups": groups,
        "comparisons": comparisons, "passes_gate": passes_gate,
        "gate": {"minimum_gain_vs_A": 0.001, "minimum_positive_groups": 2,
                 "worst_group_floor": -0.0005},
        "limitation": "Exploratory reuse of a previously tuned validation set; no untouched holdout claim.",
    }
    output = ROOT / "artifacts/phase2_last_day_20260927/flip_localization_only.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"map": metrics["map50_95"], "matched_boxes": sum(counts),
                      "comparisons": comparisons, "passes_gate": passes_gate}), flush=True)


if __name__ == "__main__":
    main()
