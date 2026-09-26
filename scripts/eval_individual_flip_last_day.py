"""Compare exactly A+hflip832 and A+hflip1216 using existing validation caches.

No checkpoint loading, prediction generation, training or production inputs.
This exploratory follow-up reports both candidates under the unchanged gate.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_multiscale_flip_last_day as base


def individual_sources(control: list, extra: dict, scale: int) -> list:
    if len(control) != 8 or scale not in base.NEW_SCALES:
        raise ValueError("Expected eight A sources and one fixed extra flip")
    return [*control, extra[scale]]


def main() -> None:
    for path, expected in base.PINNED.items():
        base.require_hash(path, expected)
    base.require_hash(base.WEIGHTS, base.CHECKPOINT_SHA)
    base.require_validation_lineage(json.loads(base.EVIDENCE.read_text()))
    names, directory = base._dataset_details(base.DATA)
    paths = base._numeric_paths(directory, ".npy")
    sizes, inventory = base.validation_inventory(paths)
    identity = base._remap_full_cache(base.FULL_CACHE, paths)
    if tuple(identity) != base.SCALES:
        raise ValueError("Identity scale order changed")
    with base.FLIP_CACHE.open("rb") as handle:
        flip1024 = base._remap_flip_cache(pickle.load(handle), paths)
    calibration, _ = base.load_audited_calibration(base.AUDIT)
    control_sources = [*[identity[scale] for scale in base.SCALES], flip1024]
    raw_a = list(base._fused_prediction_sources(
        control_sources, paths, fusion_iou=0.65, max_det=300, support_gain=0.125,
    ))
    raw_a_metrics = base._evaluate_predictions(iter(raw_a), names)
    base.require_control(raw_a_metrics["map50_95"], 0.7058779597408886)
    calibrated_a = base._calibrate_predictions(raw_a, calibration)
    a_metrics = base._evaluate_predictions(iter(calibrated_a), names)
    base.require_control(a_metrics["map50_95"], base.EXPECTED_A_CALIBRATED)
    a_groups = base.evaluate_groups(calibrated_a, names)
    print(f"A_CONTROL_PASS calibrated_map={a_metrics['map50_95']}", flush=True)

    extra, cache_checks = {}, {}
    for scale in base.NEW_SCALES:
        extra[scale], cache_checks[str(scale)] = base.cached_validation_flip(
            scale, paths, sizes, inventory, generate=False, model_holder=[],
        )
    if any(item["new_inference_images"] != 0 for item in cache_checks.values()):
        raise RuntimeError("Cache-only assessment unexpectedly generated predictions")

    candidates = {}
    for scale in base.NEW_SCALES:
        sources = individual_sources(control_sources, extra, scale)
        raw = list(base._fused_prediction_sources(
            sources, paths, fusion_iou=0.65, max_det=300, support_gain=0.125,
        ))
        calibrated = base._calibrate_predictions(raw, calibration)
        metrics = base._evaluate_predictions(iter(calibrated), names)
        groups = base.evaluate_groups(calibrated, names)
        gain = metrics["map50_95"] - a_metrics["map50_95"]
        deltas = [candidate["map50_95"] - reference["map50_95"]
                  for candidate, reference in zip(groups, a_groups, strict=True)]
        class_changes = [
            {"class_id": index, "class_name": names[index], "map50_95": value,
             "A_map50_95": reference, "delta_vs_A": value - reference}
            for index, (value, reference) in enumerate(zip(
                metrics["per_class_map50_95"], a_metrics["per_class_map50_95"], strict=True,
            ))
        ]
        candidates[str(scale)] = {
            "sources": [*[f"identity{s}" for s in base.SCALES], "hflip1024", f"hflip{scale}"],
            "total_sources": len(sources), "metrics": metrics, "groups": groups,
            "gain_vs_A": gain, "group_deltas_vs_A": deltas,
            "per_class_changes": class_changes,
            "boxes_delta_vs_A": metrics["predictions"] - a_metrics["predictions"],
            "passes_gate": base.passes_stability_gate(gain, deltas),
        }
        print(json.dumps({"extra_flip": scale, "map": metrics["map50_95"],
                          "gain_vs_A": gain, "group_deltas": deltas,
                          "passes_gate": candidates[str(scale)]["passes_gate"]}), flush=True)

    report = {
        "schema_version": 1, "experiment": "A_plus_one_extra_flip",
        "cache_only": True, "checkpoint_loaded": False, "new_inference_images": 0,
        "training_performed": False, "validation_only": True,
        "validation_checkpoint_sha256": base.CHECKPOINT_SHA,
        "frozen_cache_checkpoint_lineage_verified": True,
        "input_inventory": inventory, "cache_checks": cache_checks,
        "source_hashes": {str(path.relative_to(base.ROOT)): value for path, value in base.PINNED.items()},
        "code_sha256": {str(path.relative_to(base.ROOT)): base.sha256(path) for path in [
            Path(__file__), Path(base.__file__), base.ROOT / "scripts/predict_submission.py",
            base.ROOT / "scripts/compare_inference_modes.py", base.ROOT / "scripts/eval_box_calibration.py",
            base.ROOT / "src/hsi_detection/box_calibration.py"]},
        "fusion_iou": 0.65, "support_gain": 0.125, "max_det": 300, "boxscale": 1.01,
        "control": {"raw_A": raw_a_metrics, "calibrated_A": a_metrics, "groups_A": a_groups},
        "candidates": candidates,
        "gate": {"minimum_gain": 0.001, "minimum_positive_groups": 2, "worst_group_floor": -0.0005},
        "limitation": "Exploratory follow-up on reused validation data; groups are not independent holdout evidence.",
        "command": base.command(),
    }
    output = base.WORK / "individual_flip_validation.json"
    base.write_json(output, report)
    print(f"REPORT {output}", flush=True)


if __name__ == "__main__":
    main()
