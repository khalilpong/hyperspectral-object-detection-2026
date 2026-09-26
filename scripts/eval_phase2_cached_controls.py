"""Audit the complete calibrated Phase 2 controls using frozen validation caches.

No checkpoint is loaded and no model inference, fitting, or submission occurs.
The three image groups are consistency diagnostics, not untouched holdout sets:
this validation set has already been used in earlier experiment selection.
"""

from __future__ import annotations

import argparse
import hashlib
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
from eval_tiled_inference import _remap_full_cache
from predict_submission import _fused_prediction_sources, _numeric_paths


ROOT = Path(__file__).resolve().parents[1]
FULL_CACHE = ROOT / "artifacts/ensemble_cache/m.pkl"
FLIP_CACHE = ROOT / "artifacts/flip_tta_cache/m_ablation_hflip_i1024.pkl"
EVIDENCE = ROOT / "artifacts/inference_mode_validation/m_ablation_flip_only_20260921.json"
AUDIT = ROOT / "artifacts/ensemble/single_m_box_calibration_oof_20260922.json"
DATA = ROOT / "data/processed/hsi16_shared_p005_995/dataset.yaml"
PINNED = {
    FULL_CACHE: "B63C2E81D683316DC5308DFF522D5BAE9AEEDE2BE7F4D0F1F826CCDF62731FD1",
    FLIP_CACHE: "A74BE45B29C9856446164008E4A51E2CF4C6DCFD803D59EF5FE20A3F967149F3",
    EVIDENCE: "49FA5146E4C1392F71C2D9BFE2B3609F742D3F85B2AAC29D9A855D7A4650F904",
    AUDIT: "056C8DD8A9274B3A4F49B8D5A9160BC4B42B021EA54EAD09A44E40F6B87844E2",
}
SCALES = (832, 896, 960, 1024, 1088, 1152, 1216)
CONTROLS = {
    "baseline": (0.74, False, 0.705708182597841),
    "A": (0.65, True, 0.7058779597408886),
    "B": (0.82, True, 0.705812158799068),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def require_control(actual: float, expected: float, *, tolerance: float = 1e-10) -> None:
    if not np.isfinite(actual) or abs(actual - expected) > tolerance:
        raise RuntimeError(f"Cache control drift: {actual} vs expected {expected}")


def group_indices(image_count: int) -> list[list[int]]:
    return [[index for index in range(image_count) if index % 3 == group] for group in range(3)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/phase2_last_day_20260927/cached_controls.json",
    )
    args = parser.parse_args()
    for path, expected in PINNED.items():
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"Frozen cache/evidence hash changed: {path}")
    names, directory = _dataset_details(DATA)
    paths = _numeric_paths(directory, ".npy")
    if len(paths) != 600:
        raise RuntimeError(f"Expected exactly 600 validation images, received {len(paths)}")
    identity = _remap_full_cache(FULL_CACHE, paths)
    if tuple(identity) != SCALES:
        raise RuntimeError("Identity cache scale order changed")
    with FLIP_CACHE.open("rb") as handle:
        flip = _remap_flip_cache(pickle.load(handle), paths)
    calibration, _ = load_audited_calibration(AUDIT)
    label_root = directory.parents[1] / "labels" / directory.name
    label_hashes = {path.stem: sha256(label_root / f"{path.stem}.txt") for path in paths}
    report = {
        "schema_version": 1,
        "cache_only": True,
        "new_inference_images": 0,
        "checkpoint_loaded": False,
        "validation_checkpoint_sha256": json.loads(EVIDENCE.read_text())["weights_sha256"],
        "source_hashes": {str(path.relative_to(ROOT)): value for path, value in PINNED.items()},
        "data_sha256": sha256(DATA),
        "labels_sha256": label_hashes,
        "script_sha256": sha256(Path(__file__)),
        "command": [sys.executable, *sys.argv],
        "groups": [[paths[index].stem for index in group] for group in group_indices(len(paths))],
        "limitation": "Previously tuned validation set; groups are consistency diagnostics, not new holdout evidence.",
        "controls": {},
    }
    for name, (fusion_iou, include_flip, expected) in CONTROLS.items():
        sources = [identity[scale] for scale in SCALES]
        if include_flip:
            sources.append(flip)
        raw = list(_fused_prediction_sources(
            sources, paths, fusion_iou=fusion_iou, max_det=300, support_gain=0.125,
        ))
        raw_metrics = _evaluate_predictions(iter(raw), names)
        require_control(raw_metrics["map50_95"], expected)
        calibrated = _calibrate_predictions(raw, calibration)
        result = {
            "fusion_iou": fusion_iou,
            "horizontal_flip_1024": include_flip,
            "raw_control": raw_metrics,
            "calibrated": _evaluate_predictions(iter(calibrated), names),
            "calibrated_groups": [
                _evaluate_predictions((calibrated[index] for index in group), names)
                for group in group_indices(len(paths))
            ],
        }
        report["controls"][name] = result
        print(json.dumps({"control": name, "raw_map": raw_metrics["map50_95"],
                          "calibrated_map": result["calibrated"]["map50_95"],
                          "groups": [group["map50_95"] for group in result["calibrated_groups"]]}), flush=True)
    base = report["controls"]["baseline"]
    for name in ("A", "B"):
        result = report["controls"][name]
        result["delta_vs_calibrated_baseline"] = result["calibrated"]["map50_95"] - base["calibrated"]["map50_95"]
        result["group_deltas_vs_calibrated_baseline"] = [
            group["map50_95"] - baseline["map50_95"]
            for group, baseline in zip(result["calibrated_groups"], base["calibrated_groups"], strict=True)
        ]
        result["class_deltas_vs_calibrated_baseline"] = {
            names[index]: float(value - base["calibrated"]["per_class_map50_95"][index])
            for index, value in enumerate(result["calibrated"]["per_class_map50_95"])
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"CACHE_CONTROL_AUDIT_COMPLETE {args.output}", flush=True)


if __name__ == "__main__":
    main()
