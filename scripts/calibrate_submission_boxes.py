"""Apply one audited global box calibration to an existing submission CSV."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from hsi_detection.box_calibration import (
    BoxCalibration,
    box_calibration_name,
    calibrate_xyxy,
)
from hsi_detection.submission import SUBMISSION_COLUMNS, validate_submission_frame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _image_sizes(directory: Path) -> dict[int, tuple[int, int]]:
    sizes: dict[int, tuple[int, int]] = {}
    for path in directory.glob("*.png"):
        with Image.open(path) as image:
            sizes[int(path.stem)] = image.size
    if not sizes:
        raise FileNotFoundError(f"No PNG shape references found in {directory.resolve()}")
    return sizes


def _calibration_from_mapping(mapping: object) -> BoxCalibration:
    if not isinstance(mapping, dict):
        raise RuntimeError("Calibration audit has no calibration mapping")
    try:
        return BoxCalibration(
            width_scale=float(mapping["width_scale"]),
            height_scale=float(mapping["height_scale"]),
            center_x_shift=float(mapping["center_x_shift"]),
            center_y_shift=float(mapping["center_y_shift"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Calibration audit has malformed parameters") from exc


def load_audited_calibration(path: Path) -> tuple[BoxCalibration, dict[str, object]]:
    """Load a stable OOF-selected calibration and reject unaudited parameters."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise RuntimeError("Calibration audit has no contract")
    required_contract = {
        "cache_only": True,
        "single_checkpoint_multiscale": True,
        "global_calibration_only": True,
        "joint_scale_and_shift_search": False,
    }
    for key, expected in required_contract.items():
        if contract.get(key) is not expected:
            raise RuntimeError(f"Calibration audit contract mismatch for {key}")
    if payload.get("passes_stability_gate") is not True:
        raise RuntimeError("Calibration audit did not pass its stability gate")

    full_selection = payload.get("full_selection")
    folds = payload.get("folds")
    oof = payload.get("oof")
    if not isinstance(full_selection, dict) or not isinstance(folds, list) or not folds:
        raise RuntimeError("Calibration audit has incomplete selection records")
    if not isinstance(oof, dict):
        raise RuntimeError("Calibration audit has no OOF record")
    chosen = _calibration_from_mapping(full_selection.get("chosen"))
    fold_choices = [
        _calibration_from_mapping(fold.get("chosen"))
        for fold in folds
        if isinstance(fold, dict)
    ]
    if len(fold_choices) != len(folds) or any(choice != chosen for choice in fold_choices):
        raise RuntimeError("Calibration audit folds do not unanimously match full selection")

    minimum_oof_gain = float(contract.get("minimum_oof_gain", 0.0))
    oof_delta = float(oof.get("delta_map50_95", float("nan")))
    fold_deltas = [float(fold["delta_map50_95"]) for fold in folds]
    if (
        not np.isfinite(oof_delta)
        or oof_delta < minimum_oof_gain
        or sum(delta > 0.0 for delta in fold_deltas) < 2
        or min(fold_deltas) < -0.0005
    ):
        raise RuntimeError("Calibration audit metrics do not satisfy the stability gate")
    return chosen, payload


def calibrate_submission_frame(
    frame: pd.DataFrame,
    image_sizes: dict[int, tuple[int, int]],
    calibration: BoxCalibration,
) -> pd.DataFrame:
    calibrated = frame.copy(deep=True)
    if list(calibrated.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"Submission columns must be exactly {SUBMISSION_COLUMNS}")
    for image_id, indices in calibrated.groupby("image_id", sort=False).groups.items():
        integer_image_id = int(image_id)
        if integer_image_id not in image_sizes:
            raise ValueError(f"Submission references unknown image_id {integer_image_id}")
        width, height = image_sizes[integer_image_id]
        boxes = calibrated.loc[indices, ["x1", "y1", "x2", "y2"]].to_numpy(
            dtype=np.float32
        )
        transformed, valid = calibrate_xyxy(
            boxes,
            image_width=width,
            image_height=height,
            calibration=calibration,
        )
        if not bool(valid.all()):
            raise ValueError(
                f"Calibration invalidated {int((~valid).sum())} boxes for image {integer_image_id}"
            )
        calibrated.loc[indices, ["x1", "y1", "x2", "y2"]] = transformed
    return calibrated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--expected-input-sha256")
    parser.add_argument("--class-count", type=int, default=18)
    args = parser.parse_args()

    if args.input.resolve() == args.output.resolve():
        parser.error("--output must not overwrite --input")
    if not args.input.is_file():
        raise FileNotFoundError(f"Input submission not found: {args.input.resolve()}")
    input_hash = _sha256(args.input)
    if args.expected_input_sha256:
        expected = args.expected_input_sha256.strip().upper()
        if input_hash != expected:
            raise RuntimeError(
                f"Input SHA-256 mismatch: observed {input_hash}, expected {expected}"
            )

    if not args.audit.is_file():
        raise FileNotFoundError(f"Calibration audit not found: {args.audit.resolve()}")
    audit_hash = _sha256(args.audit)
    calibration, audit = load_audited_calibration(args.audit)
    if _sha256(args.audit) != audit_hash:
        raise RuntimeError("Calibration audit changed while it was being read")
    image_sizes = _image_sizes(args.images)
    source = pd.read_csv(args.input)
    source_issues = validate_submission_frame(source, image_sizes, args.class_count)
    if source_issues:
        raise RuntimeError(f"Input submission is invalid: {source_issues[:10]}")
    output = calibrate_submission_frame(source, image_sizes, calibration)
    output_issues = validate_submission_frame(output, image_sizes, args.class_count)
    if output_issues:
        raise RuntimeError(f"Calibrated submission is invalid: {output_issues[:10]}")

    unchanged_columns = ["id", "image_id", "class_id", "confidence"]
    if not source[unchanged_columns].equals(output[unchanged_columns]):
        raise RuntimeError("Calibration unexpectedly changed identity, class, or confidence")
    coordinates = ["x1", "y1", "x2", "y2"]
    changed_rows = int(
        np.any(
            source[coordinates].to_numpy(dtype=np.float64)
            != output[coordinates].to_numpy(dtype=np.float64),
            axis=1,
        ).sum()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    output_hash = _sha256(args.output)
    payload = {
        "schema_version": 2,
        "single_checkpoint_postprocess": True,
        "calibration_audit": str(args.audit.resolve()),
        "calibration_audit_sha256": audit_hash,
        "audit_oof_delta_map50_95": audit["oof"]["delta_map50_95"],
        "audit_fold_deltas_map50_95": [
            fold["delta_map50_95"] for fold in audit["folds"]
        ],
        "source": str(args.input.resolve()),
        "source_sha256": input_hash,
        "output": str(args.output.resolve()),
        "output_sha256": output_hash,
        "images": len(image_sizes),
        "rows": len(output),
        "changed_coordinate_rows": changed_rows,
        "calibration": asdict(calibration),
        "calibration_name": box_calibration_name(calibration),
        "preserved_columns": unchanged_columns,
        "validation_issues": output_issues,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
