from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import sys

import numpy as np
import pandas as pd
from PIL import Image

from hsi_detection.submission import SUBMISSION_COLUMNS, validate_submission_frame


DEFAULT_CHECKPOINT_SHA256 = (
    "8E4BFBF7AC36BAA55274C464EF3A8BC6581C1450601AB4DFB97D3D8FAA4254C3"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _image_sizes(directory: Path, *, expected_count: int, label: str) -> dict[int, tuple[int, int]]:
    if expected_count <= 0:
        raise ValueError("expected image count must be positive")
    paths = sorted(directory.glob("*.png"))
    sizes: dict[int, tuple[int, int]] = {}
    for path in paths:
        if not path.stem.isdigit():
            raise ValueError(f"{label} image stem must be numeric: {path.name}")
        image_id = int(path.stem)
        if image_id in sizes:
            raise ValueError(f"{label} has duplicate numeric image_id {image_id}")
        with Image.open(path) as image:
            width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"{label} image has invalid dimensions: {path}")
        sizes[image_id] = (width, height)
    if len(sizes) != expected_count:
        raise ValueError(
            f"Expected {expected_count} {label} PNGs, found {len(sizes)} in {directory}"
        )
    return sizes


def _submission_image_ids(frame: pd.DataFrame, *, label: str) -> set[int]:
    numeric = pd.to_numeric(frame["image_id"], errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{label} image_id values must be finite integers")
    return {int(value) for value in values}


def _validated_submission(
    path: Path,
    image_sizes: dict[int, tuple[int, int]],
    *,
    class_count: int,
    label: str,
) -> tuple[pd.DataFrame, set[int]]:
    frame = pd.read_csv(path)
    if list(frame.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"{label} columns must be exactly {SUBMISSION_COLUMNS}")
    issues = validate_submission_frame(frame, image_sizes, class_count)
    if issues:
        preview = "; ".join(issues[:10])
        raise ValueError(f"{label} failed submission validation: {preview}")
    image_ids = _submission_image_ids(frame, label=label)
    expected_ids = set(image_sizes)
    if image_ids != expected_ids:
        missing = sorted(expected_ids - image_ids)[:10]
        unexpected = sorted(image_ids - expected_ids)[:10]
        raise ValueError(
            f"{label} coverage mismatch: {len(image_ids)}/{len(expected_ids)} IDs; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return frame, image_ids


def _load_lineage_manifest(path: Path, expected_output_sha256: str) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    recorded = str(document.get("output_sha256", "")).upper()
    if recorded != expected_output_sha256:
        raise ValueError(
            "Phase 1 lineage manifest output_sha256 does not match the test CSV"
        )
    return document


def build_phase2_submission(
    *,
    test_csv: Path,
    ranking_csv: Path,
    test_images: Path,
    ranking_images: Path,
    output: Path,
    manifest: Path,
    checkpoint: Path,
    expected_checkpoint_sha256: str,
    phase1_manifest: Path,
    ranking_command: str,
    expected_test_images: int = 1000,
    expected_ranking_images: int = 1000,
    class_count: int = 18,
) -> dict[str, object]:
    for source in (test_csv, ranking_csv, checkpoint, phase1_manifest):
        if not source.is_file():
            raise FileNotFoundError(source)
    if output.resolve() in {test_csv.resolve(), ranking_csv.resolve()}:
        raise ValueError("Output must not overwrite either input CSV")
    if output.exists() or manifest.exists():
        raise FileExistsError("Output CSV and manifest must be new, non-existing paths")
    if not ranking_command.strip():
        raise ValueError("ranking_command must record the exact inference command")
    expected_hash = expected_checkpoint_sha256.strip().upper()
    if not re.fullmatch(r"[0-9A-F]{64}", expected_hash):
        raise ValueError("expected checkpoint SHA-256 must be 64 hexadecimal characters")
    checkpoint_sha256 = _sha256(checkpoint)
    if checkpoint_sha256 != expected_hash:
        raise ValueError(
            f"Checkpoint SHA-256 mismatch: expected {expected_hash}, got {checkpoint_sha256}"
        )

    test_sizes = _image_sizes(
        test_images, expected_count=expected_test_images, label="test"
    )
    ranking_sizes = _image_sizes(
        ranking_images, expected_count=expected_ranking_images, label="ranking"
    )
    overlap = set(test_sizes) & set(ranking_sizes)
    if overlap:
        raise ValueError(f"test/ranking image_id overlap: {sorted(overlap)[:10]}")
    test_frame, test_ids = _validated_submission(
        test_csv, test_sizes, class_count=class_count, label="test CSV"
    )
    ranking_frame, ranking_ids = _validated_submission(
        ranking_csv, ranking_sizes, class_count=class_count, label="ranking CSV"
    )
    test_sha256 = _sha256(test_csv)
    ranking_sha256 = _sha256(ranking_csv)
    _load_lineage_manifest(phase1_manifest, test_sha256)

    combined = pd.concat([test_frame, ranking_frame], ignore_index=True)
    combined["id"] = np.arange(len(combined), dtype=np.int64)
    combined_sizes = {**test_sizes, **ranking_sizes}
    issues = validate_submission_frame(combined, combined_sizes, class_count)
    if issues:
        raise RuntimeError(f"Combined submission is invalid: {'; '.join(issues[:10])}")
    combined_ids = _submission_image_ids(combined, label="combined CSV")
    if combined_ids != test_ids | ranking_ids:
        raise RuntimeError("Combined CSV coverage changed during concatenation")

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    combined.to_csv(temporary_output, index=False)
    os.replace(temporary_output, output)
    output_sha256 = _sha256(output)
    reread = pd.read_csv(output)
    reread_issues = validate_submission_frame(reread, combined_sizes, class_count)
    if reread_issues or _submission_image_ids(reread, label="written combined CSV") != combined_ids:
        raise RuntimeError("Written combined CSV failed the final round-trip checker")

    document: dict[str, object] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase2_combined_submission": True,
        "single_checkpoint": True,
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "bytes": checkpoint.stat().st_size,
            "sha256": checkpoint_sha256,
        },
        "inference": {
            "ranking_command": ranking_command,
            "merge_command": shlex.join(sys.argv),
            "ranking_inference_only": True,
            "model_state_updates": False,
            "different_checkpoint_fusion": False,
        },
        "phase1_lineage_manifest": {
            "path": str(phase1_manifest.resolve()),
            "sha256": _sha256(phase1_manifest),
        },
        "test": {
            "csv": str(test_csv.resolve()),
            "csv_bytes": test_csv.stat().st_size,
            "csv_sha256": test_sha256,
            "rows": len(test_frame),
            "images": len(test_ids),
            "id_min": min(test_ids),
            "id_max": max(test_ids),
        },
        "ranking": {
            "csv": str(ranking_csv.resolve()),
            "csv_bytes": ranking_csv.stat().st_size,
            "csv_sha256": ranking_sha256,
            "rows": len(ranking_frame),
            "images": len(ranking_ids),
            "id_min": min(ranking_ids),
            "id_max": max(ranking_ids),
        },
        "combined": {
            "csv": str(output.resolve()),
            "csv_bytes": output.stat().st_size,
            "csv_sha256": output_sha256,
            "columns": SUBMISSION_COLUMNS,
            "rows": len(combined),
            "images": len(combined_ids),
            "test_ranking_overlap": 0,
            "id_is_consecutive": True,
            "validation_issues": [],
        },
    }
    temporary_manifest = manifest.with_suffix(manifest.suffix + ".tmp")
    temporary_manifest.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, manifest)
    return document


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strictly merge Phase 2 test and ranking predictions with lineage evidence"
    )
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--ranking-csv", type=Path, required=True)
    parser.add_argument("--test-images", type=Path, required=True)
    parser.add_argument("--ranking-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--expected-checkpoint-sha256",
        default=DEFAULT_CHECKPOINT_SHA256,
    )
    parser.add_argument("--phase1-manifest", type=Path, required=True)
    parser.add_argument("--ranking-command", required=True)
    parser.add_argument("--expected-test-images", type=int, default=1000)
    parser.add_argument("--expected-ranking-images", type=int, default=1000)
    parser.add_argument("--class-count", type=int, default=18)
    args = parser.parse_args()
    document = build_phase2_submission(
        test_csv=args.test_csv,
        ranking_csv=args.ranking_csv,
        test_images=args.test_images,
        ranking_images=args.ranking_images,
        output=args.output,
        manifest=args.manifest,
        checkpoint=args.checkpoint,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        phase1_manifest=args.phase1_manifest,
        ranking_command=args.ranking_command,
        expected_test_images=args.expected_test_images,
        expected_ranking_images=args.expected_ranking_images,
        class_count=args.class_count,
    )
    print(json.dumps(document, indent=2))
    print(
        "PHASE2_MERGE_OK: "
        f"{document['combined']['rows']} rows across "
        f"{document['combined']['images']} images"
    )


if __name__ == "__main__":
    main()
