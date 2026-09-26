from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image
import pytest

from hsi_detection.submission import SUBMISSION_COLUMNS
from scripts.merge_phase2_submission import build_phase2_submission


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _write_images(directory: Path, image_ids: tuple[int, ...]) -> None:
    directory.mkdir(parents=True)
    for image_id in image_ids:
        Image.new("RGB", (10, 8)).save(directory / f"{image_id}.png")


def _write_submission(path: Path, image_ids: tuple[int, ...]) -> None:
    rows = [
        [index, image_id, 2, 0.9, 1.0, 1.0, 9.0, 7.0]
        for index, image_id in enumerate(image_ids)
    ]
    pd.DataFrame(rows, columns=SUBMISSION_COLUMNS).to_csv(path, index=False)


def test_build_phase2_submission_enforces_lineage_coverage_and_reindexing(
    tmp_path: Path,
) -> None:
    test_images = tmp_path / "test_images"
    ranking_images = tmp_path / "ranking_images"
    _write_images(test_images, (1, 2))
    _write_images(ranking_images, (101, 102))
    test_csv = tmp_path / "test.csv"
    ranking_csv = tmp_path / "ranking.csv"
    _write_submission(test_csv, (1, 2))
    _write_submission(ranking_csv, (101, 102))
    phase1_manifest = tmp_path / "phase1_manifest.json"
    phase1_manifest.write_text(
        json.dumps({"output_sha256": _sha256(test_csv)}), encoding="utf-8"
    )
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"one checkpoint")
    output = tmp_path / "phase2.csv"
    manifest = tmp_path / "phase2.json"

    document = build_phase2_submission(
        test_csv=test_csv,
        ranking_csv=ranking_csv,
        test_images=test_images,
        ranking_images=ranking_images,
        output=output,
        manifest=manifest,
        checkpoint=checkpoint,
        expected_checkpoint_sha256=_sha256(checkpoint),
        phase1_manifest=phase1_manifest,
        ranking_command="python scripts/predict_submission.py --weights last.pt",
        expected_test_images=2,
        expected_ranking_images=2,
    )

    combined = pd.read_csv(output)
    assert combined["id"].tolist() == [0, 1, 2, 3]
    assert combined["image_id"].tolist() == [1, 2, 101, 102]
    assert document["single_checkpoint"] is True
    assert document["combined"]["images"] == 4
    assert document["combined"]["test_ranking_overlap"] == 0
    assert document["combined"]["csv_sha256"] == _sha256(output)


def test_build_phase2_submission_rejects_cross_set_overlap(tmp_path: Path) -> None:
    test_images = tmp_path / "test_images"
    ranking_images = tmp_path / "ranking_images"
    _write_images(test_images, (1,))
    _write_images(ranking_images, (1,))
    test_csv = tmp_path / "test.csv"
    ranking_csv = tmp_path / "ranking.csv"
    _write_submission(test_csv, (1,))
    _write_submission(ranking_csv, (1,))
    phase1_manifest = tmp_path / "phase1_manifest.json"
    phase1_manifest.write_text(
        json.dumps({"output_sha256": _sha256(test_csv)}), encoding="utf-8"
    )
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"one checkpoint")

    with pytest.raises(ValueError, match="overlap"):
        build_phase2_submission(
            test_csv=test_csv,
            ranking_csv=ranking_csv,
            test_images=test_images,
            ranking_images=ranking_images,
            output=tmp_path / "phase2.csv",
            manifest=tmp_path / "phase2.json",
            checkpoint=checkpoint,
            expected_checkpoint_sha256=_sha256(checkpoint),
            phase1_manifest=phase1_manifest,
            ranking_command="predict",
            expected_test_images=1,
            expected_ranking_images=1,
        )
