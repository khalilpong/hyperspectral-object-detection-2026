from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import yaml

from scripts import prepare_multispectral


BASELINE_ORDER = (5, 8, 13, 0, 1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15)
TARGET_ORDER = (13, 8, 5, 0, 1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15)


def _write_fixture(raw_root: Path, manifest: Path) -> None:
    train_images = raw_root / "data_train" / "VIS"
    annotations = raw_root / "data_train" / "Annotations" / "VIS"
    test_images = raw_root / "data_test" / "VIS"
    train_images.mkdir(parents=True)
    annotations.mkdir(parents=True)
    test_images.mkdir(parents=True)
    (raw_root / "class.txt").write_text("car\n", encoding="utf-8")
    manifest.write_text(
        "image_id,split\ntrain_a,train\ntrain_b,val\n", encoding="utf-8"
    )

    for offset, image_id in enumerate(("train_a", "train_b", "test_a")):
        mosaic = (np.arange(64, dtype=np.uint16).reshape(8, 8) + offset * 100)
        destination = (
            test_images if image_id == "test_a" else train_images
        ) / f"{image_id}.png"
        Image.fromarray(mosaic).save(destination)
        if image_id != "test_a":
            (annotations / f"{image_id}.xml").write_text(
                "<annotation>"
                f"<filename>{image_id}.png</filename>"
                "<size><width>2</width><height>2</height><depth>16</depth></size>"
                "<object><name>car</name><bndbox>"
                "<xmin>0</xmin><ymin>0</ymin><xmax>1</xmax><ymax>1</ymax>"
                "</bndbox></object>"
                "</annotation>",
                encoding="utf-8",
            )


def _run_prepare(
    monkeypatch: pytest.MonkeyPatch,
    *,
    raw_root: Path,
    manifest: Path,
    output: Path,
    band_order: tuple[int, ...],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_multispectral.py",
            "--raw-root",
            str(raw_root),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--workers",
            "1",
            "--lower-percentile",
            "0",
            "--upper-percentile",
            "100",
            "--band-order",
            ",".join(str(band) for band in band_order),
        ],
    )
    prepare_multispectral.main()


def test_parse_band_order_accepts_only_a_complete_permutation() -> None:
    assert prepare_multispectral._parse_band_order(
        ",".join(str(band) for band in TARGET_ORDER)
    ) == TARGET_ORDER

    for invalid in (
        "0,1,2",
        "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,14",
        "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16",
        "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,x",
    ):
        with pytest.raises(argparse.ArgumentTypeError):
            prepare_multispectral._parse_band_order(invalid)


def test_custom_order_changes_only_channel_positions_and_preserves_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_root = tmp_path / "raw"
    manifest = tmp_path / "split_manifest.csv"
    _write_fixture(raw_root, manifest)
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"

    _run_prepare(
        monkeypatch,
        raw_root=raw_root,
        manifest=manifest,
        output=baseline,
        band_order=BASELINE_ORDER,
    )
    _run_prepare(
        monkeypatch,
        raw_root=raw_root,
        manifest=manifest,
        output=candidate,
        band_order=TARGET_ORDER,
    )

    baseline_array = np.load(baseline / "images" / "train" / "train_a.npy")
    candidate_array = np.load(candidate / "images" / "train" / "train_a.npy")
    candidate_indices_in_baseline = [BASELINE_ORDER.index(band) for band in TARGET_ORDER]
    np.testing.assert_array_equal(
        candidate_array, baseline_array[:, :, candidate_indices_in_baseline]
    )
    assert (
        baseline / "labels" / "train" / "train_a.txt"
    ).read_bytes() == (
        candidate / "labels" / "train" / "train_a.txt"
    ).read_bytes()
    assert (
        baseline / "labels" / "val" / "train_b.txt"
    ).read_bytes() == (
        candidate / "labels" / "val" / "train_b.txt"
    ).read_bytes()
    assert (baseline / "split_manifest.csv").read_bytes() == manifest.read_bytes()
    assert (candidate / "split_manifest.csv").read_bytes() == manifest.read_bytes()

    baseline_report = json.loads(
        (baseline / "preparation_report.json").read_text(encoding="utf-8")
    )
    candidate_report = json.loads(
        (candidate / "preparation_report.json").read_text(encoding="utf-8")
    )
    assert baseline_report["encoding"]["per_image_low"] == candidate_report[
        "encoding"
    ]["per_image_low"]
    assert baseline_report["encoding"]["per_image_high"] == candidate_report[
        "encoding"
    ]["per_image_high"]
    assert candidate_report["encoding"]["band_order"] == list(TARGET_ORDER)
    dataset = yaml.safe_load((candidate / "dataset.yaml").read_text(encoding="utf-8"))
    assert dataset["hsi_band_order"] == list(TARGET_ORDER)
    assert dataset["channels"] == 16
    assert dataset["training_scope"] == "fixed_2400_train_600_val"


def test_preparation_contract_rejects_band_order_drift_and_unsafe_resume(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "split_manifest.csv"
    manifest.write_text("image_id,split\na,train\nb,val\n", encoding="utf-8")
    split_by_id = {"a": "train", "b": "val"}
    baseline_contract = prepare_multispectral._preparation_contract(
        band_order=BASELINE_ORDER,
        lower_percentile=0.5,
        upper_percentile=99.5,
        manifest_path=manifest,
        split_by_id=split_by_id,
    )
    candidate_contract = prepare_multispectral._preparation_contract(
        band_order=TARGET_ORDER,
        lower_percentile=0.5,
        upper_percentile=99.5,
        manifest_path=manifest,
        split_by_id=split_by_id,
    )

    output = tmp_path / "contracted"
    prepare_multispectral._ensure_preparation_contract(output, baseline_contract)
    prepare_multispectral._ensure_preparation_contract(output, baseline_contract)
    with pytest.raises(ValueError, match="different ordinary-HSI16"):
        prepare_multispectral._ensure_preparation_contract(output, candidate_contract)

    unsafe = tmp_path / "unsafe"
    image_dir = unsafe / "images" / "train"
    image_dir.mkdir(parents=True)
    np.save(image_dir / "a.npy", np.zeros((1, 1, 16), dtype=np.uint8))
    with pytest.raises(ValueError, match="unsafe resume"):
        prepare_multispectral._ensure_preparation_contract(
            unsafe, candidate_contract
        )
