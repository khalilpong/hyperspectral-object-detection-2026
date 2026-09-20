from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

from hsi_detection.tiling import (
    LabeledBox,
    axis_starts,
    crop_labeled_boxes,
    map_tile_boxes_to_image,
    parse_yolo_rows,
    tile_windows,
    to_yolo_rows,
)
from scripts import prepare_object_crops


def test_axis_starts_cover_far_edge_without_gaps() -> None:
    assert axis_starts(495, 256, 192) == (0, 192, 239)
    assert axis_starts(249, 128, 96) == (0, 96, 121)
    assert axis_starts(100, 128, 96) == (0,)
    with pytest.raises(ValueError, match="coverage would have gaps"):
        axis_starts(500, 128, 129)


def test_tile_cores_assign_overlap_centers_to_exactly_one_tile() -> None:
    windows = tile_windows(
        249,
        495,
        tile_height=128,
        tile_width=256,
        stride_height=96,
        stride_width=192,
    )
    assert len(windows) == 9
    assert {(window.x0, window.y0) for window in windows} == {
        (x, y) for y in (0, 96, 121) for x in (0, 192, 239)
    }

    global_box = np.asarray([[220.0, 110.0, 240.0, 130.0]], dtype=np.float32)
    owners = 0
    for window in windows:
        local_box = global_box - np.asarray(
            [window.x0, window.y0, window.x0, window.y0], dtype=np.float32
        )
        _, keep = map_tile_boxes_to_image(local_box, window)
        owners += int(keep[0])
    assert owners == 1


def test_crop_labels_keeps_only_sufficiently_visible_boxes() -> None:
    boxes = [
        LabeledBox(1, 50.0, 40.0, 70.0, 60.0),
        LabeledBox(2, 5.0, 40.0, 25.0, 60.0),
    ]
    cropped = crop_labeled_boxes(
        boxes,
        x0=20,
        y0=20,
        crop_width=80,
        crop_height=80,
        minimum_visible=0.9,
    )
    assert cropped == [LabeledBox(1, 30.0, 20.0, 50.0, 40.0)]
    rows = to_yolo_rows(cropped, width=80, height=80)
    reparsed = parse_yolo_rows("\n".join(rows), width=80, height=80)
    assert len(reparsed) == 1
    assert reparsed[0].class_id == cropped[0].class_id
    assert (reparsed[0].x1, reparsed[0].y1, reparsed[0].x2, reparsed[0].y2) == pytest.approx(
        (cropped[0].x1, cropped[0].y1, cropped[0].x2, cropped[0].y2)
    )


def test_prepare_object_crops_preserves_encoded_values_and_val_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True)
    for split in ("train", "val"):
        (root / "labels" / split).mkdir(parents=True)

    array = np.arange(160 * 300 * 16, dtype=np.uint32).reshape(160, 300, 16).astype(np.uint8)
    np.save(root / "images" / "train" / "1.npy", array, allow_pickle=False)
    (root / "labels" / "train" / "1.txt").write_text(
        "3 0.50000000 0.50000000 0.06666667 0.12500000\n", encoding="utf-8"
    )
    base_config = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "channels": 16,
        "names": {3: "target"},
    }
    (root / "dataset.yaml").write_text(
        yaml.safe_dump(base_config, sort_keys=False), encoding="utf-8"
    )
    all_config = dict(base_config)
    all_config["train"] = ["images/train", "images/val"]
    (root / "dataset_all.yaml").write_text(
        yaml.safe_dump(all_config, sort_keys=False), encoding="utf-8"
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_object_crops.py",
            "--dataset-root",
            str(root),
            "--tile-height",
            "80",
            "--tile-width",
            "120",
            "--jitter-fraction",
            "0",
        ],
    )
    prepare_object_crops.main()

    crop_paths = list((root / "images" / "train_object_crops").glob("*.npy"))
    assert len(crop_paths) == 1
    crop = np.load(crop_paths[0], allow_pickle=False)
    assert crop.shape == (80, 120, 16)
    assert np.array_equal(crop, array[40:120, 90:210])
    crop_label = root / "labels" / "train_object_crops" / f"{crop_paths[0].stem}.txt"
    assert crop_label.read_text(encoding="utf-8").startswith("3 ")

    fixed = yaml.safe_load((root / "dataset_object_crops.yaml").read_text(encoding="utf-8"))
    assert fixed["train"] == ["images/train", "images/train_object_crops"]
    assert fixed["val"] == "images/val"
    report = json.loads((root / "object_crop_report.json").read_text(encoding="utf-8"))
    assert report["generated_crops"] == 1
    assert report["settings"]["renormalized"] is False
