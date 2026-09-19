from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import yaml
from tqdm import tqdm

from hsi_detection.annotations import read_voc_annotation, sanitize_annotation, to_yolo_rows
from hsi_detection.layout import discover_layout, read_classes
from hsi_detection.spectral import make_multispectral_uint8, x2cube


DEFAULT_BAND_ORDER = (5, 8, 13, 0, 1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15)


def _load_manifest(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["image_id"]: row["split"] for row in csv.DictReader(handle)}


def _write_encoded_image(
    source: Path,
    destination_png: Path,
    band_order: tuple[int, ...],
    lower_percentile: float,
    upper_percentile: float,
) -> dict[str, float | bool]:
    destination_npy = destination_png.with_suffix(".npy")
    if destination_png.exists() and destination_npy.exists():
        try:
            existing = np.load(destination_npy, mmap_mode="r")
            if existing.ndim == 3 and existing.shape[2] == len(band_order) and existing.dtype == np.uint8:
                return {"skipped": True, "low": 0.0, "high": 0.0}
        except (OSError, ValueError):
            pass

    with Image.open(source) as image:
        cube = x2cube(np.asarray(image))
    selected = cube[:, :, band_order]
    low, high = np.percentile(selected, [lower_percentile, upper_percentile])
    encoded = make_multispectral_uint8(
        cube,
        band_order=band_order,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )
    destination_png.parent.mkdir(parents=True, exist_ok=True)
    temporary_npy = destination_npy.with_suffix(".npy.tmp")
    temporary_png = destination_png.with_suffix(".png.tmp")
    with temporary_npy.open("wb") as handle:
        np.save(handle, encoded, allow_pickle=False)
    Image.fromarray(encoded[:, :, :3], mode="RGB").save(temporary_png, format="PNG", compress_level=3)
    os.replace(temporary_npy, destination_npy)
    os.replace(temporary_png, destination_png)
    return {"skipped": False, "low": float(low), "high": float(high)}


def _dataset_yaml(
    output: Path,
    classes: list[str],
    train: object,
    scope: str,
    lower_percentile: float,
    upper_percentile: float,
) -> dict[str, object]:
    return {
        "path": str(output.resolve()).replace("\\", "/"),
        "train": train,
        "val": "images/val",
        "test": "images/test",
        "channels": 16,
        "names": {index: name for index, name in enumerate(classes)},
        "hsi_band_order": list(DEFAULT_BAND_ORDER),
        "hsi_encoding": (
            f"per-image shared {lower_percentile:g}th-to-{upper_percentile:g}th-percentile uint8"
        ),
        "split_seed": 2026,
        "training_scope": scope,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 16-channel NPY inputs for native Ultralytics training")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/extracted"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/pseudo_rgb/split_manifest.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed/hsi16_shared"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lower-percentile", type=float, default=1.0)
    parser.add_argument("--upper-percentile", type=float, default=99.0)
    args = parser.parse_args()

    output = args.output.resolve()
    layout = discover_layout(args.raw_root)
    classes = read_classes(layout.class_file)
    class_to_id = {name: index for index, name in enumerate(classes)}
    split_by_id = _load_manifest(args.manifest)
    annotations = {
        path.stem: read_voc_annotation(path)
        for path in sorted(layout.train_annotations.glob("*.xml"))
    }
    if set(annotations) != set(split_by_id):
        raise ValueError("Manifest image IDs do not exactly match the official labeled images")

    output.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[Path, Path]] = []
    clipped_boxes = 0
    dropped_boxes = 0
    for image_id, annotation in annotations.items():
        sanitization = sanitize_annotation(annotation)
        clipped_boxes += sanitization.clipped_objects
        dropped_boxes += sanitization.dropped_objects
        split = split_by_id[image_id]
        destination = output / "images" / split / annotation.filename
        label_path = output / "labels" / split / f"{image_id}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(
            "\n".join(to_yolo_rows(sanitization.annotation, class_to_id)) + "\n",
            encoding="utf-8",
        )
        jobs.append((layout.train_images / annotation.filename, destination))
    for source in sorted(layout.test_images.glob("*.png")):
        jobs.append((source, output / "images" / "test" / source.name))

    worker_count = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _write_encoded_image,
                source,
                destination,
                DEFAULT_BAND_ORDER,
                args.lower_percentile,
                args.upper_percentile,
            )
            for source, destination in jobs
        ]
        results = [future.result() for future in tqdm(futures, desc="Encoding 16-channel cubes")]

    manifest_destination = output / "split_manifest.csv"
    manifest_destination.write_text(args.manifest.read_text(encoding="utf-8"), encoding="utf-8")
    dataset = _dataset_yaml(
        output,
        classes,
        "images/train",
        "fixed_2400_train_600_val",
        args.lower_percentile,
        args.upper_percentile,
    )
    dataset_all = _dataset_yaml(
        output,
        classes,
        ["images/train", "images/val"],
        "all_3000_labeled_images",
        args.lower_percentile,
        args.upper_percentile,
    )
    (output / "dataset.yaml").write_text(
        yaml.safe_dump(dataset, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (output / "dataset_all.yaml").write_text(
        yaml.safe_dump(dataset_all, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    generated = [result for result in results if not result["skipped"]]
    report = {
        "images": {
            "total": len(results),
            "generated": len(generated),
            "resumed_existing": len(results) - len(generated),
            "train": sum(split == "train" for split in split_by_id.values()),
            "val": sum(split == "val" for split in split_by_id.values()),
            "test": len(results) - len(split_by_id),
        },
        "encoding": {
            "channels": len(DEFAULT_BAND_ORDER),
            "band_order": list(DEFAULT_BAND_ORDER),
            "lower_percentile": args.lower_percentile,
            "upper_percentile": args.upper_percentile,
            "shared_scale_across_channels": True,
            "dtype": "uint8",
            "per_image_low": _summary([float(item["low"]) for item in generated]),
            "per_image_high": _summary([float(item["high"]) for item in generated]),
        },
        "box_repairs": {"clipped": clipped_boxes, "dropped": dropped_boxes},
    }
    (output / "preparation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"dataset: {(output / 'dataset.yaml').resolve()}")


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p50": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "p50": float(np.median(array)),
        "max": float(array.max()),
    }


if __name__ == "__main__":
    main()
