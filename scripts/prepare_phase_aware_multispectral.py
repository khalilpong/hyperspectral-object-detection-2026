"""Prepare phase-aware 16-band inputs from the 4x4 snapshot mosaic.

The ordinary HSI16 route unpacks each 4x4 cell to one compact 16-channel
pixel.  A later resize then treats all 16 measurements as if they were taken
at the same spatial location.  This route keeps the physical row/column phase
of every band and bilinearly reconstructs it on the image geometry that the
1024-pixel detector will actually see.

Only the input representation changes:

* the official train/validation manifest is reused;
* labels are generated with the same sanitation and normalized-coordinate
  implementation as the ordinary HSI16 dataset;
* the 16-channel order and per-image shared P0.5-P99.5 scale are unchanged;
* the percentile bounds are measured on the native unpacked cube, then reused
  after reconstruction so interpolation does not silently change exposure
  normalization;
* source aspect ratio is preserved and the long edge is set to 1024, so the
  normalized YOLO labels remain valid and Ultralytics owns the final padding.

This is phase-faithful interpolation, not super-resolution.  It cannot invent
unmeasured high-frequency information.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import yaml
from tqdm import tqdm

from hsi_detection.annotations import read_voc_annotation, sanitize_annotation, to_yolo_rows
from hsi_detection.layout import discover_layout, read_classes
from hsi_detection.spectral import (
    encode_multispectral_uint8,
    multispectral_percentile_bounds,
    phase_aware_reconstruct,
    phase_aware_target_shape,
    x2cube,
)


CELL_SIZE = 4
DEFAULT_BAND_ORDER = (5, 8, 13, 0, 1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15)
METHOD = "phase_aware_bilinear_v1"
SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load_manifest(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["image_id", "split"]:
            raise ValueError(
                "Manifest header must be exactly image_id,split; "
                f"got {reader.fieldnames}"
            )
        split_by_id: dict[str, str] = {}
        for row in reader:
            image_id = row["image_id"].strip()
            split = row["split"].strip()
            if not image_id:
                raise ValueError("Manifest contains an empty image_id")
            if image_id in split_by_id:
                raise ValueError(f"Manifest contains duplicate image_id {image_id}")
            if split not in {"train", "val"}:
                raise ValueError(f"Manifest split must be train or val, got {split!r}")
            split_by_id[image_id] = split
    if not split_by_id:
        raise ValueError("Manifest is empty")
    return split_by_id


def _contract(
    *,
    target_long_edge: int,
    lower_percentile: float,
    upper_percentile: float,
    limit: int,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "cell_size": CELL_SIZE,
        "mosaic_band_order": "row-major",
        "output_band_order": list(DEFAULT_BAND_ORDER),
        "target_long_edge": target_long_edge,
        "target_geometry": "preserve_aspect_ratio_no_padding",
        "coordinate_convention": "pixel_centers_align_corners_false",
        "interpolation": "bilinear_per_band_phase_aware",
        "border_mode": "replicate",
        "encoding_dtype": "uint8",
        "encoding_scale": "per_image_shared_native_cube_bounds",
        "lower_percentile": lower_percentile,
        "upper_percentile": upper_percentile,
        "limit_per_split": limit,
    }


def _ensure_contract(output: Path, contract: dict[str, object]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / "preparation_config.json"
    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing != contract:
            raise ValueError(
                "Output already has a different phase-aware preparation contract; "
                "use a new output directory instead of mixing encodings"
            )
        return

    image_root = output / "images"
    if image_root.exists() and next(image_root.rglob("*.npy"), None) is not None:
        raise ValueError(
            "Output contains NPY files but no preparation_config.json; refusing an unsafe resume"
        )
    temporary = contract_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, contract_path)


def _existing_output_is_valid(
    destination_npy: Path,
    destination_png: Path,
    expected_shape: tuple[int, int, int],
) -> bool:
    if not destination_npy.exists() or not destination_png.exists():
        return False
    try:
        existing = np.load(destination_npy, mmap_mode="r", allow_pickle=False)
        if existing.shape != expected_shape or existing.dtype != np.uint8:
            return False
        with Image.open(destination_png) as preview:
            if preview.mode != "RGB" or preview.size != (expected_shape[1], expected_shape[0]):
                return False
    except (OSError, ValueError):
        return False
    return True


def _write_phase_aware_image(
    source: Path,
    destination_png: Path,
    lower_percentile: float,
    upper_percentile: float,
    target_long_edge: int,
) -> dict[str, int | float | bool | None]:
    with Image.open(source) as image:
        raw = np.asarray(image)
    if raw.ndim != 2:
        raise ValueError(f"Expected one-channel mosaic at {source}, got shape {raw.shape}")
    source_height, source_width = raw.shape
    target_height, target_width = phase_aware_target_shape(
        raw.shape, target_long_edge=target_long_edge
    )
    destination_npy = destination_png.with_suffix(".npy")
    expected_shape = (target_height, target_width, len(DEFAULT_BAND_ORDER))
    if _existing_output_is_valid(destination_npy, destination_png, expected_shape):
        return {
            "skipped": True,
            "low": None,
            "high": None,
            "source_height": source_height,
            "source_width": source_width,
            "target_height": target_height,
            "target_width": target_width,
            "bytes": destination_npy.stat().st_size,
        }

    native_cube = x2cube(raw, cell_size=CELL_SIZE)
    low, high = multispectral_percentile_bounds(
        native_cube,
        band_order=DEFAULT_BAND_ORDER,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )
    reconstructed = phase_aware_reconstruct(
        raw,
        target_shape=(target_height, target_width),
        band_order=DEFAULT_BAND_ORDER,
        cell_size=CELL_SIZE,
    )
    encoded = encode_multispectral_uint8(reconstructed, low=low, high=high)
    if encoded.shape != expected_shape or encoded.dtype != np.uint8:
        raise RuntimeError(
            f"Unexpected encoded output for {source}: {encoded.shape} {encoded.dtype}"
        )

    destination_png.parent.mkdir(parents=True, exist_ok=True)
    temporary_npy = destination_npy.with_suffix(".npy.tmp")
    temporary_png = destination_png.with_suffix(".png.tmp")
    with temporary_npy.open("wb") as handle:
        np.save(handle, encoded, allow_pickle=False)
    Image.fromarray(encoded[:, :, :3], mode="RGB").save(
        temporary_png, format="PNG", compress_level=3
    )
    os.replace(temporary_npy, destination_npy)
    os.replace(temporary_png, destination_png)
    return {
        "skipped": False,
        "low": low,
        "high": high,
        "source_height": source_height,
        "source_width": source_width,
        "target_height": target_height,
        "target_width": target_width,
        "bytes": destination_npy.stat().st_size,
    }


def _dataset_yaml(
    output: Path,
    classes: list[str],
    train: object,
    scope: str,
    contract: dict[str, object],
) -> dict[str, object]:
    return {
        "path": str(output.resolve()).replace("\\", "/"),
        "train": train,
        "val": "images/val",
        "test": "images/test",
        "channels": len(DEFAULT_BAND_ORDER),
        "names": {index: name for index, name in enumerate(classes)},
        "hsi_band_order": list(DEFAULT_BAND_ORDER),
        "hsi_encoding": (
            f"per-image shared native-cube {contract['lower_percentile']:g}th-to-"
            f"{contract['upper_percentile']:g}th-percentile uint8"
        ),
        "phase_aware": True,
        "phase_method": contract["method"],
        "phase_cell_size": CELL_SIZE,
        "phase_target_long_edge": contract["target_long_edge"],
        "phase_target_geometry": contract["target_geometry"],
        "phase_coordinate_convention": contract["coordinate_convention"],
        "phase_border_mode": contract["border_mode"],
        "split_seed": 2026,
        "training_scope": scope,
    }


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p50": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "p50": float(np.median(array)),
        "max": float(array.max()),
    }


def _select_ids_by_split(
    split_by_id: dict[str, str], limit: int
) -> tuple[list[str], list[str]]:
    train_ids = sorted(image_id for image_id, split in split_by_id.items() if split == "train")
    val_ids = sorted(image_id for image_id, split in split_by_id.items() if split == "val")
    if limit:
        train_ids = train_ids[:limit]
        val_ids = val_ids[:limit]
    return train_ids, val_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare phase-aware 16-channel NPY inputs for Ultralytics training"
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/extracted"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/pseudo_rgb/split_manifest.csv"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/hsi16_phase_p005_995_l1024")
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lower-percentile", type=float, default=0.5)
    parser.add_argument("--upper-percentile", type=float, default=99.5)
    parser.add_argument("--target-long-edge", type=int, default=1024)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N images per split for a smoke test; 0 means all images",
    )
    args = parser.parse_args()

    if not 0 <= args.lower_percentile < args.upper_percentile <= 100:
        parser.error("percentiles must satisfy 0 <= lower < upper <= 100")
    if args.target_long_edge <= 0:
        parser.error("--target-long-edge must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.limit < 0:
        parser.error("--limit must be non-negative")

    output = args.output.resolve()
    contract = _contract(
        target_long_edge=args.target_long_edge,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
        limit=args.limit,
    )
    _ensure_contract(output, contract)

    layout = discover_layout(args.raw_root)
    classes = read_classes(layout.class_file)
    class_to_id = {name: index for index, name in enumerate(classes)}
    split_by_id = _load_manifest(args.manifest)
    annotations = {
        path.stem: read_voc_annotation(path)
        for path in sorted(layout.train_annotations.glob("*.xml"))
    }
    if set(annotations) != set(split_by_id):
        missing_annotations = sorted(set(split_by_id) - set(annotations))[:5]
        missing_manifest = sorted(set(annotations) - set(split_by_id))[:5]
        raise ValueError(
            "Manifest image IDs do not exactly match the official labeled images; "
            f"missing annotations={missing_annotations}, missing manifest rows={missing_manifest}"
        )

    train_ids, val_ids = _select_ids_by_split(split_by_id, args.limit)
    selected_by_split = {"train": train_ids, "val": val_ids}
    jobs: list[tuple[str, Path, Path]] = []
    clipped_boxes = 0
    dropped_boxes = 0
    for split, image_ids in selected_by_split.items():
        for image_id in image_ids:
            annotation = annotations[image_id]
            sanitization = sanitize_annotation(annotation)
            clipped_boxes += sanitization.clipped_objects
            dropped_boxes += sanitization.dropped_objects
            source = layout.train_images / annotation.filename
            if not source.is_file():
                raise FileNotFoundError(f"Missing source image {source}")
            destination = output / "images" / split / annotation.filename
            label_path = output / "labels" / split / f"{image_id}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text(
                "\n".join(to_yolo_rows(sanitization.annotation, class_to_id)) + "\n",
                encoding="utf-8",
            )
            jobs.append((split, source, destination))

    test_sources = sorted(layout.test_images.glob("*.png"))
    if args.limit:
        test_sources = test_sources[: args.limit]
    for source in test_sources:
        jobs.append(("test", source, output / "images" / "test" / source.name))

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                _write_phase_aware_image,
                source,
                destination,
                args.lower_percentile,
                args.upper_percentile,
                args.target_long_edge,
            )
            for _split, source, destination in jobs
        ]
        results = [
            future.result()
            for future in tqdm(
                futures,
                desc="Reconstructing phase-aware HSI16",
                mininterval=10,
            )
        ]

    # Preserve the exact full manifest for production runs.  Smoke datasets get
    # a filtered manifest so their own artifact remains internally consistent.
    manifest_destination = output / "split_manifest.csv"
    if args.limit:
        with manifest_destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_id", "split"])
            for split in ("train", "val"):
                for image_id in selected_by_split[split]:
                    writer.writerow([image_id, split])
    else:
        manifest_destination.write_bytes(args.manifest.read_bytes())

    scope = (
        f"smoke_limit_{args.limit}_per_split"
        if args.limit
        else "fixed_2400_train_600_val"
    )
    dataset = _dataset_yaml(output, classes, "images/train", scope, contract)
    dataset_all = _dataset_yaml(
        output,
        classes,
        ["images/train", "images/val"],
        "all_selected_labeled_images" if args.limit else "all_3000_labeled_images",
        contract,
    )
    (output / "dataset.yaml").write_text(
        yaml.safe_dump(dataset, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (output / "dataset_all.yaml").write_text(
        yaml.safe_dump(dataset_all, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    counts = {
        split: sum(job_split == split for job_split, _source, _destination in jobs)
        for split in ("train", "val", "test")
    }
    for split in ("train", "val"):
        image_stems = {
            path.stem for path in (output / "images" / split).glob("*.npy")
        }
        label_stems = {
            path.stem for path in (output / "labels" / split).glob("*.txt")
        }
        expected_stems = set(selected_by_split[split])
        if image_stems != expected_stems or label_stems != expected_stems:
            raise RuntimeError(
                f"{split} image/label stems do not match the selected manifest IDs"
            )
    if any((output / "labels" / "test").glob("*.txt")):
        raise RuntimeError("Test split must not contain labels")

    generated = [result for result in results if not result["skipped"]]
    report = {
        "images": {
            "total": len(results),
            "generated": len(generated),
            "resumed_existing": len(results) - len(generated),
            **counts,
        },
        "contract": contract,
        "manifest": {
            "source": str(args.manifest.resolve()),
            "source_sha256": _sha256(args.manifest),
            "output_sha256": _sha256(manifest_destination),
        },
        "geometry": {
            "source_height": _summary([float(item["source_height"]) for item in results]),
            "source_width": _summary([float(item["source_width"]) for item in results]),
            "target_height": _summary([float(item["target_height"]) for item in results]),
            "target_width": _summary([float(item["target_width"]) for item in results]),
        },
        "encoding": {
            "channels": len(DEFAULT_BAND_ORDER),
            "band_order": list(DEFAULT_BAND_ORDER),
            "lower_percentile": args.lower_percentile,
            "upper_percentile": args.upper_percentile,
            "shared_scale_across_channels": True,
            "bounds_measured_on_native_cube": True,
            "dtype": "uint8",
            "per_image_low": _summary(
                [float(item["low"]) for item in generated if item["low"] is not None]
            ),
            "per_image_high": _summary(
                [float(item["high"]) for item in generated if item["high"] is not None]
            ),
            "npy_bytes": int(sum(int(item["bytes"]) for item in results)),
        },
        "box_repairs": {"clipped": clipped_boxes, "dropped": dropped_boxes},
    }
    (output / "preparation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"dataset: {(output / 'dataset.yaml').resolve()}")


if __name__ == "__main__":
    main()
