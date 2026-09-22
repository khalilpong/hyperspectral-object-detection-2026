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
from hsi_detection.spectral import make_multispectral_uint8, x2cube


DEFAULT_BAND_ORDER = (5, 8, 13, 0, 1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15)
METHOD = "compact_4x4_unpack_shared_scale_v1"
SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _parse_band_order(text: str) -> tuple[int, ...]:
    try:
        bands = tuple(int(value.strip()) for value in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "band order must be 16 comma-separated integers"
        ) from error
    if len(bands) != 16:
        raise argparse.ArgumentTypeError(
            f"band order must contain exactly 16 entries, got {len(bands)}"
        )
    if set(bands) != set(range(16)):
        raise argparse.ArgumentTypeError(
            "band order must be a permutation of the physical bands 0..15"
        )
    return bands


def _preparation_contract(
    *,
    band_order: tuple[int, ...],
    lower_percentile: float,
    upper_percentile: float,
    manifest_path: Path,
    split_by_id: dict[str, str],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "cell_size": 4,
        "mosaic_band_order": "row-major",
        "output_band_order": list(band_order),
        "channels": len(band_order),
        "encoding_dtype": "uint8",
        "encoding_scale": "per_image_shared_selected_band_bounds",
        "lower_percentile": lower_percentile,
        "upper_percentile": upper_percentile,
        "split_manifest_sha256": _sha256(manifest_path),
        "split_counts": {
            "train": sum(split == "train" for split in split_by_id.values()),
            "val": sum(split == "val" for split in split_by_id.values()),
        },
    }


def _ensure_preparation_contract(output: Path, contract: dict[str, object]) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / "preparation_config.json"
    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing != contract:
            raise ValueError(
                "Output already has a different ordinary-HSI16 preparation contract; "
                "use a new output directory instead of mixing band orders or encodings"
            )
        return contract_path

    image_root = output / "images"
    if image_root.exists() and next(image_root.rglob("*.npy"), None) is not None:
        raise ValueError(
            "Output contains NPY files but no preparation_config.json; refusing an unsafe resume"
        )
    temporary = contract_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, contract_path)
    return contract_path


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
    band_order: tuple[int, ...],
    lower_percentile: float,
    upper_percentile: float,
) -> dict[str, object]:
    return {
        "path": str(output.resolve()).replace("\\", "/"),
        "train": train,
        "val": "images/val",
        "test": "images/test",
        "channels": len(band_order),
        "names": {index: name for index, name in enumerate(classes)},
        "hsi_band_order": list(band_order),
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
    parser.add_argument(
        "--band-order",
        type=_parse_band_order,
        default=DEFAULT_BAND_ORDER,
        help="16 comma-separated physical band indices; default preserves the baseline order",
    )
    args = parser.parse_args()

    if not 0 <= args.lower_percentile < args.upper_percentile <= 100:
        parser.error("percentiles must satisfy 0 <= lower < upper <= 100")
    if args.workers <= 0:
        parser.error("--workers must be positive")

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

    contract = _preparation_contract(
        band_order=args.band_order,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
        manifest_path=args.manifest,
        split_by_id=split_by_id,
    )
    contract_path = _ensure_preparation_contract(output, contract)
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

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                _write_encoded_image,
                source,
                destination,
                args.band_order,
                args.lower_percentile,
                args.upper_percentile,
            )
            for source, destination in jobs
        ]
        results = [future.result() for future in tqdm(futures, desc="Encoding 16-channel cubes")]

    manifest_destination = output / "split_manifest.csv"
    manifest_destination.write_bytes(args.manifest.read_bytes())
    dataset = _dataset_yaml(
        output,
        classes,
        "images/train",
        "fixed_2400_train_600_val",
        args.band_order,
        args.lower_percentile,
        args.upper_percentile,
    )
    dataset_all = _dataset_yaml(
        output,
        classes,
        ["images/train", "images/val"],
        "all_3000_labeled_images",
        args.band_order,
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
        "contract": contract,
        "manifest": {
            "source_sha256": _sha256(args.manifest),
            "output_sha256": _sha256(manifest_destination),
        },
        "encoding": {
            "channels": len(args.band_order),
            "band_order": list(args.band_order),
            "lower_percentile": args.lower_percentile,
            "upper_percentile": args.upper_percentile,
            "shared_scale_across_channels": True,
            "dtype": "uint8",
            "per_image_low": _summary([float(item["low"]) for item in generated]),
            "per_image_high": _summary([float(item["high"]) for item in generated]),
        },
        "preparation_config_sha256": _sha256(contract_path),
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
