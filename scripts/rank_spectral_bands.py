from __future__ import annotations

import argparse
import csv
import hashlib
from itertools import combinations
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image

from hsi_detection.annotations import read_voc_annotation, sanitize_annotation
from hsi_detection.layout import discover_layout, read_classes
from hsi_detection.spectral import x2cube


def load_split_manifest(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["image_id"]: row["split"] for row in csv.DictReader(handle)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_fingerprint(image_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(image_ids).encode("utf-8")).hexdigest()


def select_image_ids(
    image_ids: list[str],
    annotations: dict[str, object],
    limit: int,
    seed: int,
) -> list[str]:
    """Choose a deterministic, roughly class-balanced image sample."""
    shuffled = image_ids.copy()
    random.Random(seed).shuffle(shuffled)
    selected: list[str] = []
    class_counts: dict[str, int] = {}
    remaining = shuffled.copy()
    while remaining and len(selected) < limit:
        best_id = max(
            remaining,
            key=lambda image_id: sum(
                1.0 / (1.0 + class_counts.get(obj.name, 0))
                for obj in annotations[image_id].objects
            ),
        )
        remaining.remove(best_id)
        selected.append(best_id)
        for obj in annotations[best_id].objects:
            class_counts[obj.name] = class_counts.get(obj.name, 0) + 1
    return selected


def box_spectra(
    cube: np.ndarray,
    obj: object,
    channel_span: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    x1 = max(0, int(np.floor(obj.xmin)))
    y1 = max(0, int(np.floor(obj.ymin)))
    x2 = min(cube.shape[1], int(np.ceil(obj.xmax)))
    y2 = min(cube.shape[0], int(np.ceil(obj.ymax)))
    if x2 <= x1 or y2 <= y1:
        return None

    width = x2 - x1
    height = y2 - y1
    margin_x = int(round(width * 0.2)) if width >= 5 else 0
    margin_y = int(round(height * 0.2)) if height >= 5 else 0
    ix1, ix2 = x1 + margin_x, x2 - margin_x
    iy1, iy2 = y1 + margin_y, y2 - margin_y
    if ix2 <= ix1 or iy2 <= iy1:
        ix1, ix2, iy1, iy2 = x1, x2, y1, y2
    object_pixels = cube[iy1:iy2, ix1:ix2].reshape(-1, cube.shape[2])
    if not len(object_pixels):
        return None
    object_median = np.median(object_pixels, axis=0).astype(np.float64)

    pad_x = max(3, int(round(width * 0.25)))
    pad_y = max(3, int(round(height * 0.25)))
    rx1, rx2 = max(0, x1 - pad_x), min(cube.shape[1], x2 + pad_x)
    ry1, ry2 = max(0, y1 - pad_y), min(cube.shape[0], y2 + pad_y)
    ring = cube[ry1:ry2, rx1:rx2]
    ring_mask = np.ones(ring.shape[:2], dtype=bool)
    ring_mask[y1 - ry1 : y2 - ry1, x1 - rx1 : x2 - rx1] = False
    background_pixels = ring[ring_mask]
    if not len(background_pixels):
        background_pixels = cube.reshape(-1, cube.shape[2])
    background_median = np.median(background_pixels, axis=0).astype(np.float64)
    contrast = np.abs(object_median - background_median) / channel_span
    return object_median, contrast


def collect_features(
    image_ids: list[str],
    annotations: dict[str, object],
    train_images: Path,
    class_to_id: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    contrasts: list[np.ndarray] = []
    labels: list[int] = []
    for image_id in image_ids:
        annotation = sanitize_annotation(annotations[image_id]).annotation
        image_path = train_images / annotation.filename
        with Image.open(image_path) as image:
            cube = x2cube(np.asarray(image))
        low, high = np.percentile(cube, [1.0, 99.0], axis=(0, 1))
        channel_span = np.maximum(high - low, 1.0)
        for obj in annotation.objects:
            spectra = box_spectra(cube, obj, channel_span)
            if spectra is not None:
                feature, contrast = spectra
                features.append(feature)
                contrasts.append(contrast)
                labels.append(class_to_id[obj.name])
    return (
        np.stack(features),
        np.asarray(labels, dtype=np.int64),
        np.stack(contrasts),
    )


def normalized_log_spectra(features: np.ndarray) -> np.ndarray:
    transformed = np.log1p(features)
    return transformed - transformed.mean(axis=1, keepdims=True)


def balanced_centroid_accuracy(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    bands: tuple[int, int, int],
    class_count: int,
) -> float:
    train = train_x[:, bands]
    val = val_x[:, bands]
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-9] = 1.0
    train = (train - mean) / scale
    val = (val - mean) / scale
    centroids = np.stack([train[train_y == cls].mean(axis=0) for cls in range(class_count)])
    predictions = np.square(val[:, None, :] - centroids[None, :, :]).sum(axis=2).argmin(axis=1)
    per_class = [
        float((predictions[val_y == cls] == cls).mean())
        for cls in range(class_count)
        if np.any(val_y == cls)
    ]
    return float(np.mean(per_class))


def balanced_detection_contrast(
    contrasts: np.ndarray,
    labels: np.ndarray,
    bands: tuple[int, int, int],
    class_count: int,
) -> float:
    per_object = np.sqrt(np.square(contrasts[:, bands]).mean(axis=1))
    per_class = [
        float(np.median(per_object[labels == cls]))
        for cls in range(class_count)
        if np.any(labels == cls)
    ]
    return float(np.mean(per_class))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank three-band pseudo-RGB candidates using fixed-split object spectra"
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/extracted"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/pseudo_rgb/split_manifest.csv"),
    )
    parser.add_argument("--train-images", type=int, default=500)
    parser.add_argument("--val-images", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/spectral_band_ranking.json"),
    )
    args = parser.parse_args()

    layout = discover_layout(args.raw_root)
    classes = read_classes(layout.class_file)
    class_to_id = {name: index for index, name in enumerate(classes)}
    annotations = {
        path.stem: read_voc_annotation(path)
        for path in sorted(layout.train_annotations.glob("*.xml"))
    }
    split_by_id = load_split_manifest(args.manifest)
    train_pool = sorted(image_id for image_id, split in split_by_id.items() if split == "train")
    val_pool = sorted(image_id for image_id, split in split_by_id.items() if split == "val")
    train_ids = select_image_ids(
        train_pool, annotations, min(args.train_images, len(train_pool)), args.seed
    )
    val_ids = select_image_ids(
        val_pool, annotations, min(args.val_images, len(val_pool)), args.seed + 1
    )

    train_raw, train_y, _train_contrast = collect_features(
        train_ids, annotations, layout.train_images, class_to_id
    )
    val_raw, val_y, val_contrast = collect_features(
        val_ids, annotations, layout.train_images, class_to_id
    )
    train_x = normalized_log_spectra(train_raw)
    val_x = normalized_log_spectra(val_raw)

    rankings = []
    for bands in combinations(range(train_x.shape[1]), 3):
        accuracy = balanced_centroid_accuracy(
            train_x, train_y, val_x, val_y, bands, len(classes)
        )
        contrast = balanced_detection_contrast(
            val_contrast, val_y, bands, len(classes)
        )
        rankings.append(
            {
                "bands": list(bands),
                "balanced_accuracy": accuracy,
                "balanced_detection_contrast": contrast,
            }
        )
    accuracy_values = np.asarray([row["balanced_accuracy"] for row in rankings])
    contrast_values = np.asarray(
        [row["balanced_detection_contrast"] for row in rankings]
    )
    accuracy_z = (accuracy_values - accuracy_values.mean()) / accuracy_values.std()
    contrast_z = (contrast_values - contrast_values.mean()) / contrast_values.std()
    for row, combined in zip(rankings, (accuracy_z + contrast_z) / 2.0, strict=True):
        row["combined_z_score"] = float(combined)

    by_combined = sorted(
        rankings, key=lambda row: row["combined_z_score"], reverse=True
    )
    by_accuracy = sorted(
        rankings, key=lambda row: row["balanced_accuracy"], reverse=True
    )
    by_contrast = sorted(
        rankings,
        key=lambda row: row["balanced_detection_contrast"],
        reverse=True,
    )

    reference = (5, 8, 13)
    reference_accuracy = balanced_centroid_accuracy(
        train_x, train_y, val_x, val_y, reference, len(classes)
    )
    reference_contrast = balanced_detection_contrast(
        val_contrast, val_y, reference, len(classes)
    )
    reference_row = next(row for row in rankings if row["bands"] == list(reference))
    report = {
        "method": (
            "Per-object inner-box median spectra; log transform and per-object mean centering; "
            "training-standardized nearest-class-centroid balanced accuracy on the fixed val split; "
            "plus per-class-balanced median object-to-local-background contrast normalized by each "
            "image's 1st-to-99th-percentile channel span. The combined score averages metric z-scores. "
            "These are screening proxies, not object-detection mAP."
        ),
        "seed": args.seed,
        "manifest_sha256": sha256_file(args.manifest),
        "sampled_images": {"train": len(train_ids), "val": len(val_ids)},
        "sample_fingerprints": {
            "train_sha256": sample_fingerprint(train_ids),
            "val_sha256": sample_fingerprint(val_ids),
        },
        "sampled_objects": {"train": len(train_y), "val": len(val_y)},
        "reference": {
            "bands": list(reference),
            "balanced_accuracy": reference_accuracy,
            "balanced_detection_contrast": reference_contrast,
            "combined_z_score": reference_row["combined_z_score"],
            "ranks": {
                "combined": next(
                    index + 1
                    for index, row in enumerate(by_combined)
                    if row["bands"] == list(reference)
                ),
                "accuracy": next(
                    index + 1
                    for index, row in enumerate(by_accuracy)
                    if row["bands"] == list(reference)
                ),
                "contrast": next(
                    index + 1
                    for index, row in enumerate(by_contrast)
                    if row["bands"] == list(reference)
                ),
            },
        },
        "top_by_combined_score": by_combined[: max(1, args.top)],
        "top_by_accuracy": by_accuracy[: max(1, args.top)],
        "top_by_detection_contrast": by_contrast[: max(1, args.top)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
