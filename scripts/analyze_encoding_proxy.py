from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from hsi_detection.annotations import read_voc_annotation, sanitize_annotation
from hsi_detection.layout import discover_layout, read_classes
from hsi_detection.spectral import x2cube


BAND_ORDER = (5, 8, 13, 0, 1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15)


def _load_split(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["image_id"]: row["split"] for row in csv.DictReader(handle)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inner_box_median(image: np.ndarray, obj: object) -> np.ndarray | None:
    x1 = max(0, int(np.floor(obj.xmin)))
    y1 = max(0, int(np.floor(obj.ymin)))
    x2 = min(image.shape[1], int(np.ceil(obj.xmax)))
    y2 = min(image.shape[0], int(np.ceil(obj.ymax)))
    if x2 <= x1 or y2 <= y1:
        return None
    margin_x = int(round((x2 - x1) * 0.2)) if x2 - x1 >= 5 else 0
    margin_y = int(round((y2 - y1) * 0.2)) if y2 - y1 >= 5 else 0
    inner = image[y1 + margin_y : y2 - margin_y, x1 + margin_x : x2 - margin_x]
    if not inner.size:
        inner = image[y1:y2, x1:x2]
    if not inner.size:
        return None
    return np.median(inner.reshape(-1, image.shape[2]), axis=0)


def _collect_split(
    image_ids: list[str],
    split_name: str,
    annotations: dict[str, object],
    train_images: Path,
    pseudo_root: Path,
    multispectral_root: Path,
    class_to_id: dict[str, int],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rows: dict[str, list[np.ndarray]] = {
        "raw3": [],
        "raw16": [],
        "independent3": [],
        "shared3": [],
        "shared16": [],
        "shared16_p0_100": [],
        "shared16_p0.1_99.9": [],
        "shared16_p0.5_99.5": [],
    }
    labels: list[int] = []
    for image_id in tqdm(image_ids, desc=f"Encoding proxy {split_name}"):
        annotation = sanitize_annotation(annotations[image_id]).annotation
        with Image.open(train_images / annotation.filename) as image:
            cube = x2cube(np.asarray(image))
        raw16 = cube[:, :, BAND_ORDER]
        with Image.open(pseudo_root / "images" / split_name / annotation.filename) as image:
            independent3 = np.asarray(image.convert("RGB"))
        shared16 = np.load(
            multispectral_root / "images" / split_name / f"{image_id}.npy",
            allow_pickle=False,
        )
        if raw16.shape != shared16.shape:
            raise ValueError(
                f"Shape mismatch for {image_id}: raw {raw16.shape}, shared {shared16.shape}"
            )
        if independent3.shape != raw16[:, :, :3].shape:
            raise ValueError(
                f"Shape mismatch for {image_id}: pseudo {independent3.shape}, raw3 {raw16[:, :, :3].shape}"
            )

        scale_bounds = {
            "shared16_p0_100": np.percentile(raw16, [0.0, 100.0]),
            "shared16_p0.1_99.9": np.percentile(raw16, [0.1, 99.9]),
            "shared16_p0.5_99.5": np.percentile(raw16, [0.5, 99.5]),
        }

        for obj in annotation.objects:
            raw_feature = _inner_box_median(raw16, obj)
            independent_feature = _inner_box_median(independent3, obj)
            shared_feature = _inner_box_median(shared16, obj)
            if raw_feature is None or independent_feature is None or shared_feature is None:
                continue
            rows["raw3"].append(raw_feature[:3])
            rows["raw16"].append(raw_feature)
            rows["independent3"].append(independent_feature)
            rows["shared3"].append(shared_feature[:3])
            rows["shared16"].append(shared_feature)
            for encoding, (low, high) in scale_bounds.items():
                if high <= low:
                    encoded_feature = np.zeros_like(raw_feature)
                else:
                    encoded_feature = np.rint(
                        np.clip((raw_feature - low) / (high - low), 0.0, 1.0) * 255.0
                    )
                rows[encoding].append(encoded_feature)
            labels.append(class_to_id[obj.name])
    return (
        {name: np.asarray(values, dtype=np.float64) for name, values in rows.items()},
        np.asarray(labels, dtype=np.int64),
    )


def _log_shape(features: np.ndarray) -> np.ndarray:
    transformed = np.log1p(features)
    return transformed - transformed.mean(axis=1, keepdims=True)


def _balanced_centroid_accuracy(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    classes: list[str],
) -> tuple[float, dict[str, float]]:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    train = (train_x - mean) / scale
    val = (val_x - mean) / scale
    centroids = np.stack([train[train_y == index].mean(axis=0) for index in range(len(classes))])
    predictions = np.square(val[:, None, :] - centroids[None, :, :]).sum(axis=2).argmin(axis=1)
    per_class = {
        name: float(np.mean(predictions[val_y == index] == index))
        for index, name in enumerate(classes)
    }
    return float(np.mean(list(per_class.values()))), per_class


def _profile_fidelity(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    if reference.shape != candidate.shape:
        raise ValueError(f"Profile shapes differ: {reference.shape} vs {candidate.shape}")
    reference_span = np.ptp(reference, axis=1)
    candidate_span = np.ptp(candidate, axis=1)
    usable = (reference_span > 0) & (candidate_span > 0)
    reference_profile = (reference[usable] - reference[usable].min(axis=1, keepdims=True)) / reference_span[
        usable, None
    ]
    candidate_profile = (candidate[usable] - candidate[usable].min(axis=1, keepdims=True)) / candidate_span[
        usable, None
    ]
    per_object_rmse = np.sqrt(np.square(reference_profile - candidate_profile).mean(axis=1))

    pair_matches = 0
    pair_total = 0
    for left in range(reference.shape[1]):
        for right in range(left + 1, reference.shape[1]):
            reference_sign = np.sign(reference[:, left] - reference[:, right])
            candidate_sign = np.sign(candidate[:, left] - candidate[:, right])
            non_ties = reference_sign != 0
            pair_matches += int(np.sum(candidate_sign[non_ties] == reference_sign[non_ties]))
            pair_total += int(np.sum(non_ties))
    return {
        "objects": int(reference.shape[0]),
        "nonflat_objects_for_profile_rmse": int(np.sum(usable)),
        "normalized_profile_rmse_mean": float(per_object_rmse.mean()),
        "normalized_profile_rmse_p90": float(np.quantile(per_object_rmse, 0.90)),
        "pairwise_band_order_agreement": float(pair_matches / pair_total),
        "non_tied_band_pairs": pair_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare independent-channel and shared-scale hyperspectral encodings on identical objects"
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/extracted"))
    parser.add_argument("--pseudo-root", type=Path, default=Path("data/processed/pseudo_rgb"))
    parser.add_argument(
        "--multispectral-root", type=Path, default=Path("data/processed/hsi16_shared")
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/encoding_proxy.json"))
    args = parser.parse_args()

    layout = discover_layout(args.raw_root)
    classes = read_classes(layout.class_file)
    class_to_id = {name: index for index, name in enumerate(classes)}
    manifest_path = args.pseudo_root / "split_manifest.csv"
    split_by_id = _load_split(manifest_path)
    annotations = {
        path.stem: read_voc_annotation(path)
        for path in sorted(layout.train_annotations.glob("*.xml"))
    }
    if set(annotations) != set(split_by_id):
        raise ValueError("Manifest image IDs do not exactly match the official labeled images")

    train_ids = sorted(image_id for image_id, split in split_by_id.items() if split == "train")
    val_ids = sorted(image_id for image_id, split in split_by_id.items() if split == "val")
    train_features, train_y = _collect_split(
        train_ids,
        "train",
        annotations,
        layout.train_images,
        args.pseudo_root,
        args.multispectral_root,
        class_to_id,
    )
    val_features, val_y = _collect_split(
        val_ids,
        "val",
        annotations,
        layout.train_images,
        args.pseudo_root,
        args.multispectral_root,
        class_to_id,
    )

    classification = {}
    for encoding in train_features:
        intensity_macro, intensity_per_class = _balanced_centroid_accuracy(
            train_features[encoding], train_y, val_features[encoding], val_y, classes
        )
        shape_macro, shape_per_class = _balanced_centroid_accuracy(
            _log_shape(train_features[encoding]),
            train_y,
            _log_shape(val_features[encoding]),
            val_y,
            classes,
        )
        classification[encoding] = {
            "channels": int(train_features[encoding].shape[1]),
            "standardized_intensity": {
                "balanced_accuracy": intensity_macro,
                "per_class_accuracy": intensity_per_class,
            },
            "log_profile_shape": {
                "balanced_accuracy": shape_macro,
                "per_class_accuracy": shape_per_class,
            },
        }

    report = {
        "method": (
            "All sanitized train and fixed-validation objects are evaluated at identical inner-box pixels. "
            "Nearest-class-centroid balanced accuracy is reported both on globally standardized object-median "
            "intensities and on log channel-profile shape. Profile fidelity min-max normalizes each object's "
            "band vector before comparing it with the raw sensor vector. These are encoding diagnostics and "
            "class-separability proxies, not detector mAP."
        ),
        "manifest_sha256": _sha256(manifest_path),
        "band_order": list(BAND_ORDER),
        "samples": {"train_objects": int(len(train_y)), "val_objects": int(len(val_y))},
        "classification_proxies": classification,
        "raw_profile_fidelity_on_validation": {
            "independent3_vs_raw3": _profile_fidelity(
                val_features["raw3"], val_features["independent3"]
            ),
            "shared3_vs_raw3": _profile_fidelity(val_features["raw3"], val_features["shared3"]),
            "shared16_vs_raw16": _profile_fidelity(
                val_features["raw16"], val_features["shared16"]
            ),
            "shared16_p0_100_vs_raw16": _profile_fidelity(
                val_features["raw16"], val_features["shared16_p0_100"]
            ),
            "shared16_p0.1_99.9_vs_raw16": _profile_fidelity(
                val_features["raw16"], val_features["shared16_p0.1_99.9"]
            ),
            "shared16_p0.5_99.5_vs_raw16": _profile_fidelity(
                val_features["raw16"], val_features["shared16_p0.5_99.5"]
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
