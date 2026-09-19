from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from hsi_detection.annotations import read_voc_annotation, sanitize_annotation
from hsi_detection.layout import discover_layout, read_classes


def _load_split(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["image_id"]: row["split"] for row in csv.DictReader(handle)}


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
    return np.median(inner.reshape(-1, 3), axis=0)


def _collect_features(
    image_ids: list[str],
    split_by_id: dict[str, str],
    annotations: dict[str, object],
    processed_root: Path,
    class_to_id: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    for image_id in image_ids:
        annotation = sanitize_annotation(annotations[image_id]).annotation
        path = processed_root / "images" / split_by_id[image_id] / annotation.filename
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        for obj in annotation.objects:
            feature = _inner_box_median(rgb, obj)
            if feature is not None:
                features.append(feature)
                labels.append(class_to_id[obj.name])
    return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def _random_hsv(
    rgb: np.ndarray,
    replicas: int,
    hgain: float,
    sgain: float,
    vgain: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    repeated = np.repeat(np.rint(rgb).clip(0, 255).astype(np.uint8), replicas, axis=0)
    bgr = repeated[:, None, ::-1]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, 0].astype(np.float64)
    gains = rng.uniform(-1.0, 1.0, size=(len(repeated), 3)) * np.asarray(
        [hgain, sgain, vgain]
    )
    hsv[:, 0] = np.mod(hsv[:, 0] + gains[:, 0] * 180.0, 180.0)
    hsv[:, 1] = np.clip(hsv[:, 1] * (gains[:, 1] + 1.0), 0.0, 255.0)
    hsv[:, 2] = np.clip(hsv[:, 2] * (gains[:, 2] + 1.0), 0.0, 255.0)
    transformed_bgr = cv2.cvtColor(hsv.astype(np.uint8)[:, None], cv2.COLOR_HSV2BGR)[:, 0]
    transformed_rgb = transformed_bgr[:, ::-1].astype(np.float64)
    return repeated.astype(np.float64), transformed_rgb


def _log_ratio(rgb: np.ndarray) -> np.ndarray:
    transformed = np.log1p(rgb)
    return transformed - transformed.mean(axis=1, keepdims=True)


def _balanced_centroid_accuracy(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    class_count: int,
) -> tuple[float, list[float]]:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    train = (train_x - mean) / scale
    val = (val_x - mean) / scale
    centroids = np.stack([train[train_y == cls].mean(axis=0) for cls in range(class_count)])
    predictions = np.square(val[:, None, :] - centroids[None, :, :]).sum(axis=2).argmin(axis=1)
    per_class = [float(np.mean(predictions[val_y == cls] == cls)) for cls in range(class_count)]
    return float(np.mean(per_class)), per_class


def _angle_degrees(original: np.ndarray, transformed: np.ndarray) -> np.ndarray:
    original_norm = original / np.maximum(np.linalg.norm(original, axis=1, keepdims=True), 1e-9)
    transformed_norm = transformed / np.maximum(np.linalg.norm(transformed, axis=1, keepdims=True), 1e-9)
    cosine = np.clip(np.sum(original_norm * transformed_norm, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure how YOLO HSV gains distort pseudo-spectral RGB ratios")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/extracted"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed/pseudo_rgb"))
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("artifacts/hsv_augmentation_proxy.json"))
    args = parser.parse_args()

    layout = discover_layout(args.raw_root)
    classes = read_classes(layout.class_file)
    class_to_id = {name: index for index, name in enumerate(classes)}
    split_by_id = _load_split(args.processed_root / "split_manifest.csv")
    annotations = {
        path.stem: read_voc_annotation(path)
        for path in sorted(layout.train_annotations.glob("*.xml"))
    }
    train_ids = sorted(image_id for image_id, split in split_by_id.items() if split == "train")
    val_ids = sorted(image_id for image_id, split in split_by_id.items() if split == "val")
    train_rgb, train_y = _collect_features(
        train_ids, split_by_id, annotations, args.processed_root, class_to_id
    )
    val_rgb, val_y = _collect_features(
        val_ids, split_by_id, annotations, args.processed_root, class_to_id
    )

    profiles = {
        "none": (0.0, 0.0, 0.0),
        "default": (0.015, 0.7, 0.4),
        "reduced": (0.0, 0.15, 0.2),
        "value_only": (0.0, 0.0, 0.2),
    }
    rows = []
    for index, (name, (hgain, sgain, vgain)) in enumerate(profiles.items()):
        rng = np.random.default_rng(args.seed + index)
        original_train, augmented_train = _random_hsv(
            train_rgb, args.replicas, hgain, sgain, vgain, rng
        )
        original_val, augmented_val = _random_hsv(
            val_rgb, args.replicas, hgain, sgain, vgain, rng
        )
        repeated_train_y = np.repeat(train_y, args.replicas)
        repeated_val_y = np.repeat(val_y, args.replicas)
        accuracy, per_class = _balanced_centroid_accuracy(
            _log_ratio(augmented_train),
            repeated_train_y,
            _log_ratio(augmented_val),
            repeated_val_y,
            len(classes),
        )
        angles = np.concatenate(
            [_angle_degrees(original_train, augmented_train), _angle_degrees(original_val, augmented_val)]
        )
        rows.append(
            {
                "profile": name,
                "gains": {"hsv_h": hgain, "hsv_s": sgain, "hsv_v": vgain},
                "balanced_nearest_centroid_accuracy": accuracy,
                "per_class_accuracy": {cls: value for cls, value in zip(classes, per_class, strict=True)},
                "spectral_ratio_angle_degrees": {
                    "p50": float(np.quantile(angles, 0.50)),
                    "p90": float(np.quantile(angles, 0.90)),
                    "p99": float(np.quantile(angles, 0.99)),
                },
            }
        )

    report = {
        "method": (
            "Median RGB inside each sanitized object box in the current pseudo-RGB PNGs. The installed "
            "Ultralytics additive-hue and multiplicative saturation/value transform is approximated on each "
            "object median. Nearest-centroid balanced accuracy uses log channel ratios and is only a spectral "
            "information proxy, not detector mAP."
        ),
        "samples": {"train_objects": len(train_y), "val_objects": len(val_y)},
        "replicas_per_object": args.replicas,
        "seed": args.seed,
        "profiles": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
