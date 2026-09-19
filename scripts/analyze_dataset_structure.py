from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from hsi_detection.annotations import VocAnnotation, VocObject, read_voc_annotation, sanitize_annotation
from hsi_detection.layout import discover_layout, read_classes


def _quantiles(values: Iterable[float]) -> dict[str, float | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"p10": None, "p50": None, "p90": None}
    return {
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
    }


def _normalized_box(obj: VocObject, annotation: VocAnnotation) -> tuple[str, float, float, float, float]:
    return (
        obj.name,
        obj.xmin / annotation.width,
        obj.ymin / annotation.height,
        obj.xmax / annotation.width,
        obj.ymax / annotation.height,
    )


def _iou(
    left: tuple[str, float, float, float, float],
    right: tuple[str, float, float, float, float],
) -> float:
    x1 = max(left[1], right[1])
    y1 = max(left[2], right[2])
    x2 = min(left[3], right[3])
    y2 = min(left[4], right[4])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[3] - left[1]) * max(0.0, left[4] - left[2])
    right_area = max(0.0, right[3] - right[1]) * max(0.0, right[4] - right[2])
    return intersection / (left_area + right_area - intersection + 1e-12)


def _transfer_quality(
    target_ids: list[int],
    source_by_target: dict[int, int],
    boxes_by_id: dict[int, list[tuple[str, float, float, float, float]]],
) -> dict[str, object]:
    best_ious: list[float] = []
    gaps: list[int] = []
    same_class_candidates = 0
    same_object_counts = 0
    for target_id in target_ids:
        source_id = source_by_target[target_id]
        source_boxes = boxes_by_id[source_id]
        target_boxes = boxes_by_id[target_id]
        gaps.append(abs(target_id - source_id))
        same_object_counts += len(source_boxes) == len(target_boxes)
        for target_box in target_boxes:
            candidates = [_iou(target_box, source_box) for source_box in source_boxes if source_box[0] == target_box[0]]
            same_class_candidates += bool(candidates)
            best_ious.append(max(candidates, default=0.0))

    ious = np.asarray(best_ious, dtype=np.float64)
    return {
        "target_images": len(target_ids),
        "target_boxes": int(ious.size),
        "source_id_gap": _quantiles(gaps),
        "same_object_count_image_rate": same_object_counts / len(target_ids) if target_ids else None,
        "target_box_with_same_class_source_rate": same_class_candidates / ious.size if ious.size else None,
        "best_same_class_iou": _quantiles(ious),
        "box_recall_at_iou_0_50": float(np.mean(ious >= 0.50)) if ious.size else None,
        "box_recall_at_iou_0_75": float(np.mean(ious >= 0.75)) if ious.size else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit split balance, box geometry, and numeric-ID locality")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/extracted"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/pseudo_rgb/split_manifest.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/dataset_structure_audit.json"),
    )
    args = parser.parse_args()

    layout = discover_layout(args.raw_root)
    classes = read_classes(layout.class_file)
    split_by_id: dict[int, str] = {}
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            split_by_id[int(row["image_id"])] = row["split"]

    annotations: dict[int, VocAnnotation] = {}
    boxes_by_id: dict[int, list[tuple[str, float, float, float, float]]] = {}
    split_instances = {"train": Counter(), "val": Counter()}
    geometry: dict[str, dict[str, list[float]]] = {
        name: {"width_px": [], "height_px": [], "area_fraction": [], "aspect_ratio": []}
        for name in classes
    }
    for xml_path in sorted(layout.train_annotations.glob("*.xml")):
        image_id = int(xml_path.stem)
        annotation = sanitize_annotation(read_voc_annotation(xml_path)).annotation
        annotations[image_id] = annotation
        boxes_by_id[image_id] = [_normalized_box(obj, annotation) for obj in annotation.objects]
        split = split_by_id[image_id]
        for obj in annotation.objects:
            width = obj.xmax - obj.xmin
            height = obj.ymax - obj.ymin
            split_instances[split][obj.name] += 1
            geometry[obj.name]["width_px"].append(width)
            geometry[obj.name]["height_px"].append(height)
            geometry[obj.name]["area_fraction"].append(width * height / (annotation.width * annotation.height))
            geometry[obj.name]["aspect_ratio"].append(width / height)

    train_ids = sorted(image_id for image_id, split in split_by_id.items() if split == "train")
    val_ids = sorted(image_id for image_id, split in split_by_id.items() if split == "val")
    original_train_ids = sorted(annotations)
    test_ids = sorted(int(path.stem) for path in layout.test_images.glob("*.png"))
    union_ids = sorted(set(original_train_ids) | set(test_ids))

    train_array = np.asarray(train_ids, dtype=np.int64)
    nearest_train_by_val = {
        image_id: int(train_array[np.argmin(np.abs(train_array - image_id))]) for image_id in val_ids
    }
    consecutive_pairs = [
        (left, right)
        for left, right in zip(original_train_ids, original_train_ids[1:])
        if right - left == 1
    ]
    consecutive_source = {right: left for left, right in consecutive_pairs}

    all_original_train = np.asarray(original_train_ids, dtype=np.int64)
    test_neighbor_gaps = [int(np.min(np.abs(all_original_train - image_id))) for image_id in test_ids]
    class_profile = []
    for name in classes:
        train_count = split_instances["train"][name]
        val_count = split_instances["val"][name]
        total = train_count + val_count
        class_profile.append(
            {
                "class": name,
                "instances": {"train": train_count, "val": val_count, "total": total},
                "val_instance_rate": val_count / total if total else None,
                "geometry": {metric: _quantiles(values) for metric, values in geometry[name].items()},
            }
        )

    report = {
        "grain": "one image identified by its numeric PNG stem; boxes are sanitized Pascal VOC objects",
        "counts": {
            "fixed_train_images": len(train_ids),
            "fixed_val_images": len(val_ids),
            "official_labeled_images": len(original_train_ids),
            "official_test_images": len(test_ids),
            "train_test_id_overlap": len(set(original_train_ids) & set(test_ids)),
            "union_images": len(union_ids),
            "union_id_min": min(union_ids),
            "union_id_max": max(union_ids),
            "missing_ids_inside_union_range": (max(union_ids) - min(union_ids) + 1) - len(union_ids),
        },
        "class_profile": class_profile,
        "numeric_id_locality": {
            "official_test_nearest_labeled_id_gap": _quantiles(test_neighbor_gaps),
            "official_test_with_adjacent_labeled_id_rate": float(np.mean(np.asarray(test_neighbor_gaps) == 1)),
            "fixed_val_nearest_fixed_train_label_transfer": _transfer_quality(
                val_ids, nearest_train_by_val, boxes_by_id
            ),
            "consecutive_official_labeled_pairs": {
                "pair_count": len(consecutive_pairs),
                "pair_rate_among_neighboring_sorted_ids": len(consecutive_pairs) / (len(original_train_ids) - 1),
                "label_transfer": _transfer_quality(
                    [right for _, right in consecutive_pairs], consecutive_source, boxes_by_id
                ),
            },
        },
        "interpretation": {
            "split_balance_target": 0.2,
            "numeric_id_warning": (
                "Numeric closeness is useful only if label-transfer IoU is high; low IoU rules out direct box copying "
                "even when test IDs are interleaved with labeled IDs."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["counts"], indent=2))
    print(json.dumps(report["numeric_id_locality"], indent=2))
    print(f"report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
