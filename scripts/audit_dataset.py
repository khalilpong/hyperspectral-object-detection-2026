from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image

from hsi_detection.annotations import read_voc_annotation
from hsi_detection.layout import discover_layout, read_classes
from hsi_detection.spectral import x2cube


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the official HSI dataset")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/extracted"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/data_audit.json"))
    parser.add_argument("--image-sample", type=int, default=64)
    args = parser.parse_args()

    layout = discover_layout(args.raw_root)
    classes = read_classes(layout.class_file)
    class_set = set(classes)
    train_images = sorted(layout.train_images.glob("*.png"))
    test_images = sorted(layout.test_images.glob("*.png"))
    annotations = sorted(layout.train_annotations.glob("*.xml"))
    train_stems = {path.stem for path in train_images}
    annotation_stems = {path.stem for path in annotations}

    class_instances: Counter[str] = Counter()
    images_per_class: Counter[str] = Counter()
    objects_per_image: Counter[int] = Counter()
    annotation_sizes: Counter[str] = Counter()
    issues: list[str] = []

    for xml_path in annotations:
        annotation = read_voc_annotation(xml_path)
        annotation_sizes[f"{annotation.width}x{annotation.height}x{annotation.depth}"] += 1
        if annotation.depth != 16:
            issues.append(f"{xml_path.name}: expected depth 16, got {annotation.depth}")
        objects_per_image[len(annotation.objects)] += 1
        names = {obj.name for obj in annotation.objects}
        images_per_class.update(names)
        class_instances.update(obj.name for obj in annotation.objects)
        unknown = names - class_set
        if unknown:
            issues.append(f"{xml_path.name}: unknown classes {sorted(unknown)}")
        expected_image = layout.train_images / annotation.filename
        if not expected_image.exists():
            issues.append(f"{xml_path.name}: missing image {annotation.filename}")
        for obj in annotation.objects:
            if not (0 <= obj.xmin < obj.xmax <= annotation.width):
                issues.append(f"{xml_path.name}: invalid x box {obj}")
            if not (0 <= obj.ymin < obj.ymax <= annotation.height):
                issues.append(f"{xml_path.name}: invalid y box {obj}")

    for image_id in sorted(train_stems - annotation_stems):
        issues.append(f"{image_id}.png: missing annotation")
    for image_id in sorted(annotation_stems - train_stems):
        issues.append(f"{image_id}.xml: missing image")

    sample_count = max(args.image_sample, 0)
    train_quota = min(len(train_images), (sample_count + 1) // 2)
    test_quota = min(len(test_images), sample_count - train_quota)
    sampled = train_images[:train_quota] + test_images[:test_quota]
    image_modes: Counter[str] = Counter()
    mosaic_sizes: Counter[str] = Counter()
    cube_sizes: Counter[str] = Counter()
    value_min: int | None = None
    value_max: int | None = None
    for image_path in sampled:
        with Image.open(image_path) as image:
            array = np.asarray(image)
            image_modes[image.mode] += 1
        mosaic_sizes[f"{array.shape[1]}x{array.shape[0]}"] += 1
        cube = x2cube(array)
        cube_sizes[f"{cube.shape[1]}x{cube.shape[0]}x{cube.shape[2]}"] += 1
        if image_path.parent == layout.train_images:
            annotation_path = layout.train_annotations / f"{image_path.stem}.xml"
            if annotation_path.exists():
                annotation = read_voc_annotation(annotation_path)
                actual_size = (cube.shape[1], cube.shape[0], cube.shape[2])
                expected_size = (annotation.width, annotation.height, annotation.depth)
                if actual_size != expected_size:
                    issues.append(
                        f"{image_path.name}: cube size {actual_size} != annotation size {expected_size}"
                    )
        current_min, current_max = int(cube.min()), int(cube.max())
        value_min = current_min if value_min is None else min(value_min, current_min)
        value_max = current_max if value_max is None else max(value_max, current_max)

    report = {
        "paths": {
            "train_images": str(layout.train_images),
            "train_annotations": str(layout.train_annotations),
            "test_images": str(layout.test_images),
        },
        "counts": {
            "classes": len(classes),
            "train_images": len(train_images),
            "annotations": len(annotations),
            "test_images": len(test_images),
        },
        "classes": classes,
        "class_instances": dict(class_instances),
        "images_per_class": dict(images_per_class),
        "objects_per_image": {str(k): v for k, v in sorted(objects_per_image.items())},
        "annotation_sizes": dict(annotation_sizes),
        "sampled_image_properties": {
            "count": len(sampled),
            "train_count": train_quota,
            "test_count": test_quota,
            "modes": dict(image_modes),
            "mosaic_sizes": dict(mosaic_sizes),
            "cube_sizes": dict(cube_sizes),
            "value_min": value_min,
            "value_max": value_max,
        },
        "issues": issues,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["counts"], indent=2))
    print("class_instances:", json.dumps(report["class_instances"], ensure_ascii=False))
    print(f"issues: {len(issues)}")
    print(f"report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
