from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image
import yaml
from tqdm import tqdm

from hsi_detection.annotations import read_voc_annotation, sanitize_annotation, to_yolo_rows
from hsi_detection.layout import discover_layout, read_classes
from hsi_detection.spectral import make_pseudo_rgb, x2cube


def convert_image(source: Path, destination: Path, bands: tuple[int, int, int]) -> None:
    with Image.open(source) as image:
        mosaic = np.asarray(image)
    pseudo_rgb = make_pseudo_rgb(x2cube(mosaic), bands=bands)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pseudo_rgb, mode="RGB").save(destination, compress_level=3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a pseudo-RGB YOLO dataset")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/extracted"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/pseudo_rgb"))
    parser.add_argument("--bands", type=int, nargs=3, default=(5, 8, 13))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")
    if args.output.exists() and args.overwrite:
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    layout = discover_layout(args.raw_root)
    classes = read_classes(layout.class_file)
    class_to_id = {name: index for index, name in enumerate(classes)}
    xml_paths = sorted(layout.train_annotations.glob("*.xml"))
    annotations = {path.stem: read_voc_annotation(path) for path in xml_paths}
    image_ids = sorted(annotations)
    random.Random(args.seed).shuffle(image_ids)
    val_count = max(1, round(len(image_ids) * args.val_fraction))
    val_ids = set(image_ids[:val_count])
    split_by_id = {image_id: ("val" if image_id in val_ids else "train") for image_id in image_ids}

    jobs: list[tuple[Path, Path]] = []
    manifest_rows: list[dict[str, str]] = []
    box_repairs: list[dict[str, int | str]] = []
    clipped_boxes = 0
    dropped_boxes = 0
    for image_id in sorted(image_ids):
        annotation = annotations[image_id]
        sanitization = sanitize_annotation(annotation)
        clipped_boxes += sanitization.clipped_objects
        dropped_boxes += sanitization.dropped_objects
        if sanitization.clipped_objects or sanitization.dropped_objects:
            box_repairs.append(
                {
                    "image_id": image_id,
                    "clipped_boxes": sanitization.clipped_objects,
                    "dropped_boxes": sanitization.dropped_objects,
                }
            )
        split = split_by_id[image_id]
        source = layout.train_images / annotation.filename
        destination = args.output / "images" / split / annotation.filename
        label_path = args.output / "labels" / split / f"{image_id}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(
            "\n".join(to_yolo_rows(sanitization.annotation, class_to_id)) + "\n", encoding="utf-8"
        )
        jobs.append((source, destination))
        manifest_rows.append({"image_id": image_id, "split": split})

    for source in sorted(layout.test_images.glob("*.png")):
        jobs.append((source, args.output / "images" / "test" / source.name))

    bands = tuple(args.bands)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(convert_image, src, dst, bands) for src, dst in jobs]
        for future in tqdm(futures, desc="Converting mosaics"):
            future.result()

    manifest_path = args.output / "split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "split"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    dataset_yaml = {
        "path": str(args.output.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(classes)},
        "hsi_bands": list(bands),
        "split_seed": args.seed,
    }
    (args.output / "dataset.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    preparation_report = {
        "train_images": len(image_ids) - val_count,
        "val_images": val_count,
        "test_images": len(jobs) - len(image_ids),
        "bands": list(bands),
        "split_seed": args.seed,
        "box_repairs": {
            "clipped_boxes": clipped_boxes,
            "dropped_boxes": dropped_boxes,
            "affected_images": box_repairs,
        },
    }
    (args.output / "preparation_report.json").write_text(
        json.dumps(preparation_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Prepared {len(image_ids) - val_count} train, {val_count} val, and "
          f"{len(jobs) - len(image_ids)} test images")
    print(f"dataset: {(args.output / 'dataset.yaml').resolve()}")
    print(f"box repairs: clipped={clipped_boxes}, dropped={dropped_boxes}")


if __name__ == "__main__":
    main()
