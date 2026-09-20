from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from hsi_detection.tiling import LabeledBox, crop_labeled_boxes, parse_yolo_rows, to_yolo_rows


def _numeric_paths(directory: Path) -> list[Path]:
    paths = list(directory.glob("*.npy"))
    try:
        return sorted(paths, key=lambda path: int(path.stem))
    except ValueError as error:
        raise ValueError(f"Expected numeric source stems in {directory.resolve()}") from error


def _stable_seed(seed: int, image_id: str, crop_index: int) -> int:
    digest = hashlib.sha256(f"{seed}:{image_id}:{crop_index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _candidate_priority(box: LabeledBox, class_counts: Counter[int]) -> float:
    """Prefer rare and small objects without using validation labels or AP values."""
    return 1.0 / math.sqrt(max(class_counts[box.class_id], 1) * max(box.area, 1.0))


def _crop_origin_for_anchor(
    anchor: LabeledBox,
    *,
    image_width: int,
    image_height: int,
    crop_width: int,
    crop_height: int,
    jitter_fraction: float,
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    if anchor.x2 - anchor.x1 > crop_width or anchor.y2 - anchor.y1 > crop_height:
        return None

    def origin(
        lower: float,
        upper: float,
        image_length: int,
        crop_length: int,
    ) -> int | None:
        maximum_origin = image_length - crop_length
        feasible_low = max(0, math.ceil(upper - crop_length - 1e-7))
        feasible_high = min(maximum_origin, math.floor(lower + 1e-7))
        if feasible_low > feasible_high:
            return None
        centered = (lower + upper - crop_length) / 2.0
        jitter = rng.uniform(-jitter_fraction * crop_length, jitter_fraction * crop_length)
        return int(np.clip(round(centered + jitter), feasible_low, feasible_high))

    x0 = origin(anchor.x1, anchor.x2, image_width, crop_width)
    y0 = origin(anchor.y1, anchor.y2, image_height, crop_height)
    if x0 is None or y0 is None:
        return None
    return x0, y0


def _write_crop(
    array: np.ndarray,
    boxes: list[LabeledBox],
    *,
    image_id: str,
    crop_index: int,
    x0: int,
    y0: int,
    crop_width: int,
    crop_height: int,
    minimum_visible: float,
    image_directory: Path,
    label_directory: Path,
) -> tuple[str, int]:
    cropped_boxes = crop_labeled_boxes(
        boxes,
        x0=x0,
        y0=y0,
        crop_width=crop_width,
        crop_height=crop_height,
        minimum_visible=minimum_visible,
    )
    if not cropped_boxes:
        raise RuntimeError(f"Object crop for {image_id} unexpectedly retained no labels")

    stem = f"{image_id}__oc{crop_index}_x{x0}_y{y0}_w{crop_width}_h{crop_height}"
    destination_npy = image_directory / f"{stem}.npy"
    destination_png = image_directory / f"{stem}.png"
    destination_label = label_directory / f"{stem}.txt"
    crop = np.ascontiguousarray(array[y0 : y0 + crop_height, x0 : x0 + crop_width])
    if crop.shape[:2] != (crop_height, crop_width):
        raise RuntimeError(f"Unexpected crop shape for {image_id}: {crop.shape}")

    temporary_npy = destination_npy.with_suffix(".npy.tmp")
    temporary_png = destination_png.with_suffix(".png.tmp")
    temporary_label = destination_label.with_suffix(".txt.tmp")
    with temporary_npy.open("wb") as handle:
        np.save(handle, crop, allow_pickle=False)
    Image.fromarray(crop[:, :, :3], mode="RGB").save(
        temporary_png, format="PNG", compress_level=3
    )
    temporary_label.write_text(
        "\n".join(to_yolo_rows(cropped_boxes, width=crop_width, height=crop_height)) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_npy, destination_npy)
    os.replace(temporary_png, destination_png)
    os.replace(temporary_label, destination_label)
    return stem, len(cropped_boxes)


def _add_training_sources(config: dict[str, object], sources: list[str]) -> dict[str, object]:
    updated = dict(config)
    train = updated["train"]
    train_sources = [train] if isinstance(train, str) else list(train)
    for source in sources:
        if source not in train_sources:
            train_sources.append(source)
    updated["train"] = train_sources
    updated["object_crop_sources"] = sources
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add deterministic train-only object crops to an encoded HSI16 dataset"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train"], choices=("train", "val"))
    parser.add_argument("--tile-height", type=int, default=128)
    parser.add_argument("--tile-width", type=int, default=256)
    parser.add_argument("--crops-per-image", type=int, default=1)
    parser.add_argument("--jitter-fraction", type=float, default=0.15)
    parser.add_argument("--minimum-visible", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-suffix", default="object_crops")
    args = parser.parse_args()

    if args.tile_height <= 0 or args.tile_width <= 0:
        parser.error("tile dimensions must be positive")
    if args.crops_per_image <= 0:
        parser.error("--crops-per-image must be positive")
    if not 0.0 <= args.jitter_fraction <= 0.5:
        parser.error("--jitter-fraction must be in [0, 0.5]")
    if not 0.0 < args.minimum_visible <= 1.0:
        parser.error("--minimum-visible must be in (0, 1]")
    if not args.output_suffix or any(character in args.output_suffix for character in "/\\"):
        parser.error("--output-suffix must be one safe path component")

    root = args.dataset_root.resolve()
    records: list[tuple[str, str, Path, list[LabeledBox], int, int]] = []
    class_counts: Counter[int] = Counter()
    for split in dict.fromkeys(args.splits):
        image_directory = root / "images" / split
        label_directory = root / "labels" / split
        for image_path in _numeric_paths(image_directory):
            array = np.load(image_path, mmap_mode="r", allow_pickle=False)
            if array.ndim != 3 or array.shape[2] != 16 or array.dtype != np.uint8:
                raise ValueError(
                    f"Expected uint8 H x W x 16 array at {image_path}, got {array.shape}/{array.dtype}"
                )
            height, width = map(int, array.shape[:2])
            label_path = label_directory / f"{image_path.stem}.txt"
            boxes = parse_yolo_rows(
                label_path.read_text(encoding="utf-8"), width=width, height=height
            )
            class_counts.update(box.class_id for box in boxes)
            records.append((split, image_path.stem, image_path, boxes, width, height))

    output_directories: dict[str, tuple[Path, Path]] = {}
    for split in dict.fromkeys(args.splits):
        image_directory = root / "images" / f"{split}_{args.output_suffix}"
        label_directory = root / "labels" / f"{split}_{args.output_suffix}"
        if any(image_directory.glob("*")) or any(label_directory.glob("*")):
            raise FileExistsError(
                f"Crop output is not empty: {image_directory} / {label_directory}"
            )
        image_directory.mkdir(parents=True, exist_ok=True)
        label_directory.mkdir(parents=True, exist_ok=True)
        output_directories[split] = image_directory, label_directory

    generated_by_split: Counter[str] = Counter()
    anchor_classes: Counter[int] = Counter()
    retained_labels = 0
    skipped_images: list[str] = []
    manifest_rows: list[dict[str, object]] = []
    for split, image_id, image_path, boxes, width, height in records:
        crop_width = min(args.tile_width, width)
        crop_height = min(args.tile_height, height)
        ranked = sorted(
            boxes,
            key=lambda box: (
                -_candidate_priority(box, class_counts),
                box.class_id,
                box.y1,
                box.x1,
            ),
        )
        array = np.load(image_path, allow_pickle=False)
        used_windows: set[tuple[int, int]] = set()
        generated_for_image = 0
        for anchor_rank, anchor in enumerate(ranked):
            if generated_for_image >= args.crops_per_image:
                break
            rng = np.random.default_rng(_stable_seed(args.seed, image_id, anchor_rank))
            origin = _crop_origin_for_anchor(
                anchor,
                image_width=width,
                image_height=height,
                crop_width=crop_width,
                crop_height=crop_height,
                jitter_fraction=args.jitter_fraction,
                rng=rng,
            )
            if origin is None or origin in used_windows:
                continue
            x0, y0 = origin
            stem, label_count = _write_crop(
                array,
                boxes,
                image_id=image_id,
                crop_index=generated_for_image,
                x0=x0,
                y0=y0,
                crop_width=crop_width,
                crop_height=crop_height,
                minimum_visible=args.minimum_visible,
                image_directory=output_directories[split][0],
                label_directory=output_directories[split][1],
            )
            used_windows.add(origin)
            generated_for_image += 1
            generated_by_split[split] += 1
            anchor_classes[anchor.class_id] += 1
            retained_labels += label_count
            manifest_rows.append(
                {
                    "crop_id": stem,
                    "source_split": split,
                    "source_image_id": image_id,
                    "x0": x0,
                    "y0": y0,
                    "width": crop_width,
                    "height": crop_height,
                    "anchor_class_id": anchor.class_id,
                    "retained_labels": label_count,
                }
            )
        if generated_for_image == 0:
            skipped_images.append(f"{split}/{image_id}")

    crop_sources = {
        split: f"images/{split}_{args.output_suffix}" for split in dict.fromkeys(args.splits)
    }
    fixed_path = root / "dataset.yaml"
    all_path = root / "dataset_all.yaml"
    fixed_config = yaml.safe_load(fixed_path.read_text(encoding="utf-8"))
    all_config = yaml.safe_load(all_path.read_text(encoding="utf-8"))
    fixed_sources = [crop_sources["train"]] if "train" in crop_sources else []
    (root / "dataset_object_crops.yaml").write_text(
        yaml.safe_dump(
            _add_training_sources(fixed_config, fixed_sources),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    (root / "dataset_all_object_crops.yaml").write_text(
        yaml.safe_dump(
            _add_training_sources(all_config, list(crop_sources.values())),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    report = {
        "schema_version": 1,
        "dataset_root": str(root),
        "settings": {
            "splits": list(dict.fromkeys(args.splits)),
            "tile_height": args.tile_height,
            "tile_width": args.tile_width,
            "crops_per_image": args.crops_per_image,
            "jitter_fraction": args.jitter_fraction,
            "minimum_visible": args.minimum_visible,
            "seed": args.seed,
            "selection": "rarest-and-smallest-within-each-source-image",
            "renormalized": False,
        },
        "source_images": len(records),
        "generated_crops": sum(generated_by_split.values()),
        "generated_by_split": dict(sorted(generated_by_split.items())),
        "skipped_images": skipped_images,
        "retained_labels": retained_labels,
        "anchor_class_counts": {str(key): value for key, value in sorted(anchor_classes.items())},
        "class_counts": {str(key): value for key, value in sorted(class_counts.items())},
        "manifest": manifest_rows,
    }
    report_path = root / "object_crop_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "manifest"}, indent=2))
    print(f"fixed dataset: {(root / 'dataset_object_crops.yaml').resolve()}")
    print(f"full dataset: {(root / 'dataset_all_object_crops.yaml').resolve()}")


if __name__ == "__main__":
    main()
