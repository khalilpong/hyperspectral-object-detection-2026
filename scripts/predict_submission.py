from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".ultralytics").resolve()))

from ultralytics import YOLO

from hsi_detection.submission import clip_xyxy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference and create submission.csv")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=Path("data/processed/pseudo_rgb/images/test"))
    parser.add_argument("--output", type=Path, default=Path("submission_baseline.csv"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use FP16 inference on supported GPUs to reduce memory use.",
    )
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    args = parser.parse_args()

    image_paths = sorted(args.images.glob("*.png"), key=lambda p: int(p.stem))
    model = YOLO(str(args.weights.resolve()))
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        half=args.half,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        stream=True,
        verbose=False,
    )

    rows: list[dict[str, int | float]] = []
    next_id = 0
    dropped_invalid = 0
    for image_path, result in zip(image_paths, results, strict=True):
        if result.boxes is None:
            continue
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        height, width = result.orig_shape
        for box, class_id, confidence in zip(boxes, classes, confidences, strict=True):
            clipped_box = clip_xyxy(box.tolist(), width=width, height=height)
            if clipped_box is None:
                dropped_invalid += 1
                continue
            x1, y1, x2, y2 = clipped_box
            rows.append(
                {
                    "id": next_id,
                    "image_id": int(image_path.stem),
                    "class_id": int(class_id),
                    "confidence": float(confidence),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
            next_id += 1

    columns = ["id", "image_id", "class_id", "confidence", "x1", "y1", "x2", "y2"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(args.output, index=False)
    print(f"wrote {len(rows)} detections to {args.output.resolve()}")
    print(f"dropped {dropped_invalid} zero-area or non-finite predictions")


if __name__ == "__main__":
    main()
