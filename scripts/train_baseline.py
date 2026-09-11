from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".ultralytics").resolve()))

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the single-model pseudo-RGB baseline")
    parser.add_argument("--data", type=Path, default=Path("data/processed/pseudo_rgb/dataset.yaml"))
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--resume", type=Path, help="Resume an interrupted Ultralytics run from last.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--name", default="baseline")
    args = parser.parse_args()

    model = YOLO(str(args.resume.resolve()) if args.resume else args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        project=str(Path("runs").resolve()),
        name=args.name,
        exist_ok=False,
        resume=bool(args.resume),
        amp=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
