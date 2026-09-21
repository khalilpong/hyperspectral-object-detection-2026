"""Train one RT-DETR checkpoint on the HSI16 dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".ultralytics").resolve()))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="rtdetr-l.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--name", default="rtdetr_l_hsi16_fixed")
    parser.add_argument("--project", type=Path, default=Path("runs"))
    parser.add_argument("--extra-channel-init", choices=("random", "zero"), default="random")
    parser.add_argument(
        "--num-denoising",
        type=int,
        default=100,
        help="Number of RT-DETR contrastive denoising queries used during training.",
    )
    parser.add_argument("--val", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    for name in ("epochs", "imgsz", "batch", "num_denoising"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    return args


def build_train_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "data": str(args.data.resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "seed": args.seed,
        # RT-DETR uses F.grid_sample and explicitly does not support deterministic=True.
        "deterministic": False,
        # Ultralytics warns that RT-DETR AMP can produce NaNs during bipartite matching.
        "amp": False,
        "project": str(args.project.resolve()),
        "name": args.name,
        "exist_ok": False,
        "resume": False,
        "val": args.val,
        "plots": args.plots,
    }


def main(argv: Sequence[str] | None = None) -> None:
    from ultralytics import RTDETR

    from hsi_detection.rtdetr import HSIRTDETRTrainer

    args = parse_args(argv)
    trainer_type = type(
        "ConfiguredHSIRTDETRTrainer",
        (HSIRTDETRTrainer,),
        {
            "extra_channel_init": args.extra_channel_init,
            "num_denoising": args.num_denoising,
        },
    )
    model = RTDETR(args.model)
    model.train(trainer=trainer_type, **build_train_kwargs(args))


if __name__ == "__main__":
    main()
