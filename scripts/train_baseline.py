from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".ultralytics").resolve()))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the single-model pseudo-RGB baseline")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--resume", type=Path, help="Resume an interrupted Ultralytics run from last.pt")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--name")
    parser.add_argument(
        "--val",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable per-epoch validation (use --no-val to disable).",
    )
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable training and validation plots (use --no-plots to disable).",
    )
    args = parser.parse_args(argv)

    if args.resume:
        unsupported = [
            option
            for option in ("model", "epochs", "seed", "name")
            if getattr(args, option) is not None
        ]
        if unsupported:
            parser.error(
                "--resume cannot override checkpoint option(s): " + ", ".join(unsupported)
            )
    return args


def build_train_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Build safe Ultralytics overrides for a new run or a resumed run.

    Ultralytics restores most settings from the checkpoint, but a small set such
    as ``imgsz`` and ``batch`` can be overridden. Omitting an option here is
    important: passing this script's fresh-run default would silently replace
    the checkpoint value during resume.
    """
    if args.resume:
        kwargs: dict[str, Any] = {"resume": True}
        for option in ("imgsz", "batch", "device", "workers", "val", "plots"):
            value = getattr(args, option)
            if value is not None:
                kwargs[option] = value
        if args.data is not None:
            # Ultralytics only uses this as a fallback when the checkpoint's
            # original dataset path no longer exists.
            kwargs["data"] = str(args.data.resolve())
        return kwargs

    data = args.data or Path("data/processed/pseudo_rgb/dataset.yaml")
    return {
        "data": str(data.resolve()),
        "epochs": 30 if args.epochs is None else args.epochs,
        "imgsz": 640 if args.imgsz is None else args.imgsz,
        "batch": 8 if args.batch is None else args.batch,
        "device": "0" if args.device is None else args.device,
        "workers": 4 if args.workers is None else args.workers,
        "seed": 2026 if args.seed is None else args.seed,
        "deterministic": True,
        "project": str(Path("runs").resolve()),
        "name": "baseline" if args.name is None else args.name,
        "exist_ok": False,
        "resume": False,
        "amp": True,
        "val": True if args.val is None else args.val,
        "plots": True if args.plots is None else args.plots,
    }


def main(argv: Sequence[str] | None = None) -> None:
    from ultralytics import YOLO

    args = parse_args(argv)
    model_source = str(args.resume.resolve()) if args.resume else (args.model or "yolo26n.pt")
    model = YOLO(model_source)
    model.train(**build_train_kwargs(args))


if __name__ == "__main__":
    main()
