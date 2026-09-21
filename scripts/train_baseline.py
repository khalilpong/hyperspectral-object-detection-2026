from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".ultralytics").resolve()))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the single-model pseudo-RGB baseline")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--model")
    parser.add_argument(
        "--load-weights",
        type=Path,
        help=(
            "Transfer compatible weights into a YAML-defined architecture before training. "
            "This is useful for architecture-only variants such as yolo26s-p2.yaml."
        ),
    )
    parser.add_argument("--resume", type=Path, help="Resume an interrupted Ultralytics run from last.pt")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--name")
    parser.add_argument("--optimizer")
    parser.add_argument("--lr0", type=float)
    parser.add_argument("--lrf", type=float)
    parser.add_argument("--warmup-epochs", type=float)
    parser.add_argument("--box", type=float)
    parser.add_argument(
        "--box-iou-loss",
        choices=("ciou", "eiou"),
        help=(
            "Bounding-box overlap loss. Omit for the Ultralytics CIoU baseline; "
            "EIoU changes only BboxLoss and leaves assignment/DFL/inference unchanged."
        ),
    )
    parser.add_argument(
        "--dfl",
        type=float,
        help="Distribution Focal Loss gain for box-edge distance supervision.",
    )
    parser.add_argument(
        "--cls-pw",
        type=float,
        help=(
            "Class-frequency weighting power for classification BCE. "
            "Ultralytics accepts 0.0 (disabled) through 1.0 (full inverse frequency)."
        ),
    )
    parser.add_argument("--hsv-h", type=float)
    parser.add_argument("--hsv-s", type=float)
    parser.add_argument("--hsv-v", type=float)
    parser.add_argument("--mosaic", type=float)
    parser.add_argument("--close-mosaic", type=int)
    parser.add_argument("--scale", type=float)
    parser.add_argument(
        "--degrees",
        type=float,
        help="Maximum absolute random-affine rotation in degrees.",
    )
    parser.add_argument(
        "--multi-scale",
        type=float,
        help=(
            "Randomly resize each training batch within imgsz * (1 +/- value). "
            "This is distinct from random-affine --scale."
        ),
    )
    parser.add_argument(
        "--extra-channel-init",
        choices=("random", "zero"),
        help=(
            "Initialization for input channels beyond RGB when the dataset declares channels > 3. "
            "'zero' preserves the pretrained three-channel function at step zero while allowing "
            "the extra channel weights to learn normally."
        ),
    )
    parser.add_argument(
        "--spectral-stem",
        action="store_true",
        help=(
            "Train a single checkpoint with a learnable 1x1 HSI-to-RGB projection initialized "
            "from physical bands 5/8/13. Resume and prediction recover the stem from the checkpoint."
        ),
    )
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

    if args.spectral_stem and args.extra_channel_init is not None:
        parser.error("--spectral-stem and --extra-channel-init are mutually exclusive")
    if args.spectral_stem and args.box_iou_loss == "eiou":
        parser.error("--spectral-stem and --box-iou-loss eiou are separate experimental variants")
    if args.resume and args.spectral_stem:
        parser.error("--resume recovers SpectralStem from the checkpoint; do not pass --spectral-stem")
    if args.cls_pw is not None and not 0.0 <= args.cls_pw <= 1.0:
        parser.error("--cls-pw must satisfy 0.0 <= value <= 1.0")
    if args.dfl is not None and (not math.isfinite(args.dfl) or args.dfl < 0.0):
        parser.error("--dfl must be finite and non-negative")
    if args.degrees is not None and (
        not math.isfinite(args.degrees) or not 0.0 <= args.degrees <= 180.0
    ):
        parser.error("--degrees must be finite and satisfy 0.0 <= value <= 180.0")

    if args.resume:
        unsupported = [
            option
            for option in (
                "model",
                "load_weights",
                "epochs",
                "seed",
                "name",
                "optimizer",
                "lr0",
                "lrf",
                "warmup_epochs",
                "box",
                "box_iou_loss",
                "dfl",
                "cls_pw",
                "hsv_h",
                "hsv_s",
                "hsv_v",
                "mosaic",
                "close_mosaic",
                "scale",
                "degrees",
                "multi_scale",
                "extra_channel_init",
            )
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
    kwargs = {
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
    for option in (
        "optimizer",
        "lr0",
        "lrf",
        "warmup_epochs",
        "box",
        "dfl",
        "cls_pw",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "mosaic",
        "close_mosaic",
        "scale",
        "degrees",
        "multi_scale",
    ):
        value = getattr(args, option)
        if value is not None:
            kwargs[option] = value
    return kwargs


def zero_extra_input_channel_weights(trainer: object, base_channels: int = 3) -> None:
    """Zero new first-layer channels in both the train model and its EMA copy.

    Ultralytics transfers pretrained weights into the first three channels of a
    wider input convolution and leaves the remaining channels randomly
    initialized. Zeroing only those new slices makes the expanded model's
    initial response depend on the pretrained channels alone. The slices remain
    trainable and receive gradients on the first optimizer step.
    """
    import torch

    from ultralytics.utils import LOGGER
    from ultralytics.utils.torch_utils import unwrap_model

    targets = [("model", trainer.model)]
    ema = getattr(trainer, "ema", None)
    if ema is not None and getattr(ema, "ema", None) is not None:
        targets.append(("ema", ema.ema))

    initialized_channels: int | None = None
    for target_name, target in targets:
        unwrapped = unwrap_model(target)
        try:
            weight = unwrapped.model[0].conv.weight
        except (AttributeError, IndexError, TypeError) as error:
            raise RuntimeError(
                "Could not locate the first YOLO convolution for extra-channel initialization"
            ) from error
        input_channels = int(weight.shape[1])
        if input_channels <= base_channels:
            raise RuntimeError(
                f"--extra-channel-init zero requires more than {base_channels} input channels; "
                f"the constructed model has {input_channels}"
            )
        with torch.no_grad():
            weight[:, base_channels:].zero_()
        initialized_channels = input_channels - base_channels
        LOGGER.info(
            f"Zero-initialized {initialized_channels} extra input channels in the {target_name} first convolution"
        )
    trainer.extra_channel_initialization = {
        "method": "zero",
        "base_channels": base_channels,
        "extra_channels": initialized_channels,
    }


def main(argv: Sequence[str] | None = None) -> None:
    from ultralytics import YOLO

    from hsi_detection.box_loss import (
        EIoUDetectionTrainer,
        checkpoint_box_iou_loss,
        install_box_iou_loss,
        record_box_iou_loss,
    )

    args = parse_args(argv)
    model_source = str(args.resume.resolve()) if args.resume else (args.model or "yolo26n.pt")
    model = YOLO(model_source)
    if args.load_weights is not None:
        model.load(str(args.load_weights.resolve()))
    box_iou_loss = (
        checkpoint_box_iou_loss(model.model)
        if args.resume
        else (args.box_iou_loss or "ciou")
    )
    record_box_iou_loss(model.model, box_iou_loss)
    install_box_iou_loss(box_iou_loss)
    if args.extra_channel_init == "zero":
        model.add_callback("on_pretrain_routine_end", zero_extra_input_channel_weights)
    from hsi_detection.spectral_stem import SpectralDetectionTrainer, has_spectral_stem

    use_spectral_trainer = args.spectral_stem or has_spectral_stem(model.model)
    train_kwargs = build_train_kwargs(args)
    if use_spectral_trainer:
        model.train(trainer=SpectralDetectionTrainer, **train_kwargs)
    elif box_iou_loss == "eiou":
        model.train(trainer=EIoUDetectionTrainer, **train_kwargs)
    else:
        model.train(**train_kwargs)


if __name__ == "__main__":
    main()
