from __future__ import annotations

import argparse
import json
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
    parser.add_argument(
        "--momentum",
        type=float,
        help="SGD momentum or Adam beta1. Omit to keep the Ultralytics default.",
    )
    parser.add_argument(
        "--warmup-bias-lr",
        type=float,
        help="Initial bias learning rate during warmup.",
    )
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
    if args.lr0 is not None and (not math.isfinite(args.lr0) or args.lr0 <= 0.0):
        parser.error("--lr0 must be finite and positive")
    if args.momentum is not None and (
        not math.isfinite(args.momentum) or not 0.0 <= args.momentum < 1.0
    ):
        parser.error("--momentum must be finite and satisfy 0.0 <= value < 1.0")
    if args.warmup_bias_lr is not None and (
        not math.isfinite(args.warmup_bias_lr) or args.warmup_bias_lr < 0.0
    ):
        parser.error("--warmup-bias-lr must be finite and non-negative")
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
                "momentum",
                "warmup_bias_lr",
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
        "momentum",
        "warmup_bias_lr",
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


def record_optimizer_contract(trainer: object) -> None:
    """Persist the optimizer values that are effective after Ultralytics auto-selection.

    ``optimizer=auto`` mutates the learning rate, beta1, and warmup-bias LR after
    parsing the requested arguments. ``args.yaml`` therefore records request-time
    values rather than the effective optimizer. This callback runs after optimizer
    and scheduler construction and provides an auditable runtime contract.
    """
    from ultralytics.utils import LOGGER

    optimizer = trainer.optimizer
    groups = optimizer.param_groups

    def unique_numbers(key: str) -> list[float]:
        return sorted({float(group[key]) for group in groups if key in group})

    beta1_values = sorted(
        {
            float(group["betas"][0])
            for group in groups
            if "betas" in group and group["betas"] is not None
        }
    )
    args = trainer.args
    contract = {
        "optimizer_argument": str(args.optimizer),
        "lr0_argument": float(args.lr0),
        "momentum_argument": float(args.momentum),
        "effective_optimizer": type(optimizer).__name__,
        "initial_lr_values": unique_numbers("initial_lr"),
        "current_lr_values": unique_numbers("lr"),
        "beta1_values": beta1_values,
        "effective_warmup_bias_lr": float(args.warmup_bias_lr),
        "warmup_epochs_argument": float(args.warmup_epochs),
        "lrf_argument": float(args.lrf),
        "weight_decay_argument": float(args.weight_decay),
        "weight_decay_values": unique_numbers("weight_decay"),
    }
    output = Path(trainer.save_dir) / "optimizer_contract.json"
    output.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info("HSI optimizer contract: " + json.dumps(contract, sort_keys=True))


def record_yolo_model_contract(trainer: object) -> None:
    """Persist the effective YOLO detection-head architecture before training.

    A YAML-defined architecture can transfer most parameters from a pretrained
    checkpoint while changing a small head detail such as ``reg_max``.  The
    requested YAML filename alone is not sufficient evidence that the trainer
    actually built the intended 16-channel, task-specific model, so record the
    live model after Ultralytics has applied the dataset channel/class overrides.
    """
    import torch

    from ultralytics.utils import LOGGER
    from ultralytics.utils.torch_utils import unwrap_model

    model = unwrap_model(trainer.model)
    try:
        head = model.model[-1]
    except (AttributeError, IndexError, TypeError) as error:
        raise RuntimeError("Could not locate the YOLO detection head") from error
    if not hasattr(head, "reg_max") or not hasattr(head, "nc"):
        raise RuntimeError(
            f"Expected a YOLO detection head, got {type(head).__name__}"
        )

    first_conv = next(
        (module for module in model.modules() if isinstance(module, torch.nn.Conv2d)),
        None,
    )
    if first_conv is None:
        raise RuntimeError("Could not locate the first convolution in the YOLO model")

    def output_channels(branch_name: str) -> list[int] | None:
        branch = getattr(head, branch_name, None)
        if branch is None:
            return None
        values: list[int] = []
        for block in branch:
            try:
                layer = block[-1]
                values.append(int(layer.out_channels))
            except (AttributeError, IndexError, TypeError) as error:
                raise RuntimeError(
                    f"Could not inspect {branch_name} output channels"
                ) from error
        return values

    yaml_metadata = getattr(model, "yaml", {})
    yaml_reg_max = (
        int(yaml_metadata["reg_max"])
        if isinstance(yaml_metadata, dict) and "reg_max" in yaml_metadata
        else None
    )
    contract = {
        "model_class": type(model).__name__,
        "head_class": type(head).__name__,
        "yaml_file": str(getattr(model, "yaml_file", "")),
        "yaml_reg_max": yaml_reg_max,
        "reg_max": int(head.reg_max),
        "dfl_module": type(head.dfl).__name__,
        "dfl_is_identity": isinstance(head.dfl, torch.nn.Identity),
        "end2end": bool(getattr(head, "end2end", False)),
        "nc": int(head.nc),
        "names_count": len(getattr(model, "names", {})),
        "first_input_channels": int(first_conv.in_channels),
        "box_output_channels": output_channels("cv2"),
        "one2one_box_output_channels": output_channels("one2one_cv2"),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    output = Path(trainer.save_dir) / "model_contract.json"
    output.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    trainer.yolo_model_contract = contract
    LOGGER.info("HSI YOLO model contract: " + json.dumps(contract, sort_keys=True))


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
    model.add_callback("on_pretrain_routine_end", record_yolo_model_contract)
    model.add_callback("on_pretrain_routine_end", record_optimizer_contract)
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
