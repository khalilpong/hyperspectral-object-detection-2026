from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "train_baseline.py"
SPEC = importlib.util.spec_from_file_location("train_baseline", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
train_baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_baseline)


def test_new_run_keeps_documented_defaults() -> None:
    args = train_baseline.parse_args([])
    kwargs = train_baseline.build_train_kwargs(args)

    assert kwargs["epochs"] == 30
    assert kwargs["imgsz"] == 640
    assert kwargs["batch"] == 8
    assert kwargs["device"] == "0"
    assert kwargs["workers"] == 4
    assert kwargs["seed"] == 2026
    assert kwargs["val"] is True
    assert kwargs["plots"] is True
    assert kwargs["resume"] is False
    assert "cls_pw" not in kwargs
    assert "dfl" not in kwargs
    assert "degrees" not in kwargs
    assert args.box_iou_loss is None


def test_resume_does_not_apply_fresh_run_defaults() -> None:
    args = train_baseline.parse_args(["--resume", "runs/example/weights/last.pt"])

    assert train_baseline.build_train_kwargs(args) == {"resume": True}


def test_resume_only_applies_explicit_supported_overrides() -> None:
    args = train_baseline.parse_args(
        [
            "--resume",
            "runs/example/weights/last.pt",
            "--imgsz",
            "1024",
            "--batch",
            "3",
            "--device",
            "0",
            "--workers",
            "2",
            "--no-plots",
        ]
    )

    assert train_baseline.build_train_kwargs(args) == {
        "resume": True,
        "imgsz": 1024,
        "batch": 3,
        "device": "0",
        "workers": 2,
        "plots": False,
    }


def test_resume_rejects_unsupported_checkpoint_override() -> None:
    with pytest.raises(SystemExit):
        train_baseline.parse_args(
            ["--resume", "runs/example/weights/last.pt", "--epochs", "40"]
        )


def test_new_run_passes_explicit_augmentation_and_optimizer_overrides() -> None:
    args = train_baseline.parse_args(
        [
            "--optimizer",
            "AdamW",
            "--lr0",
            "0.0002",
            "--lrf",
            "0.1",
            "--warmup-epochs",
            "1",
            "--box",
            "10",
            "--dfl",
            "2.0",
            "--cls-pw",
            "0.25",
            "--hsv-h",
            "0",
            "--hsv-s",
            "0",
            "--hsv-v",
            "0.2",
            "--mosaic",
            "0",
            "--close-mosaic",
            "0",
            "--scale",
            "0.75",
            "--degrees",
            "5",
            "--multi-scale",
            "0.25",
        ]
    )

    kwargs = train_baseline.build_train_kwargs(args)

    assert kwargs["optimizer"] == "AdamW"
    assert kwargs["lr0"] == 0.0002
    assert kwargs["lrf"] == 0.1
    assert kwargs["warmup_epochs"] == 1.0
    assert kwargs["box"] == 10.0
    assert kwargs["dfl"] == 2.0
    assert kwargs["cls_pw"] == 0.25
    assert kwargs["hsv_h"] == 0.0
    assert kwargs["hsv_s"] == 0.0
    assert kwargs["hsv_v"] == 0.2
    assert kwargs["mosaic"] == 0.0
    assert kwargs["close_mosaic"] == 0
    assert kwargs["scale"] == 0.75
    assert kwargs["degrees"] == 5.0
    assert kwargs["multi_scale"] == 0.25


def test_resume_rejects_optimizer_or_augmentation_override() -> None:
    with pytest.raises(SystemExit):
        train_baseline.parse_args(
            [
                "--resume",
                "runs/example/weights/last.pt",
                "--lr0",
                "0.001",
                "--hsv-s",
                "0",
            ]
        )


def test_cls_pw_validates_range_and_cannot_override_resume() -> None:
    assert train_baseline.parse_args(["--cls-pw", "0.25"]).cls_pw == 0.25

    for invalid in ("-0.01", "1.01"):
        with pytest.raises(SystemExit):
            train_baseline.parse_args(["--cls-pw", invalid])

    with pytest.raises(SystemExit):
        train_baseline.parse_args(
            ["--resume", "runs/example/weights/last.pt", "--cls-pw", "0.25"]
        )


def test_dfl_validates_range_and_cannot_override_resume() -> None:
    assert train_baseline.parse_args(["--dfl", "2.0"]).dfl == 2.0

    for invalid in ("-0.01", "nan", "inf"):
        with pytest.raises(SystemExit):
            train_baseline.parse_args(["--dfl", invalid])

    with pytest.raises(SystemExit):
        train_baseline.parse_args(
            ["--resume", "runs/example/weights/last.pt", "--dfl", "2.0"]
        )


def test_box_iou_loss_is_opt_in_and_cannot_override_resume() -> None:
    assert train_baseline.parse_args(["--box-iou-loss", "eiou"]).box_iou_loss == "eiou"

    with pytest.raises(SystemExit):
        train_baseline.parse_args(
            [
                "--resume",
                "runs/example/weights/last.pt",
                "--box-iou-loss",
                "eiou",
            ]
        )


def test_eiou_rejects_spectral_stem_multi_variable_variant() -> None:
    with pytest.raises(SystemExit):
        train_baseline.parse_args(["--spectral-stem", "--box-iou-loss", "eiou"])


def test_degrees_validates_range_and_cannot_override_resume() -> None:
    assert train_baseline.parse_args(["--degrees", "5"]).degrees == 5.0

    for invalid in ("-0.01", "180.01", "nan", "inf"):
        with pytest.raises(SystemExit):
            train_baseline.parse_args(["--degrees", invalid])

    with pytest.raises(SystemExit):
        train_baseline.parse_args(
            ["--resume", "runs/example/weights/last.pt", "--degrees", "5"]
        )


def test_resume_rejects_extra_channel_initialization() -> None:
    with pytest.raises(SystemExit):
        train_baseline.parse_args(
            [
                "--resume",
                "runs/example/weights/last.pt",
                "--extra-channel-init",
                "zero",
            ]
        )


def test_spectral_stem_rejects_extra_channel_initialization() -> None:
    with pytest.raises(SystemExit):
        train_baseline.parse_args(
            ["--spectral-stem", "--extra-channel-init", "zero"]
        )


def test_resume_recovers_spectral_stem_without_architecture_flag() -> None:
    args = train_baseline.parse_args(["--resume", "runs/example/weights/last.pt"])
    assert args.spectral_stem is False

    with pytest.raises(SystemExit):
        train_baseline.parse_args(
            ["--resume", "runs/example/weights/last.pt", "--spectral-stem"]
        )


def test_resume_rejects_architecture_weight_transfer() -> None:
    with pytest.raises(SystemExit):
        train_baseline.parse_args(
            [
                "--resume",
                "runs/example/weights/last.pt",
                "--load-weights",
                "yolo26s.pt",
            ]
        )


def test_zero_extra_input_channel_weights_preserves_first_three_channels() -> None:
    class FirstBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(5, 2, 1, bias=False)

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.ModuleList([FirstBlock()])

    class Ema:
        def __init__(self) -> None:
            self.ema = Model()

    class Trainer:
        def __init__(self) -> None:
            self.model = Model()
            self.ema = Ema()

    trainer = Trainer()
    with torch.no_grad():
        trainer.model.model[0].conv.weight.fill_(1.0)
        trainer.ema.ema.model[0].conv.weight.fill_(2.0)

    train_baseline.zero_extra_input_channel_weights(trainer)

    assert torch.all(trainer.model.model[0].conv.weight[:, :3] == 1.0)
    assert torch.all(trainer.model.model[0].conv.weight[:, 3:] == 0.0)
    assert torch.all(trainer.ema.ema.model[0].conv.weight[:, :3] == 2.0)
    assert torch.all(trainer.ema.ema.model[0].conv.weight[:, 3:] == 0.0)
    assert trainer.extra_channel_initialization == {
        "method": "zero",
        "base_channels": 3,
        "extra_channels": 2,
    }
