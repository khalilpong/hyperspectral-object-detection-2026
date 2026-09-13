from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


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
