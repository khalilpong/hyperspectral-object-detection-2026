from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "train_rtdetr_hsi.py"
SPEC = importlib.util.spec_from_file_location("train_rtdetr_hsi", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
train_rtdetr_hsi = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_rtdetr_hsi)


def test_rtdetr_training_defaults_disable_amp_and_determinism(tmp_path: Path) -> None:
    data = tmp_path / "dataset.yaml"
    args = train_rtdetr_hsi.parse_args(["--data", str(data)])
    kwargs = train_rtdetr_hsi.build_train_kwargs(args)

    assert args.model == "rtdetr-l.pt"
    assert kwargs["epochs"] == 30
    assert kwargs["imgsz"] == 1024
    assert kwargs["batch"] == 2
    assert kwargs["amp"] is False
    assert kwargs["deterministic"] is False
    assert kwargs["data"] == str(data.resolve())


@pytest.mark.parametrize(
    "arguments",
    (["--epochs", "0"], ["--imgsz", "0"], ["--batch", "0"], ["--workers", "-1"]),
)
def test_rtdetr_training_rejects_invalid_sizes(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        train_rtdetr_hsi.parse_args(["--data", "dataset.yaml", *arguments])
