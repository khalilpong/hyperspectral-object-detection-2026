from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parents[1]
MAKE_KERNEL = PROJECT_ROOT / "kaggle_remote" / "make_kernel.py"
RUNNER = PROJECT_ROOT / "kaggle_remote" / "run_hsi_yolo26.py"


def _load_make_kernel():
    spec = importlib.util.spec_from_file_location("make_kernel_under_test", MAKE_KERNEL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_percentile_validation_and_stable_tags() -> None:
    module = _load_make_kernel()

    module._validate_percentiles(0.5, 99.5)
    module._validate_percentiles(1.0, 99.0)
    assert module._percentile_tag(0.5) == "005"
    assert module._percentile_tag(99.5) == "995"
    assert module._percentile_tag(1.0) == "010"
    assert module._percentile_tag(99.0) == "990"
    with pytest.raises(ValueError):
        module._validate_percentiles(99.0, 1.0)


def test_make_kernel_renders_isolated_p1_p99_variant(tmp_path: Path, monkeypatch) -> None:
    module = _load_make_kernel()
    module.HERE = tmp_path
    (tmp_path / "run_hsi_yolo26.py").write_text(
        RUNNER.read_text(encoding="utf-8"), encoding="utf-8"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_kernel.py",
            "--mode",
            "full",
            "--model",
            "yolo26m.pt",
            "--epochs",
            "30",
            "--multiscale",
            "--lower-percentile",
            "1",
            "--upper-percentile",
            "99",
        ],
    )

    module.main()

    folder = tmp_path / "kernel_full_p010_990"
    rendered = (folder / "run_hsi_yolo26.py").read_text(encoding="utf-8")
    metadata = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert '"RUN_NAME": "kaggle_full_yolo26m_p010_990_e30"' in rendered
    assert '"LOWER_PERCENTILE": 1.0' in rendered
    assert '"UPPER_PERCENTILE": 99.0' in rendered
    assert '"SEED": 2026' in rendered
    assert metadata["id"] == "zephyrpong/hsi-yolo26m-p010-990-full"
    assert metadata["is_private"] is True
