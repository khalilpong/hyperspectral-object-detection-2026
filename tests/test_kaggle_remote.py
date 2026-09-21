from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parents[1]
MAKE_KERNEL = PROJECT_ROOT / "kaggle_remote" / "make_kernel.py"
CROP_DATASET_BUILDER = PROJECT_ROOT / "kaggle_remote" / "make_crop_code_dataset.py"
RUNNER = PROJECT_ROOT / "kaggle_remote" / "run_hsi_yolo26.py"


def _load_make_kernel():
    spec = importlib.util.spec_from_file_location("make_kernel_under_test", MAKE_KERNEL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_crop_dataset_builder():
    spec = importlib.util.spec_from_file_location(
        "make_crop_code_dataset_under_test", CROP_DATASET_BUILDER
    )
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
    module._validate_cls_pw(0.25)
    assert module._fraction_tag(0.25) == "025"
    assert module._fraction_tag(0.5) == "050"
    with pytest.raises(ValueError):
        module._validate_cls_pw(1.01)
    module._validate_scale(0.0)
    module._validate_scale(0.3)
    module._validate_scale(1.0)
    with pytest.raises(ValueError):
        module._validate_scale(-0.01)
    with pytest.raises(ValueError):
        module._validate_scale(1.01)


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


def test_make_kernel_renders_isolated_zero_channel_init_variant(
    tmp_path: Path, monkeypatch
) -> None:
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
            "ablation",
            "--model",
            "yolo26m.pt",
            "--epochs",
            "30",
            "--extra-channel-init",
            "zero",
        ],
    )

    module.main()

    folder = tmp_path / "kernel_ablation_xczero"
    rendered = (folder / "run_hsi_yolo26.py").read_text(encoding="utf-8")
    metadata = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert '"RUN_NAME": "kaggle_ablation_yolo26m_xczero_e30"' in rendered
    assert '"EXTRA_CHANNEL_INIT": "zero"' in rendered
    assert metadata["id"] == "zephyrpong/hsi-yolo26m-xczero-ablation"
    assert metadata["is_private"] is True
    assert metadata["dataset_sources"] == [
        "zephyrpong/hsi-detection-code",
        "zephyrpong/hsi-competition-raw",
    ]


def test_make_kernel_renders_isolated_spectral_stem_variant(
    tmp_path: Path, monkeypatch
) -> None:
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
            "ablation",
            "--model",
            "yolo26m.pt",
            "--epochs",
            "30",
            "--spectral-stem",
        ],
    )

    module.main()

    folder = tmp_path / "kernel_ablation_stem"
    rendered = (folder / "run_hsi_yolo26.py").read_text(encoding="utf-8")
    metadata = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert '"RUN_NAME": "kaggle_ablation_yolo26m_stem_e30"' in rendered
    assert '"SPECTRAL_STEM": 1' in rendered
    assert '"EXTRA_CHANNEL_INIT": "random"' in rendered
    assert metadata["id"] == "zephyrpong/hsi-yolo26m-stem-ablation"
    assert metadata["is_private"] is True
    assert metadata["dataset_sources"] == [
        "zephyrpong/hsi-detection-code",
        "zephyrpong/hsi-competition-raw",
    ]


def test_make_kernel_renders_isolated_phase_aware_variant(
    tmp_path: Path, monkeypatch
) -> None:
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
            "ablation",
            "--model",
            "yolo26m.pt",
            "--epochs",
            "30",
            "--data",
            "hsi16_phase",
            "--phase-target-long-edge",
            "1024",
        ],
    )

    module.main()

    folder = tmp_path / "kernel_ablation_phase"
    rendered = (folder / "run_hsi_yolo26.py").read_text(encoding="utf-8")
    metadata = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert '"RUN_NAME": "kaggle_ablation_yolo26m_phase_e30"' in rendered
    assert '"DATA": "hsi16_phase"' in rendered
    assert '"PHASE_TARGET_LONG_EDGE": 1024' in rendered
    assert '"EXTRA_CHANNEL_INIT": "random"' in rendered
    assert '"SPECTRAL_STEM": 0' in rendered
    assert '"OBJECT_CROPS": 0' in rendered
    assert '"TILE_INFERENCE": 0' in rendered
    assert metadata["id"] == "zephyrpong/hsi-yolo26m-phase-ablation"
    assert metadata["is_private"] is True
    assert metadata["dataset_sources"] == [
        "zephyrpong/hsi-detection-code",
        "zephyrpong/hsi-competition-raw",
    ]


def test_make_kernel_renders_isolated_cls_pw_variant(tmp_path: Path, monkeypatch) -> None:
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
            "ablation",
            "--model",
            "yolo26m.pt",
            "--epochs",
            "30",
            "--cls-pw",
            "0.25",
        ],
    )

    module.main()

    folder = tmp_path / "kernel_ablation_clspw025"
    rendered = (folder / "run_hsi_yolo26.py").read_text(encoding="utf-8")
    metadata = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert '"RUN_NAME": "kaggle_ablation_yolo26m_clspw025_e30"' in rendered
    assert '"CLS_PW": 0.25' in rendered
    assert '"SCALE": 0.5' in rendered
    assert '"DATA": "hsi16"' in rendered
    assert '"EXTRA_CHANNEL_INIT": "random"' in rendered
    assert '"SPECTRAL_STEM": 0' in rendered
    assert '"OBJECT_CROPS": 0' in rendered
    assert '"TILE_INFERENCE": 0' in rendered
    assert metadata["id"] == "zephyrpong/hsi-yolo26m-clspw025-ablation"
    assert metadata["is_private"] is True
    assert metadata["dataset_sources"] == [
        "zephyrpong/hsi-detection-code",
        "zephyrpong/hsi-competition-raw",
    ]


def test_make_kernel_renders_isolated_random_affine_scale_variant(
    tmp_path: Path, monkeypatch
) -> None:
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
            "ablation",
            "--model",
            "yolo26m.pt",
            "--epochs",
            "30",
            "--scale",
            "0.30",
        ],
    )

    module.main()

    folder = tmp_path / "kernel_ablation_scale030"
    rendered = (folder / "run_hsi_yolo26.py").read_text(encoding="utf-8")
    metadata = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert '"RUN_NAME": "kaggle_ablation_yolo26m_scale030_e30"' in rendered
    assert '"SCALE": 0.3' in rendered
    assert '"CLS_PW": 0.0' in rendered
    assert '"DATA": "hsi16"' in rendered
    assert '"EXTRA_CHANNEL_INIT": "random"' in rendered
    assert '"SPECTRAL_STEM": 0' in rendered
    assert '"OBJECT_CROPS": 0' in rendered
    assert '"TILE_INFERENCE": 0' in rendered
    assert metadata["id"] == "zephyrpong/hsi-yolo26m-scale030-ablation"
    assert metadata["is_private"] is True
    assert metadata["dataset_sources"] == [
        "zephyrpong/hsi-detection-code",
        "zephyrpong/hsi-competition-raw",
    ]


def test_make_kernel_rejects_conflicting_spectral_initialization(
    tmp_path: Path, monkeypatch
) -> None:
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
            "ablation",
            "--model",
            "yolo26m.pt",
            "--epochs",
            "30",
            "--spectral-stem",
            "--extra-channel-init",
            "zero",
        ],
    )

    with pytest.raises(SystemExit):
        module.main()


def test_make_kernel_renders_object_crop_and_tile_variant(
    tmp_path: Path, monkeypatch
) -> None:
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
            "ablation",
            "--model",
            "yolo26m.pt",
            "--epochs",
            "30",
            "--object-crops",
            "--tile-inference",
        ],
    )

    module.main()

    folder = tmp_path / "kernel_ablation_crop_tile"
    rendered = (folder / "run_hsi_yolo26.py").read_text(encoding="utf-8")
    metadata = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert '"RUN_NAME": "kaggle_ablation_yolo26m_crop_tile_e30"' in rendered
    assert '"OBJECT_CROPS": 1' in rendered
    assert '"TILE_INFERENCE": 1' in rendered
    assert metadata["id"] == "zephyrpong/hsi-yolo26m-crop-tile-ablation"
    assert metadata["is_private"] is True
    assert metadata["dataset_sources"] == [
        "zephyrpong/hsi-detection-code",
        "zephyrpong/hsi-competition-raw",
        "zephyrpong/hsi-object-crop-code",
    ]


def test_crop_code_dataset_contains_only_the_three_variant_sources(tmp_path: Path) -> None:
    module = _load_crop_dataset_builder()

    manifest = module.stage(tmp_path)

    expected_sources = {
        "hsi_detection.tiling.py": "src/hsi_detection/tiling.py",
        "scripts.prepare_object_crops.py": "scripts/prepare_object_crops.py",
        "scripts.predict_submission.py": "scripts/predict_submission.py",
    }
    assert {path.name for path in tmp_path.iterdir()} == {
        *expected_sources,
        "dataset-metadata.json",
        "payload-manifest.json",
    }
    assert manifest["dataset_id"] == "zephyrpong/hsi-object-crop-code"
    assert {
        name: details["source"] for name, details in manifest["files"].items()
    } == expected_sources
    for name, details in manifest["files"].items():
        assert details["bytes"] == (tmp_path / name).stat().st_size
        assert details["sha256"] == module.sha256(tmp_path / name)


def test_crop_code_dataset_rejects_unexpected_payload(tmp_path: Path) -> None:
    module = _load_crop_dataset_builder()
    (tmp_path / "unexpected.bin").write_bytes(b"do not upload")

    with pytest.raises(RuntimeError, match="未预期文件"):
        module.stage(tmp_path)
