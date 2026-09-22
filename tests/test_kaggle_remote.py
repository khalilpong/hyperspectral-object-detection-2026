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


def _load_runner():
    spec = importlib.util.spec_from_file_location("remote_runner_under_test", RUNNER)
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
    module._validate_degrees(0.0)
    module._validate_degrees(5.0)
    module._validate_degrees(180.0)
    with pytest.raises(ValueError):
        module._validate_degrees(-0.01)
    with pytest.raises(ValueError):
        module._validate_degrees(180.01)
    module._validate_dfl(0.0)
    module._validate_dfl(1.5)
    module._validate_dfl(2.0)
    with pytest.raises(ValueError):
        module._validate_dfl(-0.01)
    with pytest.raises(ValueError):
        module._validate_dfl(float("nan"))
    module._validate_inference_fusion(0.74, 0.125)
    with pytest.raises(ValueError):
        module._validate_inference_fusion(0.0, 0.125)
    with pytest.raises(ValueError):
        module._validate_inference_fusion(0.74, -0.01)
    assert module._parse_band_order(
        "13,8,5,0,1,2,3,4,6,7,9,10,11,12,14,15"
    ) == module.TARGET_HSI16_ORDER
    with pytest.raises(module.argparse.ArgumentTypeError):
        module._parse_band_order("0,1,2")


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


def test_make_kernel_renders_isolated_dfl_variant(tmp_path: Path, monkeypatch) -> None:
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
            "--dfl",
            "2.0",
        ],
    )

    module.main()

    folder = tmp_path / "kernel_ablation_dfl200"
    rendered = (folder / "run_hsi_yolo26.py").read_text(encoding="utf-8")
    metadata = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert '"RUN_NAME": "kaggle_ablation_yolo26m_dfl200_e30"' in rendered
    assert '"DFL": 2.0' in rendered
    assert '"SCALE": 0.5' in rendered
    assert '"CLS_PW": 0.0' in rendered
    assert '"DATA": "hsi16"' in rendered
    assert '"EXTRA_CHANNEL_INIT": "random"' in rendered
    assert '"SPECTRAL_STEM": 0' in rendered
    assert '"OBJECT_CROPS": 0' in rendered
    assert '"TILE_INFERENCE": 0' in rendered
    assert metadata["id"] == "zephyrpong/hsi-yolo26m-dfl200-ablation"
    assert metadata["is_private"] is True
    assert metadata["dataset_sources"] == [
        "zephyrpong/hsi-detection-code",
        "zephyrpong/hsi-competition-raw",
    ]


def test_make_kernel_renders_isolated_eiou_variant(tmp_path: Path, monkeypatch) -> None:
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
            "--box-iou-loss",
            "eiou",
        ],
    )

    module.main()

    folder = tmp_path / "kernel_ablation_eiou"
    rendered = (folder / "run_hsi_yolo26.py").read_text(encoding="utf-8")
    metadata = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert '"RUN_NAME": "kaggle_ablation_yolo26m_eiou_e30"' in rendered
    assert '"BOX_IOU_LOSS": "eiou"' in rendered
    assert '"DFL": 1.5' in rendered
    assert '"DEGREES": 0.0' in rendered
    assert metadata["id"] == "zephyrpong/hsi-yolo26m-eiou-ablation"
    assert metadata["is_private"] is True


def test_make_kernel_renders_isolated_regmax16_variant(
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
            "--reg-max",
            "16",
        ],
    )

    module.main()

    folder = tmp_path / "kernel_ablation_rm16"
    rendered = (folder / "run_hsi_yolo26.py").read_text(encoding="utf-8")
    metadata = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert '"RUN_NAME": "kaggle_ablation_yolo26m_rm16_e30"' in rendered
    assert '"MODEL": "yolo26m.pt"' in rendered
    assert '"MODEL_SOURCE_KIND": "yaml_transfer"' in rendered
    assert '"MODEL_YAML": "yolo26m-regmax16.yaml"' in rendered
    assert '"REG_MAX": 16' in rendered
    assert '"DFL": 1.5' in rendered
    assert '"BOX_IOU_LOSS": "ciou"' in rendered
    assert metadata["id"] == "zephyrpong/hsi-yolo26m-rm16-ablation"
    assert metadata["is_private"] is True


def test_make_kernel_renders_yolo11m_checkpoint_native_candidate(
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
            "yolo11m.pt",
            "--model-source-kind",
            "checkpoint_native",
            "--epochs",
            "30",
            "--reg-max",
            "16",
        ],
    )

    module.main()

    folder = tmp_path / "kernel_ablation_yolo11m_rm16"
    rendered = (folder / "run_hsi_yolo26.py").read_text(encoding="utf-8")
    metadata = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert '"RUN_NAME": "kaggle_ablation_yolo11m_rm16_e30"' in rendered
    assert '"MODEL": "yolo11m.pt"' in rendered
    assert '"MODEL_SOURCE_KIND": "checkpoint_native"' in rendered
    assert '"MODEL_YAML": ""' in rendered
    assert '"REG_MAX": 16' in rendered
    assert metadata["id"] == "zephyrpong/hsi-yolo11m-rm16-ablation"
    assert metadata["is_private"] is True


@pytest.mark.parametrize(
    "arguments",
    (
        ["--reg-max", "1"],
        ["--reg-max", "16", "--model-source-kind", "yaml_transfer"],
        ["--mode", "full", "--epochs", "30", "--reg-max", "16"],
    ),
)
def test_make_kernel_rejects_yolo11m_source_or_fixed_contract_drift(
    tmp_path: Path, monkeypatch, arguments: list[str]
) -> None:
    module = _load_make_kernel()
    module.HERE = tmp_path
    (tmp_path / "run_hsi_yolo26.py").write_text(
        RUNNER.read_text(encoding="utf-8"), encoding="utf-8"
    )
    base = ["--mode", "ablation", "--epochs", "30"]
    if "--mode" in arguments:
        base = []
    monkeypatch.setattr(
        sys,
        "argv",
        ["make_kernel.py", *base, "--model", "yolo11m.pt", *arguments],
    )

    with pytest.raises(SystemExit):
        module.main()


@pytest.mark.parametrize(
    "arguments",
    (
        ["--mode", "full", "--epochs", "30"],
        ["--mode", "ablation", "--epochs", "1"],
        ["--mode", "ablation", "--epochs", "30", "--seed", "7"],
        ["--mode", "ablation", "--epochs", "30", "--multiscale"],
        ["--mode", "ablation", "--epochs", "30", "--dfl", "2.0"],
        ["--mode", "ablation", "--epochs", "30", "--box-iou-loss", "eiou"],
    ),
)
def test_make_kernel_rejects_regmax16_contract_drift(
    tmp_path: Path, monkeypatch, arguments: list[str]
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
            *arguments,
            "--model",
            "yolo26m.pt",
            "--reg-max",
            "16",
        ],
    )

    with pytest.raises(SystemExit):
        module.main()


def test_make_kernel_renders_isolated_degrees_variant(tmp_path: Path, monkeypatch) -> None:
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
            "--degrees",
            "5",
        ],
    )

    module.main()

    folder = tmp_path / "kernel_ablation_deg5"
    rendered = (folder / "run_hsi_yolo26.py").read_text(encoding="utf-8")
    metadata = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert '"RUN_NAME": "kaggle_ablation_yolo26m_deg5_e30"' in rendered
    assert '"DEGREES": 5.0' in rendered
    assert '"SCALE": 0.5' in rendered
    assert metadata["id"] == "zephyrpong/hsi-yolo26m-deg5-ablation"
    assert metadata["is_private"] is True


def test_make_kernel_renders_clean_adamw_lr001_recipe(
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
            "--optimizer-recipe",
            "adamw_lr001",
        ],
    )

    module.main()

    folder = tmp_path / "kernel_ablation_lr001"
    rendered = (folder / "run_hsi_yolo26.py").read_text(encoding="utf-8")
    metadata = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert '"RUN_NAME": "kaggle_ablation_yolo26m_lr001_e30"' in rendered
    assert '"OPTIMIZER_RECIPE": "adamw_lr001"' in rendered
    assert '"BOX_IOU_LOSS": "ciou"' in rendered
    assert '"DFL": 1.5' in rendered
    assert metadata["id"] == "zephyrpong/hsi-yolo26m-lr001-ablation"
    assert metadata["is_private"] is True


def test_make_kernel_renders_isolated_custom_pseudo_rgb_band_order(
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
            "pseudo_rgb:13,8,5",
        ],
    )

    module.main()

    folder = tmp_path / "kernel_ablation_prgb1385"
    rendered = (folder / "run_hsi_yolo26.py").read_text(encoding="utf-8")
    metadata = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert '"RUN_NAME": "kaggle_ablation_yolo26m_prgb1385_e30"' in rendered
    assert '"DATA": "pseudo_rgb:13,8,5"' in rendered
    assert '"DEGREES": 0.0' in rendered
    assert '"SCALE": 0.5' in rendered
    assert '"DFL": 1.5' in rendered
    assert metadata["id"] == "zephyrpong/hsi-yolo26m-prgb1385-ablation"
    assert metadata["is_private"] is True
    assert metadata["dataset_sources"] == [
        "zephyrpong/hsi-detection-code",
        "zephyrpong/hsi-competition-raw",
    ]


def test_make_kernel_renders_isolated_ordinary_hsi16_band_order(
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
            "--band-order",
            "13,8,5,0,1,2,3,4,6,7,9,10,11,12,14,15",
        ],
    )

    module.main()

    folder = tmp_path / "kernel_ablation_order1385"
    rendered = (folder / "run_hsi_yolo26.py").read_text(encoding="utf-8")
    metadata = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert '"RUN_NAME": "kaggle_ablation_yolo26m_order1385_e30"' in rendered
    assert (
        '"BAND_ORDER": "13,8,5,0,1,2,3,4,6,7,9,10,11,12,14,15"'
        in rendered
    )
    assert '"REG_MAX": 1' in rendered
    assert '"BOX_IOU_LOSS": "ciou"' in rendered
    assert metadata["id"] == "zephyrpong/hsi-yolo26m-order1385-ablation"
    assert metadata["is_private"] is True


@pytest.mark.parametrize(
    "extra_arguments",
    (
        ["--mode", "full", "--epochs", "30"],
        ["--mode", "ablation", "--epochs", "1"],
        ["--mode", "ablation", "--epochs", "30", "--seed", "7"],
        ["--mode", "ablation", "--epochs", "30", "--multiscale"],
        ["--mode", "ablation", "--epochs", "30", "--reg-max", "16"],
        ["--mode", "ablation", "--epochs", "30", "--box-iou-loss", "eiou"],
        ["--mode", "ablation", "--epochs", "30", "--run-name", "ambiguous"],
    ),
)
def test_make_kernel_rejects_ordinary_hsi16_band_order_contract_drift(
    tmp_path: Path, monkeypatch, extra_arguments: list[str]
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
            *extra_arguments,
            "--model",
            "yolo26m.pt",
            "--band-order",
            "13,8,5,0,1,2,3,4,6,7,9,10,11,12,14,15",
        ],
    )

    with pytest.raises(SystemExit):
        module.main()


def test_make_kernel_renders_single_checkpoint_supported_multiscale_full_candidate(
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
            "full",
            "--model",
            "yolo26m.pt",
            "--epochs",
            "30",
            "--dfl",
            "2.0",
            "--multiscale",
            "--fusion-iou",
            "0.74",
            "--support-gain",
            "0.125",
        ],
    )

    module.main()

    folder = tmp_path / "kernel_full_dfl200_f074_sg0125"
    rendered = (folder / "run_hsi_yolo26.py").read_text(encoding="utf-8")
    metadata = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert '"MODEL": "yolo26m.pt"' in rendered
    assert '"DFL": 2.0' in rendered
    assert '"MULTISCALE": 1' in rendered
    assert '"FUSION_IOU": 0.74' in rendered
    assert '"SUPPORT_GAIN": 0.125' in rendered
    assert '"RUN_NAME": "kaggle_full_yolo26m_dfl200_e30"' in rendered
    assert metadata["id"] == "zephyrpong/hsi-yolo26m-dfl200-f074-sg0125-full"
    assert metadata["is_private"] is True


def test_make_kernel_renders_rtdetr_hsi16_smoke_with_safe_fallbacks(
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
            "smoke",
            "--architecture",
            "rtdetr",
            "--epochs",
            "1",
        ],
    )

    module.main()

    folder = tmp_path / "kernel_smoke_rtdetr"
    rendered = (folder / "run_hsi_yolo26.py").read_text(encoding="utf-8")
    metadata = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert '"ARCHITECTURE": "rtdetr"' in rendered
    assert '"MODEL": "rtdetr-l.pt"' in rendered
    assert '"ATTEMPTS": "2:2,1:2,1:0"' in rendered
    assert '"RUN_NAME": "kaggle_smoke_rtdetr-l_e1"' in rendered
    assert metadata["id"] == "zephyrpong/hsi-rtdetr-l-smoke"
    assert metadata["is_private"] is True
    assert metadata["dataset_sources"] == [
        "zephyrpong/hsi-detection-code",
        "zephyrpong/hsi-competition-raw",
    ]


def test_remote_runner_builds_rtdetr_command_without_yolo_only_gains(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_runner()
    monkeypatch.setattr(module, "ARCHITECTURE", "rtdetr")
    monkeypatch.setattr(module, "MODE", "smoke")

    command = module.build_training_command(
        tmp_path / "rtdetr-l.pt",
        tmp_path / "dataset.yaml",
        batch=2,
        device="0",
        workers=2,
    )

    assert command[1] == "scripts/train_rtdetr_hsi.py"
    assert command[command.index("--batch") + 1] == "2"
    assert command[command.index("--extra-channel-init") + 1] == "random"
    assert command[command.index("--num-denoising") + 1] == "100"
    assert "--dfl" not in command
    assert "--cls-pw" not in command
    assert "--scale" not in command
    assert "--degrees" not in command
    assert "--box-iou-loss" not in command
    assert "--no-val" not in command


def test_remote_runner_isolates_ordinary_hsi16_band_order_dataset_identity() -> None:
    module = _load_runner()
    target_order = (13, 8, 5, 0, 1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15)

    assert (
        module._hsi16_dataset_id(module.DEFAULT_BAND_ORDER, 0.5, 99.5)
        == "hsi16_shared_p005_995"
    )
    assert module._hsi16_dataset_id(target_order, 0.5, 99.5) == (
        "hsi16_order_13-8-5-0-1-2-3-4-6-7-9-10-11-12-14-15_p005_995"
    )


def test_remote_runner_builds_yolo_command_with_box_iou_loss(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_runner()
    monkeypatch.setattr(module, "ARCHITECTURE", "yolo")
    monkeypatch.setattr(module, "BOX_IOU_LOSS", "eiou")

    command = module.build_training_command(
        tmp_path / "yolo26m.pt",
        tmp_path / "dataset.yaml",
        batch=8,
        device="0",
        workers=2,
    )

    assert command[1] == "scripts/train_baseline.py"
    assert command[command.index("--box-iou-loss") + 1] == "eiou"


def test_remote_runner_builds_yaml_model_with_pretrained_weight_transfer(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_runner()
    monkeypatch.setattr(module, "ARCHITECTURE", "yolo")

    command = module.build_training_command(
        tmp_path / "yolo26m-regmax16.yaml",
        tmp_path / "dataset.yaml",
        batch=8,
        device="0",
        workers=2,
        load_weights=tmp_path / "yolo26m.pt",
    )

    assert command[command.index("--model") + 1].endswith("yolo26m-regmax16.yaml")
    assert command[command.index("--load-weights") + 1].endswith("yolo26m.pt")


def test_remote_runner_resolves_audited_yaml_and_packaged_weights(
    tmp_path: Path,
) -> None:
    module = _load_runner()
    weights = tmp_path / "yolo26m.pt"
    weights.write_bytes(b"pretrained")
    model_yaml = tmp_path / "yolo26m-regmax16.yaml"
    model_yaml.write_text("nc: 80\nreg_max: 16\n", encoding="utf-8")

    model_source, load_weights, audit = module.resolve_training_model_sources(
        tmp_path,
        model="yolo26m.pt",
        model_source_kind="yaml_transfer",
        model_yaml="yolo26m-regmax16.yaml",
        expected_reg_max=16,
    )

    assert model_source == model_yaml
    assert load_weights == weights
    assert audit is not None
    assert audit["model_source_kind"] == "yaml_transfer"
    assert audit["checkpoint_origin"] == "packaged_code_dataset"
    assert audit["reg_max"] == 16
    assert audit["pretrained_weights_bytes"] == len(b"pretrained")
    assert len(audit["model_yaml_sha256"]) == 64
    assert len(audit["pretrained_weights_sha256"]) == 64


def test_remote_runner_rejects_yaml_variant_without_packaged_weights(
    tmp_path: Path,
) -> None:
    module = _load_runner()
    (tmp_path / "yolo26m-regmax16.yaml").write_text(
        "nc: 80\nreg_max: 16\n", encoding="utf-8"
    )

    with pytest.raises(FileNotFoundError, match="预训练权重"):
        module.resolve_training_model_sources(
            tmp_path,
            model="yolo26m.pt",
            model_source_kind="yaml_transfer",
            model_yaml="yolo26m-regmax16.yaml",
            expected_reg_max=16,
        )


def test_remote_runner_rejects_yaml_regmax_drift(tmp_path: Path) -> None:
    module = _load_runner()
    (tmp_path / "yolo26m.pt").write_bytes(b"pretrained")
    (tmp_path / "yolo26m-regmax16.yaml").write_text(
        "nc: 80\nreg_max: 8\n", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="reg_max"):
        module.resolve_training_model_sources(
            tmp_path,
            model="yolo26m.pt",
            model_source_kind="yaml_transfer",
            model_yaml="yolo26m-regmax16.yaml",
            expected_reg_max=16,
        )


def test_remote_runner_audits_packaged_checkpoint_native_source(
    tmp_path: Path,
) -> None:
    module = _load_runner()
    weights = tmp_path / "yolo11m.pt"
    weights.write_bytes(b"official-checkpoint")

    model_source, load_weights, audit = module.resolve_training_model_sources(
        tmp_path,
        model="yolo11m.pt",
        model_source_kind="checkpoint_native",
        model_yaml="",
        expected_reg_max=16,
    )

    assert model_source == weights
    assert load_weights is None
    assert audit["model_source_kind"] == "checkpoint_native"
    assert audit["checkpoint_origin"] == "packaged_code_dataset"
    assert audit["pretrained_weights"] == "yolo11m.pt"
    assert audit["pretrained_weights_bytes"] == len(b"official-checkpoint")
    assert len(audit["pretrained_weights_sha256"]) == 64


def test_remote_runner_downloads_and_audits_missing_checkpoint_native_source(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_runner()
    downloaded = tmp_path / "yolo11m.pt"

    def fake_download(model: str) -> str:
        assert model == "yolo11m.pt"
        downloaded.write_bytes(b"downloaded-official-checkpoint")
        return str(downloaded)

    import ultralytics.utils.downloads

    monkeypatch.setattr(
        ultralytics.utils.downloads,
        "attempt_download_asset",
        fake_download,
    )

    model_source, load_weights, audit = module.resolve_training_model_sources(
        tmp_path / "empty-code-root",
        model="yolo11m.pt",
        model_source_kind="checkpoint_native",
        model_yaml="",
        expected_reg_max=16,
    )

    assert model_source == downloaded
    assert load_weights is None
    assert audit["checkpoint_origin"] == "official_asset_download"
    assert audit["pretrained_weights_bytes"] == len(
        b"downloaded-official-checkpoint"
    )


def test_remote_runner_builds_clean_adamw_lr001_command(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_runner()
    monkeypatch.setattr(module, "ARCHITECTURE", "yolo")
    monkeypatch.setattr(module, "OPTIMIZER_RECIPE", "adamw_lr001")

    command = module.build_training_command(
        tmp_path / "yolo26m.pt",
        tmp_path / "dataset.yaml",
        batch=8,
        device="0",
        workers=2,
    )

    assert command[command.index("--optimizer") + 1] == "AdamW"
    assert command[command.index("--lr0") + 1] == "0.001"
    assert command[command.index("--momentum") + 1] == "0.9"
    assert command[command.index("--warmup-bias-lr") + 1] == "0.0"


@pytest.mark.parametrize(
    "arguments",
    (
        ["--dfl", "2.0"],
        ["--cls-pw", "0.25"],
        ["--spectral-stem"],
        ["--scale", "0.3"],
        ["--degrees", "5"],
        ["--box-iou-loss", "eiou"],
        ["--optimizer-recipe", "adamw_lr001"],
        ["--reg-max", "16"],
        ["--data", "hsi16_phase"],
        ["--object-crops"],
        ["--tile-inference"],
    ),
)
def test_make_kernel_rejects_yolo_only_variants_for_rtdetr(
    tmp_path: Path, monkeypatch, arguments: list[str]
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
            "smoke",
            "--architecture",
            "rtdetr",
            "--epochs",
            "1",
            *arguments,
        ],
    )

    with pytest.raises(SystemExit):
        module.main()


def test_make_kernel_renders_rtdetr_num_denoising_variant(
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
            "--architecture",
            "rtdetr",
            "--epochs",
            "30",
            "--rtdetr-num-denoising",
            "200",
        ],
    )

    module.main()

    folder = tmp_path / "kernel_ablation_rtdetr_nd200"
    rendered = (folder / "run_hsi_yolo26.py").read_text(encoding="utf-8")
    metadata = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert '"RUN_NAME": "kaggle_ablation_rtdetr-l_nd200_e30"' in rendered
    assert '"RTDETR_NUM_DENOISING": 200' in rendered
    assert '"DEGREES": 0.0' in rendered
    assert metadata["id"] == "zephyrpong/hsi-rtdetr-l-nd200-ablation"
    assert metadata["is_private"] is True


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
