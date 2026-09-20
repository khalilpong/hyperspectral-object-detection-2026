from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from ultralytics import YOLO

from hsi_detection.spectral_stem import (
    DEFAULT_HSI_BAND_ORDER,
    SPECTRAL_STEM_METADATA_KEY,
    SpectralDetectionTrainer,
    attach_spectral_stem,
    has_spectral_stem,
    make_spectral_projection,
    resolve_identity_input_indices,
    validate_spectral_dataset_contract,
    validate_spectral_stem,
)


class _FirstBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1, bias=False)
        self.i = 0
        self.f = -1
        self.type = "FirstBlock"
        self.np = sum(parameter.numel() for parameter in self.parameters())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(inputs)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(_FirstBlock())
        self.yaml = {"channels": 3, "nc": 1}


def test_current_band_order_maps_physical_rgb_to_first_three_channels() -> None:
    assert resolve_identity_input_indices(DEFAULT_HSI_BAND_ORDER) == (0, 1, 2)


def test_dataset_contract_requires_matching_band_metadata() -> None:
    assert validate_spectral_dataset_contract(
        {"channels": 16, "hsi_band_order": list(DEFAULT_HSI_BAND_ORDER)}
    ) == (16, DEFAULT_HSI_BAND_ORDER, (0, 1, 2))

    with pytest.raises(ValueError, match="16 channels"):
        validate_spectral_dataset_contract(
            {"channels": 16, "hsi_band_order": list(DEFAULT_HSI_BAND_ORDER[:-1])}
        )


def test_projection_is_identity_initialized_and_extra_bands_receive_gradients() -> None:
    projection = make_spectral_projection(16, (0, 1, 2))
    expected = torch.zeros_like(projection.weight)
    expected[0, 0, 0, 0] = 1
    expected[1, 1, 0, 0] = 1
    expected[2, 2, 0, 0] = 1
    assert torch.equal(projection.weight, expected)

    inputs = torch.randn(2, 16, 4, 5)
    outputs = projection(inputs)
    assert torch.equal(outputs, inputs[:, :3])
    outputs.square().mean().backward()
    assert projection.weight.grad is not None
    assert projection.weight.grad[:, 3:].abs().sum() > 0


def test_attach_preserves_detector_and_checkpoint_uses_only_standard_modules() -> None:
    model = _Model()
    detector = model.model[0]
    detector_state = deepcopy(detector.state_dict())
    inputs = torch.randn(2, 16, 8, 8)
    expected = detector(inputs[:, :3])

    returned = attach_spectral_stem(model)

    assert returned is model
    assert has_spectral_stem(model)
    wrapper = model.model[0]
    assert type(wrapper) is nn.Sequential
    assert type(wrapper.projection) is nn.Conv2d
    assert wrapper.detector is detector
    assert torch.equal(wrapper(inputs), expected)
    assert detector_state.keys() == detector.state_dict().keys()
    for name, value in detector_state.items():
        assert torch.equal(value, detector.state_dict()[name])
    assert model.yaml["channels"] == 16
    assert model.yaml[SPECTRAL_STEM_METADATA_KEY]["identity_input_indices"] == [0, 1, 2]
    assert validate_spectral_stem(model)["band_order"] == list(DEFAULT_HSI_BAND_ORDER)


def test_attach_is_idempotent_but_rejects_a_different_dataset_order() -> None:
    model = _Model()
    attach_spectral_stem(model)
    wrapper = model.model[0]

    assert attach_spectral_stem(model) is model
    assert model.model[0] is wrapper
    with pytest.raises(RuntimeError, match="band order"):
        attach_spectral_stem(model, band_order=tuple(reversed(DEFAULT_HSI_BAND_ORDER)))


def test_trainer_reuses_a_spectral_checkpoint_without_rebuilding() -> None:
    yolo = YOLO("yolo26n.yaml")
    attach_spectral_stem(yolo.model)
    model = yolo.model
    original_state = deepcopy(model.state_dict())

    trainer = object.__new__(SpectralDetectionTrainer)
    trainer.data = {
        "channels": 16,
        "hsi_band_order": list(DEFAULT_HSI_BAND_ORDER),
        "nc": int(model.yaml["nc"]),
        "names": model.names,
    }
    trainer.args = SimpleNamespace(cls_remap=True)

    returned = trainer.get_model(cfg=deepcopy(model.yaml), weights=model, verbose=False)

    assert returned is model
    assert validate_spectral_stem(returned)["identity_input_indices"] == [0, 1, 2]
    assert original_state.keys() == returned.state_dict().keys()
    for name, value in original_state.items():
        assert torch.equal(value, returned.state_dict()[name])


def test_safe_checkpoint_round_trip_uses_no_project_defined_module(tmp_path: Path) -> None:
    yolo = YOLO("yolo26n.yaml")
    attach_spectral_stem(yolo.model)
    checkpoint = tmp_path / "spectral-standard-modules.pt"
    yolo.save(checkpoint)

    child_code = textwrap.dedent(
        """
        import builtins
        import sys
        import torch

        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.startswith("hsi_detection"):
                raise RuntimeError("checkpoint attempted to import a project-defined class")
            return original_import(name, *args, **kwargs)
        builtins.__import__ = guarded_import

        from ultralytics import YOLO

        yolo = YOLO(sys.argv[1])
        assert yolo.model.yaml["channels"] == 16
        assert type(yolo.model.model[0]) is torch.nn.Sequential
        assert type(yolo.model.model[0].projection) is torch.nn.Conv2d
        yolo.model.eval()
        with torch.no_grad():
            yolo.model(torch.zeros(1, 16, 64, 64))
        print("fresh-process-load=ok")
        """
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["ULTRALYTICS_SAFE_LOAD"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", child_code, str(checkpoint)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "fresh-process-load=ok" in completed.stdout
