from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import torch
from torch import nn
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK


SPECTRAL_STEM_METADATA_KEY = "spectral_stem"
SPECTRAL_STEM_VERSION = 1
DEFAULT_PHYSICAL_BANDS = (5, 8, 13)
DEFAULT_HSI_BAND_ORDER = (5, 8, 13, 0, 1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15)


def _integer_tuple(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    try:
        result = tuple(int(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a sequence of integer indices") from error
    if not result:
        raise ValueError(f"{name} must not be empty")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def resolve_identity_input_indices(
    band_order: Sequence[int],
    physical_bands: Sequence[int] = DEFAULT_PHYSICAL_BANDS,
) -> tuple[int, int, int]:
    """Resolve physical pseudo-RGB bands to their channel indices in an HSI tensor."""
    ordered_bands = _integer_tuple(band_order, name="band_order")
    selected_bands = _integer_tuple(physical_bands, name="physical_bands")
    if len(selected_bands) != 3:
        raise ValueError("physical_bands must contain exactly three bands")
    missing = [band for band in selected_bands if band not in ordered_bands]
    if missing:
        raise ValueError(f"physical_bands {missing} are absent from band_order {ordered_bands}")
    return tuple(ordered_bands.index(band) for band in selected_bands)  # type: ignore[return-value]


def validate_spectral_dataset_contract(
    data: Mapping[str, Any],
) -> tuple[int, tuple[int, ...], tuple[int, int, int]]:
    """Validate and return the external channel contract used by the spectral stem."""
    try:
        channels = int(data["channels"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("SpectralStem requires an integer dataset 'channels' field") from error
    if channels <= 3:
        raise ValueError(f"SpectralStem requires more than three input channels, got {channels}")
    try:
        band_order = _integer_tuple(data["hsi_band_order"], name="hsi_band_order")
    except KeyError as error:
        raise ValueError("SpectralStem requires dataset metadata 'hsi_band_order'") from error
    if len(band_order) != channels:
        raise ValueError(
            f"hsi_band_order has {len(band_order)} entries but the dataset declares {channels} channels"
        )
    identity_indices = resolve_identity_input_indices(band_order)
    return channels, band_order, identity_indices


def make_spectral_projection(
    in_channels: int,
    identity_input_indices: Sequence[int],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> nn.Conv2d:
    """Create a trainable 1x1 projection initialized as a three-band identity selector."""
    in_channels = int(in_channels)
    if in_channels <= 3:
        raise ValueError(f"in_channels must be greater than three, got {in_channels}")
    indices = _integer_tuple(identity_input_indices, name="identity_input_indices")
    if len(indices) != 3:
        raise ValueError("identity_input_indices must contain exactly three channel indices")
    if min(indices) < 0 or max(indices) >= in_channels:
        raise ValueError(
            f"identity_input_indices {indices} are invalid for {in_channels} input channels"
        )

    projection = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
    projection = projection.to(device=device, dtype=dtype)
    with torch.no_grad():
        projection.weight.zero_()
        for output_channel, input_channel in enumerate(indices):
            projection.weight[output_channel, input_channel, 0, 0] = 1.0
    return projection


def _stem_parts(model: nn.Module) -> tuple[nn.Sequential, nn.Conv2d, nn.Module] | None:
    layers = getattr(model, "model", None)
    if not isinstance(layers, (nn.Sequential, nn.ModuleList)) or not len(layers):
        return None
    wrapper = layers[0]
    if not isinstance(wrapper, nn.Sequential):
        return None
    if tuple(wrapper._modules) != ("projection", "detector"):
        return None
    projection = wrapper._modules["projection"]
    detector = wrapper._modules["detector"]
    if not isinstance(projection, nn.Conv2d):
        return None
    return wrapper, projection, detector


def has_spectral_stem(model: nn.Module) -> bool:
    """Return whether a model advertises or structurally contains the standard-module stem."""
    yaml = getattr(model, "yaml", {})
    return _stem_parts(model) is not None or (
        isinstance(yaml, Mapping) and SPECTRAL_STEM_METADATA_KEY in yaml
    )


def validate_spectral_stem(
    model: nn.Module,
    *,
    expected_band_order: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Validate a persisted stem and return normalized metadata."""
    yaml = getattr(model, "yaml", None)
    if not isinstance(yaml, Mapping):
        raise RuntimeError("SpectralStem model is missing its YAML metadata")
    raw_metadata = yaml.get(SPECTRAL_STEM_METADATA_KEY)
    if not isinstance(raw_metadata, Mapping):
        raise RuntimeError("SpectralStem model is missing spectral_stem metadata")
    parts = _stem_parts(model)
    if parts is None:
        raise RuntimeError("SpectralStem metadata exists but model.model[0] is not a valid stem wrapper")
    _, projection, detector = parts
    detector_conv = getattr(detector, "conv", None)
    if not isinstance(detector_conv, nn.Conv2d) or detector_conv.in_channels != 3:
        raise RuntimeError("SpectralStem detector block must retain a three-channel input convolution")
    if projection.kernel_size != (1, 1) or projection.out_channels != 3 or projection.bias is not None:
        raise RuntimeError("SpectralStem projection must be a bias-free 1x1 convolution with three outputs")

    try:
        metadata = {
            "version": int(raw_metadata["version"]),
            "in_channels": int(raw_metadata["in_channels"]),
            "out_channels": int(raw_metadata["out_channels"]),
            "band_order": [int(value) for value in raw_metadata["band_order"]],
            "identity_physical_bands": [
                int(value) for value in raw_metadata["identity_physical_bands"]
            ],
            "identity_input_indices": [
                int(value) for value in raw_metadata["identity_input_indices"]
            ],
        }
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("SpectralStem metadata is incomplete or invalid") from error

    if metadata["version"] != SPECTRAL_STEM_VERSION:
        raise RuntimeError(f"Unsupported SpectralStem metadata version {metadata['version']}")
    if metadata["out_channels"] != 3 or projection.out_channels != metadata["out_channels"]:
        raise RuntimeError("SpectralStem metadata disagrees with the projection output channels")
    if projection.in_channels != metadata["in_channels"]:
        raise RuntimeError("SpectralStem metadata disagrees with the projection input channels")
    if int(yaml.get("channels", -1)) != metadata["in_channels"]:
        raise RuntimeError("Model YAML channels does not describe the SpectralStem external input")
    resolved = resolve_identity_input_indices(
        metadata["band_order"], metadata["identity_physical_bands"]
    )
    if tuple(metadata["identity_input_indices"]) != resolved:
        raise RuntimeError("SpectralStem identity indices do not match its physical-band metadata")
    if expected_band_order is not None and tuple(metadata["band_order"]) != tuple(
        int(value) for value in expected_band_order
    ):
        raise RuntimeError("Checkpoint SpectralStem band order does not match the dataset")
    return metadata


def attach_spectral_stem(
    model: nn.Module,
    *,
    band_order: Sequence[int] = DEFAULT_HSI_BAND_ORDER,
    physical_bands: Sequence[int] = DEFAULT_PHYSICAL_BANDS,
) -> nn.Module:
    """Attach a standard-module 16-to-3 stem without changing YOLO layer indices."""
    ordered_bands = _integer_tuple(band_order, name="band_order")
    selected_bands = _integer_tuple(physical_bands, name="physical_bands")
    identity_indices = resolve_identity_input_indices(ordered_bands, selected_bands)

    if has_spectral_stem(model):
        validate_spectral_stem(model, expected_band_order=ordered_bands)
        return model

    layers = getattr(model, "model", None)
    yaml = getattr(model, "yaml", None)
    if not isinstance(layers, (nn.Sequential, nn.ModuleList)) or not len(layers):
        raise RuntimeError("Could not locate model.model[0] for SpectralStem attachment")
    if not isinstance(yaml, dict):
        raise RuntimeError("Could not locate mutable model YAML for SpectralStem metadata")
    detector = layers[0]
    detector_conv = getattr(detector, "conv", None)
    if not isinstance(detector_conv, nn.Conv2d) or detector_conv.in_channels != 3:
        raise RuntimeError("SpectralStem can only wrap an unfused YOLO first block with three input channels")
    for attribute in ("i", "f", "type", "np"):
        if not hasattr(detector, attribute):
            raise RuntimeError(f"YOLO first block is missing graph attribute '{attribute}'")

    projection = make_spectral_projection(
        len(ordered_bands),
        identity_indices,
        device=detector_conv.weight.device,
        dtype=detector_conv.weight.dtype,
    )
    wrapper = nn.Sequential(
        OrderedDict(
            (
                ("projection", projection),
                ("detector", detector),
            )
        )
    )
    wrapper.i = detector.i
    wrapper.f = detector.f
    wrapper.type = f"SpectralStem[{len(ordered_bands)}->3]+{detector.type}"
    wrapper.np = sum(parameter.numel() for parameter in wrapper.parameters())
    wrapper.train(detector.training)
    layers[0] = wrapper

    yaml["channels"] = len(ordered_bands)
    yaml[SPECTRAL_STEM_METADATA_KEY] = {
        "version": SPECTRAL_STEM_VERSION,
        "in_channels": len(ordered_bands),
        "out_channels": 3,
        "band_order": list(ordered_bands),
        "identity_physical_bands": list(selected_bands),
        "identity_input_indices": list(identity_indices),
    }
    validate_spectral_stem(model, expected_band_order=ordered_bands)
    LOGGER.info(
        "Attached trainable SpectralStem %d->3 initialized from physical bands %s at input indices %s",
        len(ordered_bands),
        selected_bands,
        identity_indices,
    )
    return model


class SpectralDetectionTrainer(DetectionTrainer):
    """Detection trainer that preserves the standard-module SpectralStem across rebuilds."""

    def get_model(self, cfg: str | dict | None = None, weights=None, verbose: bool = True):
        channels, band_order, _ = validate_spectral_dataset_contract(self.data)

        if isinstance(weights, nn.Module) and has_spectral_stem(weights):
            metadata = validate_spectral_stem(weights, expected_band_order=band_order)
            if metadata["in_channels"] != channels:
                raise RuntimeError("Checkpoint SpectralStem input channels do not match the dataset")
            if int(weights.yaml.get("nc", -1)) != int(self.data["nc"]):
                raise RuntimeError("Checkpoint SpectralStem class count does not match the dataset")
            return self.set_model_names_for_load(weights.float())

        if isinstance(weights, nn.Module):
            source_channels = int(getattr(weights, "yaml", {}).get("channels", 3))
            if source_channels != 3:
                raise RuntimeError(
                    "A new SpectralStem run must start from a three-channel pretrained checkpoint"
                )

        clean_cfg = deepcopy(cfg)
        if isinstance(clean_cfg, dict):
            clean_cfg.pop(SPECTRAL_STEM_METADATA_KEY, None)
        model = self.set_model_names_for_load(
            DetectionModel(
                clean_cfg,
                nc=self.data["nc"],
                ch=3,
                verbose=verbose and RANK == -1,
            )
        )
        if weights is not None:
            model.load(weights)
        attach_spectral_stem(model, band_order=band_order)
        return model
