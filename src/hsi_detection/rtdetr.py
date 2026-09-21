from __future__ import annotations

from typing import Any

import torch
from torch import nn
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.torch_utils import unwrap_model


RTDETR_HSI_METADATA_KEY = "rtdetr_hsi_input_transfer"
RTDETR_HSI_METADATA_VERSION = 1
RTDETR_DENOISING_METADATA_KEY = "rtdetr_num_denoising"


def rtdetr_input_conv(model: nn.Module) -> nn.Conv2d:
    """Return the HGStem input convolution from an RT-DETR model or wrapper."""
    current: Any = unwrap_model(model)
    for _ in range(2):
        layers = getattr(current, "model", None)
        if isinstance(layers, (nn.Sequential, nn.ModuleList)) and len(layers):
            stem1 = getattr(layers[0], "stem1", None)
            convolution = getattr(stem1, "conv", None)
            if isinstance(convolution, nn.Conv2d):
                return convolution
        if isinstance(layers, nn.Module):
            current = layers
            continue
        break
    raise RuntimeError("Could not locate RT-DETR HGStem stem1 input convolution")


def transfer_rtdetr_input_weights(
    target: nn.Module,
    source: nn.Module,
    *,
    extra_channel_init: str = "random",
) -> dict[str, Any]:
    """Copy the pretrained RGB stem into a wider RT-DETR input convolution.

    Ultralytics' generic partial-channel transfer currently recognizes the YOLO
    key ``model.0.conv.weight`` only.  RT-DETR uses
    ``model.0.stem1.conv.weight``, so without this explicit transfer all 16 input
    channels would start randomly even though the remainder of the pretrained
    backbone transfers normally.
    """
    if extra_channel_init not in {"random", "zero"}:
        raise ValueError("extra_channel_init must be 'random' or 'zero'")

    target_conv = rtdetr_input_conv(target)
    source_conv = rtdetr_input_conv(source)
    target_shape = tuple(target_conv.weight.shape)
    source_shape = tuple(source_conv.weight.shape)
    if target_shape[0] != source_shape[0] or target_shape[2:] != source_shape[2:]:
        raise RuntimeError(
            f"RT-DETR input convolution geometry mismatch: source={source_shape}, target={target_shape}"
        )
    if source_conv.in_channels != 3:
        raise RuntimeError(
            f"Expected a three-channel pretrained RT-DETR source, got {source_conv.in_channels}"
        )
    if target_conv.in_channels <= source_conv.in_channels:
        raise RuntimeError(
            "HSI RT-DETR transfer requires a target wider than the three-channel source"
        )

    with torch.no_grad():
        target_conv.weight[:, : source_conv.in_channels].copy_(
            source_conv.weight.to(
                device=target_conv.weight.device, dtype=target_conv.weight.dtype
            )
        )
        if extra_channel_init == "zero":
            target_conv.weight[:, source_conv.in_channels :].zero_()

    metadata: dict[str, Any] = {
        "version": RTDETR_HSI_METADATA_VERSION,
        "source_channels": source_conv.in_channels,
        "target_channels": target_conv.in_channels,
        "extra_channel_init": extra_channel_init,
        "stem": "HGStem.stem1.conv",
    }
    yaml = getattr(unwrap_model(target), "yaml", None)
    if isinstance(yaml, dict):
        yaml[RTDETR_HSI_METADATA_KEY] = metadata
    LOGGER.info(
        "Transferred RT-DETR RGB stem into %d-channel HGStem; extra channels use %s initialization",
        target_conv.in_channels,
        extra_channel_init,
    )
    return metadata


def configure_rtdetr_num_denoising(model: nn.Module, num_denoising: int) -> int:
    """Set the RT-DETR decoder's training-only denoising query count.

    ``num_denoising`` is a scalar decoder setting rather than a tensor in the
    checkpoint state dict.  Set it on the newly constructed target model after
    weight loading, and mirror it into the model YAML so remote audit artifacts
    retain the exact experiment value.
    """
    if isinstance(num_denoising, bool) or not isinstance(num_denoising, int) or num_denoising <= 0:
        raise ValueError("num_denoising must be a positive integer")

    current: Any = unwrap_model(model)
    decoder: nn.Module | None = None
    for _ in range(2):
        layers = getattr(current, "model", None)
        if isinstance(layers, (nn.Sequential, nn.ModuleList)) and len(layers):
            candidate = layers[-1]
            if hasattr(candidate, "num_denoising"):
                decoder = candidate
                break
        if isinstance(layers, nn.Module):
            current = layers
            continue
        break
    if decoder is None:
        raise RuntimeError("Could not locate RT-DETR decoder num_denoising setting")

    decoder.num_denoising = num_denoising
    yaml = getattr(unwrap_model(model), "yaml", None)
    if isinstance(yaml, dict):
        yaml[RTDETR_DENOISING_METADATA_KEY] = num_denoising
    LOGGER.info("Configured RT-DETR with %d denoising queries", num_denoising)
    return num_denoising


class HSIRTDETRTrainer(RTDETRTrainer):
    """RT-DETR trainer that preserves RGB stem transfer for HSI16 input."""

    extra_channel_init = "random"
    num_denoising = 100

    def get_model(
        self,
        cfg: dict | None = None,
        weights: nn.Module | None = None,
        verbose: bool = True,
    ) -> RTDETRDetectionModel:
        channels = int(self.data.get("channels", 3))
        model = self.set_model_names_for_load(
            RTDETRDetectionModel(
                cfg,
                nc=self.data["nc"],
                ch=channels,
                verbose=verbose and RANK == -1,
            )
        )
        if weights is not None:
            model.load(weights)
            if channels > 3:
                transfer_rtdetr_input_weights(
                    model,
                    weights,
                    extra_channel_init=self.extra_channel_init,
                )
        elif channels > 3:
            LOGGER.warning(
                "Building %d-channel RT-DETR without pretrained weights; all input channels are random",
                channels,
            )
        configure_rtdetr_num_denoising(model, self.num_denoising)
        return model
