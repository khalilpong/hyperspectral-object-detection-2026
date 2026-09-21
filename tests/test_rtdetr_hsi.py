from __future__ import annotations

import torch
import pytest
from torch import nn

from hsi_detection.rtdetr import (
    RTDETR_DENOISING_METADATA_KEY,
    RTDETR_HSI_METADATA_KEY,
    configure_rtdetr_num_denoising,
    rtdetr_input_conv,
    transfer_rtdetr_input_weights,
)


class _Stem(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.stem1 = nn.Module()
        self.stem1.conv = nn.Conv2d(channels, 4, 3, bias=False)


class _Model(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.model = nn.ModuleList([_Stem(channels)])
        self.yaml: dict[str, object] = {"channels": channels}


class _Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_denoising = 100


class _DecoderModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.ModuleList([nn.Identity(), _Decoder()])
        self.yaml: dict[str, object] = {}


def test_rtdetr_input_transfer_copies_rgb_and_keeps_random_extras() -> None:
    torch.manual_seed(1)
    source = _Model(3)
    target = _Model(16)
    extra_before = target.model[0].stem1.conv.weight[:, 3:].detach().clone()

    metadata = transfer_rtdetr_input_weights(target, source, extra_channel_init="random")

    assert torch.equal(
        rtdetr_input_conv(target).weight[:, :3], rtdetr_input_conv(source).weight
    )
    assert torch.equal(rtdetr_input_conv(target).weight[:, 3:], extra_before)
    assert metadata["target_channels"] == 16
    assert target.yaml[RTDETR_HSI_METADATA_KEY] == metadata


def test_rtdetr_input_transfer_can_zero_extra_channels() -> None:
    source = _Model(3)
    target = _Model(16)

    transfer_rtdetr_input_weights(target, source, extra_channel_init="zero")

    assert torch.count_nonzero(rtdetr_input_conv(target).weight[:, 3:]) == 0


def test_configure_rtdetr_num_denoising_updates_decoder_and_metadata() -> None:
    model = _DecoderModel()

    assert configure_rtdetr_num_denoising(model, 200) == 200

    assert model.model[-1].num_denoising == 200
    assert model.yaml[RTDETR_DENOISING_METADATA_KEY] == 200


def test_configure_rtdetr_num_denoising_rejects_invalid_value() -> None:
    for invalid in (0, -1, True):
        with pytest.raises(ValueError):
            configure_rtdetr_num_denoising(_DecoderModel(), invalid)
