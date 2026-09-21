from __future__ import annotations

import torch
from torch import nn

from hsi_detection.rtdetr import (
    RTDETR_HSI_METADATA_KEY,
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
