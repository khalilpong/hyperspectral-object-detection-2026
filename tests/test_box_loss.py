from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from hsi_detection.box_loss import (
    bbox_eiou,
    checkpoint_box_iou_loss,
    install_box_iou_loss,
    record_box_iou_loss,
)


def test_eiou_identical_boxes_are_one_and_mismatch_has_finite_gradient() -> None:
    identical = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    score = bbox_eiou(identical, identical, xywh=False, CIoU=True)
    assert torch.allclose(score, torch.ones_like(score), atol=1e-6)

    prediction = torch.tensor(
        [[0.1, 0.2, 2.3, 2.5]], requires_grad=True
    )
    target = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    loss = (1.0 - bbox_eiou(prediction, target, xywh=False, CIoU=True)).sum()
    loss.backward()

    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0


def test_eiou_is_finite_for_degenerate_boxes() -> None:
    prediction = torch.zeros((1, 4), requires_grad=True)
    target = torch.zeros((1, 4))

    score = bbox_eiou(prediction, target, xywh=False, CIoU=True)
    score.sum().backward()

    assert torch.isfinite(score).all()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_eiou_rejects_non_loss_call_signatures() -> None:
    box = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    with pytest.raises(ValueError, match="replacement"):
        bbox_eiou(box, box, xywh=False)
    with pytest.raises(ValueError, match="replacement"):
        bbox_eiou(box, box, xywh=False, GIoU=True, CIoU=True)


def test_installation_changes_only_yolo_loss_module_not_assigner() -> None:
    import ultralytics.utils.loss as loss_module
    import ultralytics.utils.tal as tal_module

    original_assigner_iou = tal_module.bbox_iou
    try:
        install_box_iou_loss("eiou")
        assert loss_module.bbox_iou is bbox_eiou
        assert tal_module.bbox_iou is original_assigner_iou
    finally:
        install_box_iou_loss("ciou")

    assert loss_module.bbox_iou is not bbox_eiou
    assert tal_module.bbox_iou is original_assigner_iou


def test_checkpoint_metadata_round_trip_defaults_to_ciou() -> None:
    model = SimpleNamespace(yaml={})
    assert checkpoint_box_iou_loss(model) == "ciou"

    record_box_iou_loss(model, "eiou")
    assert checkpoint_box_iou_loss(model) == "eiou"

    model.yaml["box_iou_loss"] = "unknown"
    with pytest.raises(RuntimeError, match="unsupported"):
        checkpoint_box_iou_loss(model)
