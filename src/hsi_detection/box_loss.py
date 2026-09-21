"""Auditable YOLO box-IoU loss variants used by this project."""

from __future__ import annotations

from typing import Any

import torch
from ultralytics.models.yolo.detect import DetectionTrainer


BOX_IOU_LOSS_METADATA_KEY = "box_iou_loss"
SUPPORTED_BOX_IOU_LOSSES = ("ciou", "eiou")
_ORIGINAL_ULTRALYTICS_BBOX_IOU = None


def bbox_eiou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    GIoU: bool = False,
    DIoU: bool = False,
    CIoU: bool = False,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Return Extended IoU with the Ultralytics ``bbox_iou`` call signature.

    YOLO's :class:`BboxLoss` calls ``bbox_iou(..., xywh=False, CIoU=True)``.
    Keeping that signature lets the project replace only the localization term;
    TaskAlignedAssigner, DFL/L1 supervision, model outputs, and inference remain
    unchanged. Other IoU flags are rejected so an accidental wider monkey-patch
    cannot silently change unrelated behavior.
    """
    if GIoU or DIoU or not CIoU:
        raise ValueError(
            "bbox_eiou is an EIoU replacement for the YOLO CIoU loss call only"
        )
    if box1.shape[-1] != 4 or box2.shape[-1] != 4:
        raise ValueError("EIoU boxes must have a final dimension of four")

    if xywh:
        x1, y1, w1, h1 = box1.chunk(4, dim=-1)
        x2, y2, w2, h2 = box2.chunk(4, dim=-1)
        w1 = w1.clamp_min(0)
        h1 = h1.clamp_min(0)
        w2 = w2.clamp_min(0)
        h2 = h2.clamp_min(0)
        b1_x1, b1_x2 = x1 - w1 / 2, x1 + w1 / 2
        b1_y1, b1_y2 = y1 - h1 / 2, y1 + h1 / 2
        b2_x1, b2_x2 = x2 - w2 / 2, x2 + w2 / 2
        b2_y1, b2_y2 = y2 - h2 / 2, y2 + h2 / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, dim=-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, dim=-1)
        w1 = (b1_x2 - b1_x1).clamp_min(0)
        h1 = (b1_y2 - b1_y1).clamp_min(0)
        w2 = (b2_x2 - b2_x1).clamp_min(0)
        h2 = (b2_y2 - b2_y1).clamp_min(0)

    intersection = (
        (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_min(0)
        * (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_min(0)
    )
    union = w1 * h1 + w2 * h2 - intersection + eps
    iou = intersection / union

    enclosing_width = (
        b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ).clamp_min(0)
    enclosing_height = (
        b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    ).clamp_min(0)
    enclosing_diagonal_squared = (
        enclosing_width.square() + enclosing_height.square() + eps
    )
    center_distance_squared = (
        (b2_x1 + b2_x2 - b1_x1 - b1_x2).square()
        + (b2_y1 + b2_y2 - b1_y1 - b1_y2).square()
    ) / 4
    width_distance_squared = (w2 - w1).square()
    height_distance_squared = (h2 - h1).square()

    return iou - (
        center_distance_squared / enclosing_diagonal_squared
        + width_distance_squared / (enclosing_width.square() + eps)
        + height_distance_squared / (enclosing_height.square() + eps)
    )


def install_box_iou_loss(name: str) -> None:
    """Install one YOLO localization loss without touching the assigner IoU."""
    if name not in SUPPORTED_BOX_IOU_LOSSES:
        raise ValueError(
            f"Unsupported box IoU loss {name!r}; expected one of {SUPPORTED_BOX_IOU_LOSSES}"
        )

    import ultralytics.utils.loss as ultralytics_loss

    global _ORIGINAL_ULTRALYTICS_BBOX_IOU
    if _ORIGINAL_ULTRALYTICS_BBOX_IOU is None:
        _ORIGINAL_ULTRALYTICS_BBOX_IOU = ultralytics_loss.bbox_iou
    ultralytics_loss.bbox_iou = (
        _ORIGINAL_ULTRALYTICS_BBOX_IOU if name == "ciou" else bbox_eiou
    )


def checkpoint_box_iou_loss(model: Any) -> str:
    """Read and validate the loss recipe stored in a loaded model checkpoint."""
    yaml = getattr(model, "yaml", {})
    value = yaml.get(BOX_IOU_LOSS_METADATA_KEY, "ciou") if isinstance(yaml, dict) else "ciou"
    if value not in SUPPORTED_BOX_IOU_LOSSES:
        raise RuntimeError(f"Checkpoint declares unsupported box IoU loss {value!r}")
    return value


def record_box_iou_loss(model: Any, name: str) -> None:
    """Persist the selected recipe in the checkpoint's model YAML metadata."""
    if name not in SUPPORTED_BOX_IOU_LOSSES:
        raise ValueError(f"Unsupported box IoU loss {name!r}")
    yaml = getattr(model, "yaml", None)
    if not isinstance(yaml, dict):
        raise RuntimeError("Could not record box IoU loss in model YAML metadata")
    yaml[BOX_IOU_LOSS_METADATA_KEY] = name


class EIoUDetectionTrainer(DetectionTrainer):
    """Detection trainer that installs EIoU in parent and DDP worker processes."""

    def __init__(self, *args, **kwargs) -> None:
        install_box_iou_loss("eiou")
        super().__init__(*args, **kwargs)
        self.box_iou_loss = "eiou"
