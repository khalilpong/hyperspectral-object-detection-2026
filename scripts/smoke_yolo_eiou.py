"""CPU smoke test for YOLO EIoU forward/loss/backward on a tiny HSI batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel

from hsi_detection.box_loss import install_box_iou_loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo26m.yaml")
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--classes", type=int, default=18)
    parser.add_argument("--imgsz", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.channels <= 3 or args.classes <= 0 or args.imgsz <= 0:
        parser.error("channels must exceed 3; classes and imgsz must be positive")

    torch.manual_seed(2026)
    install_box_iou_loss("eiou")
    model = DetectionModel(
        args.model,
        ch=args.channels,
        nc=args.classes,
        verbose=False,
    )
    # DetectionTrainer normally attaches this namespace before the first loss
    # call. The standalone smoke constructs the model directly, so reproduce
    # that part of the real training contract explicitly.
    model.args = get_cfg()
    model.train()
    batch = {
        "img": torch.rand(1, args.channels, args.imgsz, args.imgsz),
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[2.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
    }
    loss, parts = model.loss(batch)
    total_loss = loss.sum()
    total_loss.backward()
    stem_gradient = model.model[0].conv.weight.grad
    if isinstance(parts, dict):
        rendered_parts = {key: float(value) for key, value in parts.items()}
        finite_parts = all(bool(torch.isfinite(value).all()) for value in parts.values())
    else:
        rendered_parts = [float(value) for value in parts]
        finite_parts = bool(torch.isfinite(parts).all())
    report = {
        "model": args.model,
        "channels": args.channels,
        "classes": args.classes,
        "imgsz": args.imgsz,
        "box_iou_loss": "eiou",
        "loss": float(total_loss.detach()),
        "loss_vector": [float(value) for value in loss.detach()],
        "loss_parts": rendered_parts,
        "loss_finite": bool(torch.isfinite(loss).all()),
        "loss_parts_finite": finite_parts,
        "stem_shape": list(model.model[0].conv.weight.shape),
        "stem_gradient_finite": bool(
            stem_gradient is not None and torch.isfinite(stem_gradient).all()
        ),
        "stem_gradient_abs_sum": (
            float(stem_gradient.abs().sum()) if stem_gradient is not None else 0.0
        ),
    }
    if not report["loss_finite"] or not report["loss_parts_finite"]:
        raise RuntimeError("YOLO EIoU smoke produced a non-finite loss")
    if not report["stem_gradient_finite"] or report["stem_gradient_abs_sum"] <= 0.0:
        raise RuntimeError("YOLO EIoU smoke produced no finite nonzero stem gradient")
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
