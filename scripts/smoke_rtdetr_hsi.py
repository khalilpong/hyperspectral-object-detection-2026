"""CPU smoke test for an HSI16 RT-DETR forward/loss/backward vertical slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from ultralytics.nn.tasks import RTDETRDetectionModel

from hsi_detection.rtdetr import rtdetr_input_conv, transfer_rtdetr_input_weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="rtdetr-l.yaml")
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--classes", type=int, default=18)
    parser.add_argument("--imgsz", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.channels <= 3 or args.classes <= 0 or args.imgsz <= 0:
        parser.error("channels must exceed 3; classes and imgsz must be positive")

    torch.manual_seed(2026)
    source = RTDETRDetectionModel(args.model, ch=3, nc=args.classes, verbose=False)
    target = RTDETRDetectionModel(
        args.model, ch=args.channels, nc=args.classes, verbose=False
    )
    metadata = transfer_rtdetr_input_weights(target, source)
    source_weight = rtdetr_input_conv(source).weight.detach()
    target_weight = rtdetr_input_conv(target).weight.detach()
    rgb_transfer_exact = bool(torch.equal(source_weight, target_weight[:, :3]))

    target.nc = args.classes
    target.train()
    image = torch.rand(1, args.channels, args.imgsz, args.imgsz)
    batch = {
        "img": image,
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[2.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
    }
    loss, parts = target.loss(batch)
    loss.backward()
    gradient = rtdetr_input_conv(target).weight.grad
    report = {
        "model": args.model,
        "channels": args.channels,
        "classes": args.classes,
        "imgsz": args.imgsz,
        "metadata": metadata,
        "rgb_transfer_exact": rgb_transfer_exact,
        "loss": float(loss.detach()),
        "loss_parts": {key: float(value) for key, value in parts.items()},
        "stem_shape": list(rtdetr_input_conv(target).weight.shape),
        "stem_gradient_finite": bool(gradient is not None and torch.isfinite(gradient).all()),
        "stem_gradient_abs_sum": float(gradient.abs().sum()) if gradient is not None else 0.0,
    }
    if not report["rgb_transfer_exact"]:
        raise RuntimeError("RT-DETR RGB stem transfer did not preserve source weights")
    if not report["stem_gradient_finite"] or report["stem_gradient_abs_sum"] <= 0.0:
        raise RuntimeError("RT-DETR HSI stem did not receive a finite nonzero gradient")
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
