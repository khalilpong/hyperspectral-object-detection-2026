"""CPU smoke for a checkpoint-native YOLO architecture widened to HSI16."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _branch_output_channels(head: torch.nn.Module, name: str) -> list[int] | None:
    branch = getattr(head, name, None)
    if branch is None:
        return None
    return [int(block[-1].out_channels) for block in branch]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expected-reg-max", type=int, required=True)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--classes", type=int, default=18)
    parser.add_argument("--imgsz", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.expected_reg_max <= 1:
        parser.error("--expected-reg-max must exceed 1 for a true DFL smoke")
    if args.channels <= 3 or args.classes <= 0 or args.imgsz <= 0:
        parser.error("channels must exceed 3; classes and imgsz must be positive")
    if not args.weights.is_file():
        parser.error("--weights must be an existing checkpoint")

    torch.manual_seed(2026)
    source = YOLO(str(args.weights.resolve())).model.cpu()
    source_yaml = copy.deepcopy(source.yaml)
    target = DetectionModel(
        source_yaml,
        ch=args.channels,
        nc=args.classes,
        verbose=False,
    ).cpu()
    source_state = source.state_dict()
    target_state = target.state_dict()
    shared_keys = set(source_state) & set(target_state)
    exact_keys = {
        key for key in shared_keys if source_state[key].shape == target_state[key].shape
    }
    source_parameters = dict(source.named_parameters())
    target_parameters = dict(target.named_parameters())
    shared_parameter_keys = set(source_parameters) & set(target_parameters)
    exact_parameter_keys = {
        key
        for key in shared_parameter_keys
        if source_parameters[key].shape == target_parameters[key].shape
    }
    source_parameter_count = sum(value.numel() for value in source.parameters())
    target_parameter_count = sum(value.numel() for value in target.parameters())
    exact_source_parameters = sum(
        source_parameters[key].numel() for key in exact_parameter_keys
    )
    exact_target_parameters = sum(
        target_parameters[key].numel() for key in exact_parameter_keys
    )
    mismatches = [
        {
            "key": key,
            "source_shape": list(source_state[key].shape),
            "target_shape": list(target_state[key].shape),
            "source_parameters": source_state[key].numel(),
            "target_parameters": target_state[key].numel(),
        }
        for key in sorted(shared_keys - exact_keys)
    ]

    target.load(source, verbose=False)
    source_stem = source.model[0].conv.weight.detach()
    target_stem = target.model[0].conv.weight.detach()
    copied_channels = min(source_stem.shape[1], target_stem.shape[1])
    stem_prefix_matches = bool(
        torch.equal(target_stem[:, :copied_channels], source_stem[:, :copied_channels])
    )
    extra_stem = target_stem[:, copied_channels:]

    target.args = get_cfg()
    target.train()
    batch = {
        "img": torch.rand(1, args.channels, args.imgsz, args.imgsz),
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[2.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
    }
    loss, parts = target.loss(batch)
    total_loss = loss.sum()
    total_loss.backward()
    stem_gradient = target.model[0].conv.weight.grad
    head = target.model[-1]
    box_head_gradients = [
        layer.weight.grad
        for branch_name in ("cv2", "one2one_cv2")
        for block in (getattr(head, branch_name, None) or [])
        for layer in [block[-1]]
    ]
    finite_nonzero_box_gradients = [
        gradient
        for gradient in box_head_gradients
        if gradient is not None
        and bool(torch.isfinite(gradient).all())
        and float(gradient.abs().sum()) > 0.0
    ]
    if isinstance(parts, dict):
        rendered_parts: dict[str, float] | list[float] = {
            key: float(value) for key, value in parts.items()
        }
        finite_parts = all(bool(torch.isfinite(value).all()) for value in parts.values())
    else:
        rendered_parts = [float(value) for value in parts]
        finite_parts = bool(torch.isfinite(parts).all())

    yaml_reg_max = (
        int(target.yaml["reg_max"])
        if isinstance(target.yaml, dict) and "reg_max" in target.yaml
        else None
    )
    report = {
        "model_source_kind": "checkpoint_native",
        "weights": str(args.weights.resolve()),
        "weights_bytes": args.weights.stat().st_size,
        "weights_sha256": _sha256(args.weights),
        "channels": args.channels,
        "classes": args.classes,
        "imgsz": args.imgsz,
        "source_model_class": type(source).__name__,
        "source_reg_max": int(source.model[-1].reg_max),
        "source_dfl_module": type(source.model[-1].dfl).__name__,
        "source_end2end": bool(getattr(source.model[-1], "end2end", False)),
        "target_reg_max": int(head.reg_max),
        "target_yaml_reg_max": yaml_reg_max,
        "target_dfl_module": type(head.dfl).__name__,
        "target_dfl_is_identity": isinstance(head.dfl, torch.nn.Identity),
        "target_end2end": bool(getattr(head, "end2end", False)),
        "target_box_output_channels": _branch_output_channels(head, "cv2"),
        "target_one2one_box_output_channels": _branch_output_channels(
            head, "one2one_cv2"
        ),
        "source_parameter_count": source_parameter_count,
        "target_parameter_count": target_parameter_count,
        "exact_shape_source_parameters": exact_source_parameters,
        "exact_shape_target_parameters": exact_target_parameters,
        "source_exact_shape_parameter_fraction": exact_source_parameters
        / source_parameter_count,
        "target_exact_shape_parameter_fraction": exact_target_parameters
        / target_parameter_count,
        "shape_mismatches": mismatches,
        "stem_shape": list(target_stem.shape),
        "stem_prefix_matches_source": stem_prefix_matches,
        "extra_stem_finite": bool(torch.isfinite(extra_stem).all()),
        "extra_stem_nonzero": bool(extra_stem.abs().sum() > 0),
        "loss": float(total_loss.detach()),
        "loss_vector": [float(value) for value in loss.detach()],
        "loss_parts": rendered_parts,
        "loss_finite": bool(torch.isfinite(loss).all()),
        "loss_parts_finite": finite_parts,
        "stem_gradient_finite": bool(
            stem_gradient is not None and torch.isfinite(stem_gradient).all()
        ),
        "stem_gradient_abs_sum": (
            float(stem_gradient.abs().sum()) if stem_gradient is not None else 0.0
        ),
        "finite_nonzero_box_gradient_tensors": len(finite_nonzero_box_gradients),
    }
    expected_box_channels = [4 * args.expected_reg_max] * 3
    failures: list[str] = []
    if report["source_reg_max"] != args.expected_reg_max:
        failures.append("source checkpoint reg_max does not match the requested value")
    if report["target_reg_max"] != args.expected_reg_max:
        failures.append("target reg_max does not match the requested value")
    if report["source_dfl_module"] != "DFL" or report["target_dfl_is_identity"]:
        failures.append("checkpoint-native true DFL contract failed")
    if report["target_box_output_channels"] != expected_box_channels:
        failures.append("target box output channels are incorrect")
    one2one_channels = report["target_one2one_box_output_channels"]
    if one2one_channels is not None and one2one_channels != expected_box_channels:
        failures.append("target one-to-one box output channels are incorrect")
    if report["source_exact_shape_parameter_fraction"] < 0.99:
        failures.append("less than 99% of source parameters have exact target shapes")
    if (
        not stem_prefix_matches
        or not report["extra_stem_finite"]
        or not report["extra_stem_nonzero"]
    ):
        failures.append("16-channel stem transfer/initialization contract failed")
    if not report["loss_finite"] or not report["loss_parts_finite"]:
        failures.append("loss is not finite")
    if not report["stem_gradient_finite"] or report["stem_gradient_abs_sum"] <= 0.0:
        failures.append("stem gradient is not finite and nonzero")
    if not finite_nonzero_box_gradients:
        failures.append("box-head gradients are not finite and nonzero")
    report["passed"] = not failures
    report["failures"] = failures
    if failures:
        raise RuntimeError(
            "YOLO checkpoint-native smoke failed: " + "; ".join(failures)
        )

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
