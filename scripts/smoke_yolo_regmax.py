"""CPU smoke for a YAML-defined YOLO reg_max variant with weight transfer."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--model-yaml", type=Path, required=True)
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
    if not args.model_yaml.is_file() or not args.weights.is_file():
        parser.error("--model-yaml and --weights must be existing files")

    torch.manual_seed(2026)
    source = YOLO(str(args.weights.resolve())).model.cpu()
    target = DetectionModel(
        str(args.model_yaml.resolve()),
        ch=args.channels,
        nc=args.classes,
        verbose=False,
    ).cpu()
    source_state = source.state_dict()
    target_state = target.state_dict()
    shared_keys = set(source_state) & set(target_state)
    exact_keys = sorted(
        key for key in shared_keys if source_state[key].shape == target_state[key].shape
    )
    mismatches = [
        {
            "key": key,
            "source_shape": list(source_state[key].shape),
            "target_shape": list(target_state[key].shape),
            "source_parameters": source_state[key].numel(),
            "target_parameters": target_state[key].numel(),
        }
        for key in sorted(shared_keys - set(exact_keys))
    ]
    source_state_element_count = sum(tensor.numel() for tensor in source_state.values())
    target_state_element_count = sum(tensor.numel() for tensor in target_state.values())
    exact_source_state_elements = sum(source_state[key].numel() for key in exact_keys)
    exact_target_state_elements = sum(target_state[key].numel() for key in exact_keys)
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
    regmax_head_mismatch_source_parameters = sum(
        source_parameters[key].numel()
        for key in shared_parameter_keys - exact_parameter_keys
        if ".cv2." in key or ".one2one_cv2." in key
    )

    target.load(source, verbose=False)
    source_stem = source.model[0].conv.weight.detach()
    target_stem = target.model[0].conv.weight.detach()
    copied_channels = min(source_stem.shape[1], target_stem.shape[1])
    stem_prefix_matches = bool(
        torch.equal(target_stem[:, :copied_channels], source_stem[:, :copied_channels])
    )
    extra_stem = target_stem[:, copied_channels:]
    extra_stem_finite = bool(torch.isfinite(extra_stem).all())
    extra_stem_nonzero = bool(extra_stem.abs().sum() > 0)

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

    report = {
        "model_yaml": str(args.model_yaml.resolve()),
        "model_yaml_sha256": _sha256(args.model_yaml),
        "weights": str(args.weights.resolve()),
        "weights_sha256": _sha256(args.weights),
        "channels": args.channels,
        "classes": args.classes,
        "imgsz": args.imgsz,
        "source_reg_max": int(source.model[-1].reg_max),
        "target_reg_max": int(head.reg_max),
        "target_yaml_reg_max": int(target.yaml["reg_max"]),
        "target_dfl_module": type(head.dfl).__name__,
        "target_dfl_is_identity": isinstance(head.dfl, torch.nn.Identity),
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
        "regmax_head_mismatch_source_parameters": regmax_head_mismatch_source_parameters,
        "source_state_element_count": source_state_element_count,
        "target_state_element_count": target_state_element_count,
        "exact_shape_source_state_elements": exact_source_state_elements,
        "exact_shape_target_state_elements": exact_target_state_elements,
        "source_exact_shape_state_fraction": exact_source_state_elements
        / source_state_element_count,
        "target_exact_shape_state_fraction": exact_target_state_elements
        / target_state_element_count,
        "shape_mismatches": mismatches,
        "stem_shape": list(target_stem.shape),
        "stem_prefix_matches_source": stem_prefix_matches,
        "extra_stem_finite": extra_stem_finite,
        "extra_stem_nonzero": extra_stem_nonzero,
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
    if report["source_reg_max"] != 1:
        failures.append("source checkpoint is not reg_max=1")
    if report["target_reg_max"] != args.expected_reg_max:
        failures.append("target reg_max does not match the requested value")
    if report["target_dfl_is_identity"]:
        failures.append("target DFL module is still Identity")
    if report["target_box_output_channels"] != expected_box_channels:
        failures.append("target one-to-many box output channels are incorrect")
    if report["target_one2one_box_output_channels"] != expected_box_channels:
        failures.append("target one-to-one box output channels are incorrect")
    if report["source_exact_shape_parameter_fraction"] < 0.99:
        failures.append("less than 99% of source parameters have exact target shapes")
    if not stem_prefix_matches or not extra_stem_finite or not extra_stem_nonzero:
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
        raise RuntimeError("YOLO reg_max smoke failed: " + "; ".join(failures))

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
