from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Iterator

import numpy as np
import torch
import yaml
from PIL import Image

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".ultralytics").resolve()))

from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics, box_iou

from predict_submission import (
    PredictionArrays,
    _checkpoint_channels,
    _collect_multiscale_predictions,
    _fused_multiscale_predictions,
    _normalize_multiscale,
    _numeric_paths,
)


def _metric_record(metrics: object, elapsed_seconds: float) -> dict[str, object]:
    box = metrics.box
    return {
        "map50_95": float(box.map),
        "map50": float(box.map50),
        "map75": float(box.map75),
        "per_class_map50_95": [float(value) for value in box.maps],
        "elapsed_seconds": elapsed_seconds,
        "speed_ms_per_image": {
            key: float(value) for key, value in metrics.speed.items()
        },
    }


def _dataset_details(data_path: Path) -> tuple[dict[int, str], Path]:
    config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    root = Path(config["path"])
    if not root.is_absolute():
        root = data_path.parent / root
    val = config["val"]
    if not isinstance(val, str):
        raise ValueError("Multi-scale comparison requires one validation image directory")
    val_directory = Path(val)
    if not val_directory.is_absolute():
        val_directory = root / val_directory
    names_config = config["names"]
    if isinstance(names_config, list):
        names = dict(enumerate(str(name) for name in names_config))
    else:
        names = {int(index): str(name) for index, name in names_config.items()}
    return names, val_directory.resolve()


def _ground_truth(path: Path) -> tuple[np.ndarray, np.ndarray]:
    preview_path = path.with_suffix(".png")
    if not preview_path.is_file():
        raise FileNotFoundError(f"PNG shape reference not found for {path.resolve()}")
    with Image.open(preview_path) as image:
        width, height = image.size
    dataset_root = path.parents[2]
    label_path = dataset_root / "labels" / path.parent.name / f"{path.stem}.txt"
    if not label_path.is_file():
        raise FileNotFoundError(f"Validation label not found: {label_path.resolve()}")
    boxes: list[list[float]] = []
    classes: list[float] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"Invalid YOLO label in {label_path.resolve()}: {line!r}")
        class_id, center_x, center_y, box_width, box_height = map(float, values)
        boxes.append(
            [
                (center_x - box_width / 2.0) * width,
                (center_y - box_height / 2.0) * height,
                (center_x + box_width / 2.0) * width,
                (center_y + box_height / 2.0) * height,
            ]
        )
        classes.append(class_id)
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4), np.asarray(classes, dtype=np.float32)


def _match_predictions(
    pred_classes: torch.Tensor,
    true_classes: torch.Tensor,
    iou: torch.Tensor,
    iouv: torch.Tensor,
) -> np.ndarray:
    correct = np.zeros((pred_classes.shape[0], iouv.shape[0]), dtype=bool)
    correct_class = true_classes[:, None] == pred_classes
    iou_array = (iou * correct_class).cpu().numpy()
    for threshold_index, threshold in enumerate(iouv.cpu().tolist()):
        matches = np.asarray(np.nonzero(iou_array >= threshold)).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou_array[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), threshold_index] = True
    return correct


def _evaluate_predictions(
    predictions: Iterator[PredictionArrays],
    names: dict[int, str],
) -> dict[str, object]:
    started = time.perf_counter()
    metrics = DetMetrics(names=names)
    iouv = torch.linspace(0.5, 0.95, 10)
    image_count = 0
    prediction_count = 0
    for prediction in predictions:
        target_boxes_array, target_classes_array = _ground_truth(prediction.path)
        target_boxes = torch.from_numpy(target_boxes_array)
        target_classes = torch.from_numpy(target_classes_array)
        predicted_boxes = torch.from_numpy(prediction.boxes.astype(np.float32, copy=False))
        predicted_classes = torch.from_numpy(prediction.classes.astype(np.float32, copy=False))
        if len(target_boxes) and len(predicted_boxes):
            correct = _match_predictions(
                predicted_classes,
                target_classes,
                box_iou(target_boxes, predicted_boxes),
                iouv,
            )
        else:
            correct = np.zeros((len(predicted_boxes), len(iouv)), dtype=bool)
        metrics.update_stats(
            {
                "tp": correct,
                "target_cls": target_classes_array,
                "target_img": np.unique(target_classes_array),
                "conf": prediction.confidences.astype(np.float32, copy=False),
                "pred_cls": prediction.classes.astype(np.float32, copy=False),
                "im_name": prediction.path.name,
            }
        )
        image_count += 1
        prediction_count += len(prediction.boxes)
    metrics.process(plot=False)
    return {
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "map75": float(metrics.box.map75),
        "per_class_map50_95": [float(value) for value in metrics.box.maps],
        "elapsed_seconds": time.perf_counter() - started,
        "images": image_count,
        "predictions": prediction_count,
    }


def _ordered_predictions(
    predictions: dict[Path, PredictionArrays],
    paths: list[Path],
) -> Iterator[PredictionArrays]:
    for path in paths:
        yield predictions[path]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare standard and single-model TTA validation for one checkpoint"
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.0001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--multi-scale",
        nargs="+",
        type=int,
        default=[960, 1024, 1088],
        help="Image sizes for same-checkpoint multi-scale box voting.",
    )
    parser.add_argument(
        "--fusion-ious",
        nargs="+",
        type=float,
        default=[0.55, 0.65, 0.75],
        help="Coordinate-voting IoU thresholds evaluated from the same predictions.",
    )
    parser.add_argument(
        "--control-tolerance",
        type=float,
        default=0.001,
        help="Maximum allowed mAP difference between model.val and the custom evaluator.",
    )
    parser.add_argument(
        "--minimum-gain",
        type=float,
        default=0.001,
        help="Require this held-out mAP50-95 gain before recommending TTA.",
    )
    args = parser.parse_args()

    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights.resolve()}")
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data.resolve()}")

    model = YOLO(str(args.weights.resolve()))
    results: dict[str, dict[str, object]] = {}
    for label, augment in (("standard", False), ("tta", True)):
        started = time.perf_counter()
        metrics = model.val(
            data=str(args.data.resolve()),
            split="val",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            half=True,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            augment=augment,
            plots=False,
            save_json=False,
            verbose=False,
            project=str(Path("artifacts/inference_mode_validation").resolve()),
            name=f"{args.name}_{label}",
            exist_ok=True,
        )
        results[label] = _metric_record(metrics, time.perf_counter() - started)

    names, val_directory = _dataset_details(args.data.resolve())
    channels = _checkpoint_channels(model)
    input_format = "npy" if channels > 3 else "png"
    image_paths = _numeric_paths(val_directory, f".{input_format}")
    if not image_paths:
        raise FileNotFoundError(
            f"No {input_format.upper()} validation inputs found in {val_directory}"
        )
    scales = _normalize_multiscale([*args.multi_scale, args.imgsz])
    prediction_started = time.perf_counter()
    predictions_by_scale = _collect_multiscale_predictions(
        model,
        image_paths,
        image_directory=val_directory,
        input_format=input_format,
        channels=channels,
        batch_size=args.batch,
        scales=scales,
        predict_kwargs={
            "batch": args.batch,
            "device": args.device,
            "half": True,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "verbose": False,
        },
    )
    prediction_elapsed = time.perf_counter() - prediction_started
    results["custom_standard"] = _evaluate_predictions(
        _ordered_predictions(predictions_by_scale[args.imgsz], image_paths),
        names,
    )
    fusion_keys: dict[str, float] = {}
    for fusion_iou in args.fusion_ious:
        key = f"multiscale_box_vote_iou_{fusion_iou:.2f}".replace(".", "_")
        results[key] = _evaluate_predictions(
            _fused_multiscale_predictions(
                predictions_by_scale,
                image_paths,
                fusion_iou=fusion_iou,
                max_det=args.max_det,
            ),
            names,
        )
        fusion_keys[key] = fusion_iou

    standard_map = float(results["standard"]["map50_95"])
    custom_standard_map = float(results["custom_standard"]["map50_95"])
    control_delta = custom_standard_map - standard_map
    control_passed = abs(control_delta) <= args.control_tolerance
    eligible_modes = {"standard": standard_map, "tta": float(results["tta"]["map50_95"])}
    if control_passed:
        eligible_modes.update(
            {key: float(results[key]["map50_95"]) for key in fusion_keys}
        )
    best_mode = max(eligible_modes, key=eligible_modes.__getitem__)
    best_gain = eligible_modes[best_mode] - standard_map
    if best_gain < args.minimum_gain:
        best_mode = "standard"
        best_gain = 0.0
    chosen_fusion_iou = fusion_keys.get(best_mode)
    report = {
        "schema_version": 1,
        "weights": str(args.weights.resolve()),
        "data": str(args.data.resolve()),
        "settings": {
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "half": True,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "minimum_gain": args.minimum_gain,
            "multi_scale": list(scales),
            "fusion_ious": args.fusion_ious,
            "control_tolerance": args.control_tolerance,
            "multi_scale_prediction_elapsed_seconds": prediction_elapsed,
        },
        "results": results,
        "custom_evaluator_control": {
            "delta_map50_95": control_delta,
            "passed": control_passed,
        },
        "recommendation": {
            "mode": best_mode,
            "augment": best_mode == "tta",
            "multi_scale": list(scales) if best_mode in fusion_keys else [],
            "fusion_iou": chosen_fusion_iou,
            "gain_map50_95": best_gain,
            "tta_gain_map50_95": float(results["tta"]["map50_95"]) - standard_map,
            "reason": (
                f"{best_mode} cleared the held-out gain threshold"
                if best_mode != "standard"
                else "No alternate inference mode cleared the held-out gain threshold"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
