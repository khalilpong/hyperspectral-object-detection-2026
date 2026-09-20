from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Iterator

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".ultralytics").resolve()))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ultralytics import YOLO

from compare_inference_modes import _dataset_details, _evaluate_predictions
from eval_tiled_inference import _remap_full_cache
from predict_submission import (
    PredictionArrays,
    _checkpoint_channels,
    _collect_horizontal_flip_predictions,
    _fused_multiscale_predictions,
    _fused_prediction_sources,
    _numeric_paths,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_predictions(
    predictions: dict[Path, PredictionArrays], image_paths: list[Path]
) -> Iterator[PredictionArrays]:
    for path in image_paths:
        yield predictions[path]


def _remap_flip_cache(
    stored: dict[Path, PredictionArrays], image_paths: list[Path]
) -> dict[Path, PredictionArrays]:
    if not isinstance(stored, dict):
        raise ValueError("Horizontal-flip cache must be a prediction dictionary")
    by_stem: dict[str, PredictionArrays] = {}
    for path, record in stored.items():
        stem = Path(path).stem
        if stem in by_stem:
            raise ValueError(f"Horizontal-flip cache contains duplicate image stem {stem}")
        by_stem[stem] = record
    expected_stems = {path.stem for path in image_paths}
    if set(by_stem) != expected_stems:
        raise ValueError(
            f"Horizontal-flip cache has {len(by_stem)} images; expected {len(expected_stems)}"
        )
    return {
        path: PredictionArrays(
            path=path,
            boxes=by_stem[path.stem].boxes,
            classes=by_stem[path.stem].classes,
            confidences=by_stem[path.stem].confidences,
            orig_shape=by_stem[path.stem].orig_shape,
        )
        for path in image_paths
    }


def _flip_manifest(
    *,
    weights: Path,
    data: Path,
    image_directory: Path,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "weights": str(weights.resolve()),
        "weights_sha256": _sha256(weights),
        "data": str(data.resolve()),
        "data_sha256": _sha256(data),
        "image_directory": str(image_directory.resolve()),
        "transform": "horizontal_flip_all_channels",
        "box_inverse": "x1=width-x2,x2=width-x1",
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "max_det": max_det,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an isolated horizontal-flip pass from one checkpoint while reusing "
            "validated original-image scale caches"
        )
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--full-cache", type=Path, required=True)
    parser.add_argument("--flip-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.0001)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--full-fusion-iou", type=float, default=0.74)
    parser.add_argument("--full-support-gain", type=float, default=0.125)
    parser.add_argument(
        "--flip-fusion-ious",
        nargs="+",
        type=float,
        default=(0.55, 0.60, 0.65, 0.70, 0.74, 0.78, 0.82),
    )
    parser.add_argument(
        "--flip-support-gains", nargs="+", type=float, default=(0.0, 0.125)
    )
    parser.add_argument("--expected-full-map", type=float)
    parser.add_argument("--control-tolerance", type=float, default=0.0002)
    parser.add_argument("--minimum-gain", type=float, default=0.001)
    args = parser.parse_args()

    for path in (args.weights, args.data, args.full_cache):
        if not path.is_file():
            raise FileNotFoundError(path.resolve())
    if any(value <= 0 for value in (args.batch, args.imgsz, args.max_det)):
        parser.error("batch, imgsz, and max-det must be positive")
    if not all(0.0 < value <= 1.0 for value in args.flip_fusion_ious):
        parser.error("every flip fusion IoU must be in (0, 1]")
    if any(value < 0.0 for value in args.flip_support_gains):
        parser.error("flip support gains must be non-negative")

    names, val_directory = _dataset_details(args.data.resolve())
    image_paths = _numeric_paths(val_directory, ".npy")
    if not image_paths:
        raise FileNotFoundError(f"No NPY validation images in {val_directory}")
    full_predictions = _remap_full_cache(args.full_cache.resolve(), image_paths)
    if args.imgsz not in full_predictions:
        raise ValueError(
            f"Full-image cache has scales {sorted(full_predictions)}, not requested {args.imgsz}"
        )

    expected_manifest = _flip_manifest(
        weights=args.weights.resolve(),
        data=args.data.resolve(),
        image_directory=val_directory,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
    )
    manifest_path = args.flip_cache.with_suffix(args.flip_cache.suffix + ".json")
    flip_started = time.perf_counter()
    cache_reused = False
    if args.flip_cache.is_file() and manifest_path.is_file():
        actual_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if actual_manifest != expected_manifest:
            raise ValueError("Horizontal-flip cache manifest does not match this evaluation")
        with args.flip_cache.open("rb") as handle:
            flip_predictions = _remap_flip_cache(pickle.load(handle), image_paths)
        cache_reused = True
    else:
        model = YOLO(str(args.weights.resolve()))
        channels = _checkpoint_channels(model)
        if channels <= 3:
            raise ValueError(f"Flip evaluator expected an HSI checkpoint, got {channels} channels")
        flip_predictions = _collect_horizontal_flip_predictions(
            model,
            image_paths,
            channels=channels,
            batch_size=args.batch,
            predict_kwargs={
                "batch": args.batch,
                "device": args.device,
                "quantize": 16,
                "imgsz": args.imgsz,
                "conf": args.conf,
                "iou": args.iou,
                "max_det": args.max_det,
                "verbose": False,
            },
        )
        args.flip_cache.parent.mkdir(parents=True, exist_ok=True)
        with args.flip_cache.open("wb") as handle:
            pickle.dump(flip_predictions, handle)
        manifest_path.write_text(json.dumps(expected_manifest, indent=2), encoding="utf-8")
    flip_elapsed = time.perf_counter() - flip_started

    original_1024 = _evaluate_predictions(
        _ordered_predictions(full_predictions[args.imgsz], image_paths), names
    )
    flip_only = _evaluate_predictions(
        _ordered_predictions(flip_predictions, image_paths), names
    )
    supported_full = _evaluate_predictions(
        _fused_multiscale_predictions(
            full_predictions,
            image_paths,
            fusion_iou=args.full_fusion_iou,
            max_det=args.max_det,
            support_gain=args.full_support_gain,
        ),
        names,
    )
    legacy_full = _evaluate_predictions(
        _fused_multiscale_predictions(
            full_predictions,
            image_paths,
            fusion_iou=args.full_fusion_iou,
            max_det=args.max_det,
            support_gain=0.0,
        ),
        names,
    )

    control_delta = None
    control_passed = True
    if args.expected_full_map is not None:
        control_delta = float(supported_full["map50_95"]) - args.expected_full_map
        control_passed = abs(control_delta) <= args.control_tolerance

    single_scale_candidates: dict[str, dict[str, object]] = {}
    full_plus_flip_candidates: dict[str, dict[str, object]] = {}
    original_scale = full_predictions[args.imgsz]
    full_sources = [full_predictions[scale] for scale in full_predictions]
    for fusion_iou in args.flip_fusion_ious:
        for support_gain in args.flip_support_gains:
            key = f"iou_{fusion_iou:.3f}_support_{support_gain:.3f}"
            single_metrics = _evaluate_predictions(
                _fused_prediction_sources(
                    [original_scale, flip_predictions],
                    image_paths,
                    fusion_iou=fusion_iou,
                    max_det=args.max_det,
                    support_gain=support_gain,
                ),
                names,
            )
            single_metrics["gain_vs_original_1024"] = (
                float(single_metrics["map50_95"]) - float(original_1024["map50_95"])
            )
            single_scale_candidates[key] = single_metrics

            full_metrics = _evaluate_predictions(
                _fused_prediction_sources(
                    [*full_sources, flip_predictions],
                    image_paths,
                    fusion_iou=fusion_iou,
                    max_det=args.max_det,
                    support_gain=support_gain,
                ),
                names,
            )
            full_metrics["gain_vs_supported_full"] = (
                float(full_metrics["map50_95"]) - float(supported_full["map50_95"])
            )
            full_plus_flip_candidates[key] = full_metrics

    best_key = max(
        full_plus_flip_candidates,
        key=lambda key: float(full_plus_flip_candidates[key]["map50_95"]),
    )
    best_gain = float(full_plus_flip_candidates[best_key]["gain_vs_supported_full"])
    passed = control_passed and best_gain >= args.minimum_gain
    report = {
        "schema_version": 1,
        "weights": str(args.weights.resolve()),
        "weights_sha256": _sha256(args.weights.resolve()),
        "data": str(args.data.resolve()),
        "full_cache": str(args.full_cache.resolve()),
        "full_cache_sha256": _sha256(args.full_cache.resolve()),
        "flip_cache": str(args.flip_cache.resolve()),
        "flip_cache_sha256": _sha256(args.flip_cache.resolve()),
        "flip_cache_reused": cache_reused,
        "flip_prediction_elapsed_seconds": flip_elapsed,
        "settings": {
            "transform": "horizontal flip of all 16 channels",
            "inverse_box_transform": "x1=width-x2,x2=width-x1",
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "quantize": 16,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "full_scales": list(full_predictions),
            "full_fusion_iou": args.full_fusion_iou,
            "full_support_gain": args.full_support_gain,
            "flip_fusion_ious": args.flip_fusion_ious,
            "flip_support_gains": args.flip_support_gains,
            "minimum_gain": args.minimum_gain,
        },
        "control": {
            "expected_full_map": args.expected_full_map,
            "actual_full_map": supported_full["map50_95"],
            "delta": control_delta,
            "tolerance": args.control_tolerance,
            "passed": control_passed,
        },
        "original_1024": original_1024,
        "flip_only": flip_only,
        "legacy_full": legacy_full,
        "supported_full": supported_full,
        "single_scale_original_plus_flip": single_scale_candidates,
        "full_plus_flip": full_plus_flip_candidates,
        "recommendation": {
            "best": best_key,
            "gain_vs_supported_full": best_gain,
            "passed": passed,
            "reason": (
                "horizontal flip cleared the cache-control and gain gates"
                if passed
                else "horizontal flip did not clear the cache-control and gain gates"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
