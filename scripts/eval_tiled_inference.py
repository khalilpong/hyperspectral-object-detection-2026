from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".ultralytics").resolve()))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ultralytics import YOLO

from compare_inference_modes import _dataset_details, _evaluate_predictions
from predict_submission import (
    PredictionArrays,
    _checkpoint_channels,
    _collect_tiled_predictions,
    _fused_full_and_tiled_predictions,
    _fused_multiscale_predictions,
    _numeric_paths,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remap_full_cache(
    cache_path: Path,
    image_paths: list[Path],
) -> dict[int, dict[Path, PredictionArrays]]:
    with cache_path.open("rb") as handle:
        stored = pickle.load(handle)
    if not isinstance(stored, dict) or len(stored) < 2:
        raise ValueError("Full-image cache must contain at least two scale passes")
    expected_stems = {path.stem for path in image_paths}
    remapped: dict[int, dict[Path, PredictionArrays]] = {}
    for scale, records in stored.items():
        by_stem = {Path(path).stem: record for path, record in records.items()}
        if set(by_stem) != expected_stems:
            raise ValueError(
                f"Full-image cache scale {scale} has {len(by_stem)} images; "
                f"expected {len(expected_stems)}"
            )
        remapped[int(scale)] = {
            path: PredictionArrays(
                path=path,
                boxes=by_stem[path.stem].boxes,
                classes=by_stem[path.stem].classes,
                confidences=by_stem[path.stem].confidences,
                orig_shape=by_stem[path.stem].orig_shape,
            )
            for path in image_paths
        }
    return remapped


def _tile_manifest(
    *,
    weights: Path,
    data: Path,
    image_directory: Path,
    tile_size: tuple[int, int],
    tile_stride: tuple[int, int],
    tile_imgsz: int,
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
        "tile_size": list(tile_size),
        "tile_stride": list(tile_stride),
        "tile_imgsz": tile_imgsz,
        "conf": conf,
        "iou": iou,
        "max_det": max_det,
    }


def _remap_tile_cache(
    stored: dict[Path, list[PredictionArrays]], image_paths: list[Path]
) -> dict[Path, list[PredictionArrays]]:
    by_stem = {Path(path).stem: records for path, records in stored.items()}
    expected_stems = {path.stem for path in image_paths}
    if set(by_stem) != expected_stems:
        raise ValueError(
            f"Tile cache has {len(by_stem)} images; expected {len(expected_stems)}"
        )
    return {
        path: [
            PredictionArrays(
                path=path,
                boxes=record.boxes,
                classes=record.classes,
                confidences=record.confidences,
                orig_shape=record.orig_shape,
            )
            for record in by_stem[path.stem]
        ]
        for path in image_paths
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate same-checkpoint tile inference while reusing validated full-scale passes"
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--full-cache", type=Path, required=True)
    parser.add_argument("--tile-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tile-size", nargs=2, type=int, default=(128, 256))
    parser.add_argument("--tile-stride", nargs=2, type=int, default=(96, 192))
    parser.add_argument("--tile-imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.0001)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--full-fusion-iou", type=float, default=0.70)
    parser.add_argument("--full-support-gain", type=float, default=0.125)
    parser.add_argument("--tile-fusion-ious", nargs="+", type=float, default=(0.55, 0.60, 0.65))
    parser.add_argument("--tile-support-gains", nargs="+", type=float, default=(0.0,))
    parser.add_argument("--expected-full-map", type=float)
    parser.add_argument("--control-tolerance", type=float, default=0.0002)
    parser.add_argument("--minimum-gain", type=float, default=0.003)
    args = parser.parse_args()

    for path in (args.weights, args.data, args.full_cache):
        if not path.is_file():
            raise FileNotFoundError(path.resolve())
    tile_size = tuple(args.tile_size)
    tile_stride = tuple(args.tile_stride)
    if any(value <= 0 for value in (*tile_size, *tile_stride, args.tile_imgsz, args.batch)):
        parser.error("batch, tile size, stride, and tile-imgsz must be positive")
    if any(stride > size for stride, size in zip(tile_stride, tile_size, strict=True)):
        parser.error("each tile stride must not exceed its tile size")

    names, val_directory = _dataset_details(args.data.resolve())
    image_paths = _numeric_paths(val_directory, ".npy")
    if not image_paths:
        raise FileNotFoundError(f"No NPY validation images in {val_directory}")
    full_predictions = _remap_full_cache(args.full_cache.resolve(), image_paths)

    expected_manifest = _tile_manifest(
        weights=args.weights.resolve(),
        data=args.data.resolve(),
        image_directory=val_directory,
        tile_size=tile_size,
        tile_stride=tile_stride,
        tile_imgsz=args.tile_imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
    )
    manifest_path = args.tile_cache.with_suffix(args.tile_cache.suffix + ".json")
    tile_started = time.perf_counter()
    cache_reused = False
    if args.tile_cache.is_file() and manifest_path.is_file():
        actual_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if actual_manifest == expected_manifest:
            with args.tile_cache.open("rb") as handle:
                tiled_predictions = _remap_tile_cache(pickle.load(handle), image_paths)
            cache_reused = True
        else:
            raise ValueError("Tile cache manifest does not match this evaluation")
    else:
        model = YOLO(str(args.weights.resolve()))
        channels = _checkpoint_channels(model)
        if channels <= 3:
            raise ValueError(f"Tile evaluator expected an HSI checkpoint, got {channels} channels")
        tiled_predictions = _collect_tiled_predictions(
            model,
            image_paths,
            channels=channels,
            batch_size=args.batch,
            tile_height=tile_size[0],
            tile_width=tile_size[1],
            stride_height=tile_stride[0],
            stride_width=tile_stride[1],
            tile_imgsz=args.tile_imgsz,
            predict_kwargs={
                "batch": args.batch,
                "device": args.device,
                "quantize": 16,
                "conf": args.conf,
                "iou": args.iou,
                "max_det": args.max_det,
                "verbose": False,
            },
        )
        args.tile_cache.parent.mkdir(parents=True, exist_ok=True)
        with args.tile_cache.open("wb") as handle:
            pickle.dump(tiled_predictions, handle)
        manifest_path.write_text(json.dumps(expected_manifest, indent=2), encoding="utf-8")
    tile_elapsed = time.perf_counter() - tile_started

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
    control_delta = None
    control_passed = True
    if args.expected_full_map is not None:
        control_delta = float(legacy_full["map50_95"]) - args.expected_full_map
        control_passed = abs(control_delta) <= args.control_tolerance

    candidates: dict[str, dict[str, object]] = {}
    for fusion_iou in args.tile_fusion_ious:
        for support_gain in args.tile_support_gains:
            key = f"tile_iou_{fusion_iou:.3f}_support_{support_gain:.3f}"
            metrics = _evaluate_predictions(
                _fused_full_and_tiled_predictions(
                    full_predictions,
                    tiled_predictions,
                    image_paths,
                    fusion_iou=fusion_iou,
                    max_det=args.max_det,
                    support_gain=support_gain,
                ),
                names,
            )
            metrics["gain_vs_legacy_full"] = (
                float(metrics["map50_95"]) - float(legacy_full["map50_95"])
            )
            metrics["gain_vs_supported_full"] = (
                float(metrics["map50_95"]) - float(supported_full["map50_95"])
            )
            candidates[key] = metrics

    best_key = max(candidates, key=lambda key: float(candidates[key]["map50_95"]))
    best_gain = float(candidates[best_key]["gain_vs_legacy_full"])
    passed = control_passed and best_gain >= args.minimum_gain
    report = {
        "schema_version": 1,
        "weights": str(args.weights.resolve()),
        "data": str(args.data.resolve()),
        "full_cache": str(args.full_cache.resolve()),
        "full_cache_sha256": _sha256(args.full_cache.resolve()),
        "tile_cache": str(args.tile_cache.resolve()),
        "tile_cache_reused": cache_reused,
        "tile_prediction_elapsed_seconds": tile_elapsed,
        "settings": {
            "tile_size": list(tile_size),
            "tile_stride": list(tile_stride),
            "tile_imgsz": args.tile_imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "full_fusion_iou": args.full_fusion_iou,
            "full_support_gain": args.full_support_gain,
            "minimum_gain": args.minimum_gain,
        },
        "control": {
            "expected_full_map": args.expected_full_map,
            "delta": control_delta,
            "tolerance": args.control_tolerance,
            "passed": control_passed,
        },
        "legacy_full": legacy_full,
        "supported_full": supported_full,
        "candidates": candidates,
        "recommendation": {
            "best": best_key,
            "gain_vs_legacy_full": best_gain,
            "passed": passed,
            "reason": (
                "tile candidate cleared control and gain gates"
                if passed
                else "tile candidate did not clear control and gain gates"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
