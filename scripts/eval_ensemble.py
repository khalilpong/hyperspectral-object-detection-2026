"""Held-out evaluation of single-model or multi-model multi-scale box voting.

Each (checkpoint, scale) pass is cached under --cache so fusion variants can be
re-evaluated without re-running the GPU. Every pass is one voting source, so an
ensemble of 2 models x 7 scales votes over 14 sources with the same _box_vote
used by predict_submission.py.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_inference_modes import _dataset_details, _evaluate_predictions  # noqa: E402
from predict_submission import (  # noqa: E402
    PredictionArrays,
    _box_vote,
    _checkpoint_channels,
    _collect_multiscale_predictions,
    _numeric_paths,
)
from ultralytics import YOLO  # noqa: E402

from hsi_detection.submission import SUBMISSION_COLUMNS, clip_xyxy  # noqa: E402


def _by_stem(passes):
    """Key every scale's predictions by image stem so models fed from different
    image directories (16-band NPY vs 3-band PNG) line up image by image."""
    return {scale: {Path(key).stem: record for key, record in records.items()} for scale, records in passes.items()}


def _cached_passes(tag, weights, image_dir, scales, cache_dir, predict_kwargs):
    cache = cache_dir / f"{tag}.pkl"
    if cache.is_file():
        with cache.open("rb") as handle:
            stored = pickle.load(handle)
        if tuple(stored) == tuple(scales):
            print(f"[{tag}] 使用缓存 {cache}", flush=True)
            return _by_stem(stored)
    model = YOLO(str(weights.resolve()))
    channels = _checkpoint_channels(model)
    input_format = "npy" if channels > 3 else "png"
    image_paths = [p.resolve() for p in _numeric_paths(image_dir, f".{input_format}")]
    print(f"[{tag}] 推理 {weights} ({channels} 通道, {len(image_paths)} 张 {image_dir}) @ {scales}", flush=True)
    passes = _collect_multiscale_predictions(
        model,
        image_paths,
        image_directory=image_dir,
        input_format=input_format,
        channels=channels,
        batch_size=1,
        scales=scales,
        predict_kwargs=predict_kwargs,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as handle:
        pickle.dump(passes, handle)
    return _by_stem(passes)


def _fused(sources, image_paths, fusion_iou, max_det, support_gain=0.0):
    """sources: list of (per-path PredictionArrays dict, confidence multiplier)."""
    for path in image_paths:
        records = [(passes[path.stem], weight) for passes, weight in sources]
        boxes = np.concatenate([r.boxes for r, _ in records], axis=0)
        classes = np.concatenate([r.classes for r, _ in records], axis=0)
        confidences = np.concatenate([r.confidences * w for r, w in records], axis=0)
        source_ids = np.concatenate(
            [np.full(len(r.boxes), i, dtype=np.int64) for i, (r, _) in enumerate(records)]
        )
        fused_boxes, fused_classes, fused_confidences = _box_vote(
            boxes,
            classes,
            confidences,
            iou_threshold=fusion_iou,
            max_det=max_det,
            source_ids=source_ids,
            support_gain=support_gain,
            total_sources=len(records),
        )
        yield PredictionArrays(path, fused_boxes, fused_classes, fused_confidences, records[0][0].orig_shape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, metavar="TAG=WEIGHTS[@MULT][#DATASET_ROOT]",
                        help="@MULT fixes that model's confidence multiplier; models without it (after the "
                             "first) use each --second-weights value in turn. #DATASET_ROOT reads that model's "
                             "images from DATASET_ROOT/images/<val|test> (e.g. data/processed/pseudo_rgb); "
                             "the first model must use the default 16-band directory.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("artifacts/ensemble_cache"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scales", nargs="+", type=int, default=[832, 896, 960, 1024, 1088, 1152, 1216])
    parser.add_argument("--fusion-ious", nargs="+", type=float, default=[0.70])
    parser.add_argument("--second-weights", nargs="+", type=float, default=[1.0],
                        help="Confidence multipliers applied to every model after the first.")
    parser.add_argument("--conf", type=float, default=0.0001)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--support-gain",
        dest="support_gains",
        nargs="+",
        type=float,
        default=[0.0],
        help=(
            "Add a normalized confidence-mass bonus from agreeing model/scale sources. "
            "Pass multiple values for a held-out grid; submission mode accepts one. "
            "Zero preserves the legacy max-confidence ranking."
        ),
    )
    parser.add_argument("--submission", type=Path,
                        help="Write a competition CSV for --images (e.g. test) using the first fusion IoU "
                             "and second weight instead of evaluating held-out labels.")
    parser.add_argument("--images", type=Path, help="Image directory for --submission.")
    args = parser.parse_args()
    if args.submission is not None and len(args.support_gains) != 1:
        parser.error("--submission accepts exactly one --support-gain value")

    names, val_dir = _dataset_details(args.data)
    if args.submission is not None:
        if args.images is None:
            parser.error("--submission requires --images")
        val_dir = args.images.resolve()
    image_paths = [p.resolve() for p in _numeric_paths(val_dir, ".npy")]
    predict_kwargs = dict(device="0", half=True, conf=args.conf, iou=args.iou, max_det=args.max_det, verbose=False)

    models = []
    fixed: dict[int, float] = {}
    for index, spec in enumerate(args.model):
        tag, weights = spec.split("=", 1)
        image_dir = val_dir
        if "#" in weights:
            weights, root = weights.rsplit("#", 1)
            image_dir = (Path(root) / "images" / val_dir.name).resolve()
        if "@" in weights:
            weights, multiplier = weights.rsplit("@", 1)
            fixed[index] = float(multiplier)
        passes = _cached_passes(tag, Path(weights), image_dir, args.scales, args.cache, predict_kwargs)
        missing = {p.stem for p in image_paths} - set(next(iter(passes.values())))
        if missing:
            raise RuntimeError(f"[{tag}] 缺少 {len(missing)} 张图的预测，例如 {sorted(missing)[:3]}")
        models.append((tag, passes))

    def _sources(weight):
        return [
            (passes[s], 1.0 if i == 0 else fixed.get(i, weight))
            for i, (_, passes) in enumerate(models)
            for s in passes
        ]

    def _label(fusion_iou, weight, support_gain):
        tags = "+".join(tag + (f"x{fixed[i]:g}" if i in fixed else "") for i, (tag, _) in enumerate(models))
        return f"{tags} f{fusion_iou:g} w{weight:g} sg{support_gain:g}"

    if args.submission is not None:
        weight, fusion_iou = args.second_weights[0], args.fusion_ious[0]
        support_gain = args.support_gains[0]
        sources = _sources(weight)
        rows, dropped = [], 0
        for prediction in _fused(
            sources, image_paths, fusion_iou, args.max_det, support_gain
        ):
            height, width = prediction.orig_shape
            for box, class_id, confidence in zip(prediction.boxes, prediction.classes, prediction.confidences):
                clipped = clip_xyxy(box.tolist(), width=width, height=height)
                if clipped is None:
                    dropped += 1
                    continue
                rows.append([len(rows), int(prediction.path.stem), int(class_id), float(confidence), *clipped])
        args.submission.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=SUBMISSION_COLUMNS).to_csv(args.submission, index=False)
        print(f"wrote {len(rows)} detections ({dropped} dropped) from {len(image_paths)} images to {args.submission}")
        return

    results = []
    for fusion_iou in args.fusion_ious:
        for tag, passes in models:
            record = _evaluate_predictions(
                _fused([(passes[s], 1.0) for s in passes], image_paths, fusion_iou, args.max_det), names
            )
            results.append({"config": f"{tag} f{fusion_iou:g}", **record})
            print(f"{results[-1]['config']:<32} mAP50-95 {record['map50_95']:.5f}", flush=True)
        if len(models) == 1:
            tag, passes = models[0]
            single_sources = [(passes[scale], 1.0) for scale in passes]
            for support_gain in args.support_gains:
                if support_gain == 0.0:
                    continue
                record = _evaluate_predictions(
                    _fused(
                        single_sources,
                        image_paths,
                        fusion_iou,
                        args.max_det,
                        support_gain,
                    ),
                    names,
                )
                label = f"{tag} f{fusion_iou:g} sg{support_gain:g}"
                results.append({"config": label, **record})
                print(f"{label:<32} mAP50-95 {record['map50_95']:.5f}", flush=True)
            continue
        for support_gain in args.support_gains:
            for weight in args.second_weights:
                record = _evaluate_predictions(
                    _fused(
                        _sources(weight),
                        image_paths,
                        fusion_iou,
                        args.max_det,
                        support_gain,
                    ),
                    names,
                )
                label = _label(fusion_iou, weight, support_gain)
                results.append({"config": label, **record})
                print(f"{label:<32} mAP50-95 {record['map50_95']:.5f}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
