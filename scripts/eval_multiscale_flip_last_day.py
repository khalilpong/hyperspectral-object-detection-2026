"""Evaluate the fixed 832+1024+1216 horizontal-flip hypothesis on validation only.

The old seven identity passes and 1024 flip are reused. Only two additional
600-image passes may be generated. This entry point cannot access ranking/test
inputs or produce a competition submission.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import ultralytics

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibrate_submission_boxes import load_audited_calibration
from compare_inference_modes import _dataset_details, _evaluate_predictions
from eval_box_calibration import _calibrate_predictions
from eval_flip_tta import _remap_flip_cache
from eval_phase2_cached_controls import (
    AUDIT, DATA, EVIDENCE, FLIP_CACHE, FULL_CACHE, PINNED, ROOT, SCALES,
    group_indices, require_control, sha256,
)
from eval_tiled_inference import _remap_full_cache
from predict_submission import (
    _checkpoint_channels, _collect_horizontal_flip_predictions,
    _fused_prediction_sources, _numeric_paths,
)
from produce_phase2_flip import (
    command, load_record, require_hash, save_record, state_hash,
    validate_record, write_json,
)


WEIGHTS = ROOT / "kaggle_remote/outputs/ablation/kaggle_ablation_yolo26m_e30/last.pt"
CHECKPOINT_SHA = "A4E7B10B93CADC241F6C6E10BEB9DFEE4E0887F96C265EC022323AE82C4C7657"
WORK = ROOT / "artifacts/phase2_last_day_20260927"
NEW_SCALES = (832, 1216)
SETTINGS = {"batch": 1, "device": "0", "quantize": 16, "conf": 0.0001,
            "iou": 0.7, "max_det": 300, "verbose": False, "augment": False}
EXPECTED_A_CALIBRATED = 0.7079261502266522


def require_validation_lineage(evidence: dict) -> None:
    expected = {
        "weights_sha256": CHECKPOINT_SHA,
        "full_cache_sha256": PINNED[FULL_CACHE],
        "flip_cache_sha256": PINNED[FLIP_CACHE],
    }
    for key, value in expected.items():
        if str(evidence.get(key, "")).upper() != value:
            raise ValueError(f"Frozen validation lineage mismatch: {key}")


def passes_stability_gate(gain: float, group_deltas: list[float]) -> bool:
    return bool(
        len(group_deltas) == 3
        and np.isfinite([gain, *group_deltas]).all()
        and gain >= 0.001
        and sum(value > 0 for value in group_deltas) >= 2
        and min(group_deltas) >= -0.0005
    )


def validation_inventory(paths: list[Path]) -> tuple[dict, dict]:
    if len(paths) != 600:
        raise ValueError("Only the original 600 validation images are allowed")
    entries, sizes = [], {}
    for path in paths:
        if path.parent.name != "val" or path.parent.parent.name != "images":
            raise ValueError("New inference must be restricted to validation")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 3 or array.shape[2] != 16 or array.dtype != np.uint8:
            raise ValueError(f"Invalid HSI16 uint8 input: {path}")
        h, w, _ = array.shape
        sizes[int(path.stem)] = (w, h)
        label = path.parents[2] / "labels/val" / f"{path.stem}.txt"
        entries.append({"image_id": int(path.stem), "path": str(path.resolve()),
                        "shape": list(array.shape), "sha256": sha256(path),
                        "label_sha256": sha256(label)})
    target = WORK / "validation_inputs.json"
    write_json(target, {"split": "fixed_600_validation_only", "images": 600,
                        "data_sha256": sha256(DATA), "files": entries})
    return sizes, {"path": str(target), "sha256": sha256(target)}


def cached_validation_flip(
    scale: int, paths: list[Path], sizes: dict, inventory: dict,
    *, generate: bool, model_holder: list,
) -> tuple[dict, dict]:
    if scale not in NEW_SCALES:
        raise ValueError("Only the preregistered new flip scales are allowed")
    require_hash(WEIGHTS, CHECKPOINT_SHA)
    directory = WORK / "validation_flip_cache" / str(scale)
    contract = {
        "schema_version": 1, "validation_only": True,
        "checkpoint_sha256": CHECKPOINT_SHA, "input_inventory": inventory,
        "settings": {**SETTINGS, "imgsz": scale}, "channels": 16,
        "transform": "horizontal_flip_all_16_channels",
        "inverse_box": "x1=width-x2,x2=width-x1",
        "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "predict_source_sha256": sha256(ROOT / "scripts/predict_submission.py"),
    }
    contract_path = directory / "contract.json"
    if generate:
        write_json(contract_path, contract)
    if not contract_path.is_file() or json.loads(contract_path.read_text()) != contract:
        raise ValueError("Validation flip cache contract mismatch or missing")
    contract_sha = sha256(contract_path)
    records, receipts, generated = {}, [], 0
    initial_state = None
    start = time.perf_counter()
    for index, path in enumerate(paths, 1):
        cache_path = directory / f"{path.stem}.npz"
        receipt_path = directory / f"{path.stem}.json"
        if cache_path.exists() or receipt_path.exists():
            if not cache_path.is_file() or not receipt_path.is_file():
                raise ValueError("Incomplete prediction/receipt pair; refusing to adopt it")
            receipt = json.loads(receipt_path.read_text())
            if receipt["image_id"] != int(path.stem) or receipt["contract_sha256"] != contract_sha:
                raise ValueError("Prediction receipt contract/image mismatch")
            require_hash(cache_path, receipt["sha256"])
        else:
            if not generate:
                raise FileNotFoundError(cache_path)
            if not model_holder:
                model = ultralytics.YOLO(str(WEIGHTS))
                if _checkpoint_channels(model) != 16:
                    raise ValueError("Validation checkpoint must have 16 channels")
                model.model.eval()
                model_holder.append(model)
            model = model_holder[0]
            with torch.inference_mode():
                record = _collect_horizontal_flip_predictions(
                    model, [path], channels=16, batch_size=1,
                    predict_kwargs={**SETTINGS, "imgsz": scale},
                )[path]
            w, h = sizes[int(path.stem)]
            validate_record(record, (h, w))
            save_record(cache_path, record)
            write_json(receipt_path, {"image_id": int(path.stem), "sha256": sha256(cache_path),
                                      "contract_sha256": contract_sha})
            generated += 1
            if initial_state is None:
                initial_state = state_hash(model)
        records[path] = load_record(cache_path, path, sizes)
        receipts.append({"image_id": int(path.stem), "sha256": sha256(cache_path),
                         "receipt_sha256": sha256(receipt_path),
                         "boxes": len(records[path].boxes)})
        if index % 100 == 0:
            print(f"validation/hflip{scale}: {index}/{len(paths)}, new={generated}", flush=True)
    if initial_state is not None and state_hash(model_holder[0]) != initial_state:
        raise RuntimeError("Model state changed during validation inference")
    require_hash(WEIGHTS, CHECKPOINT_SHA)
    manifest = directory / "manifest.json"
    write_json(manifest, {"contract": contract, "records": receipts})
    return records, {
        "manifest": str(manifest), "manifest_sha256": sha256(manifest),
        "new_inference_images": generated, "cached_images": len(paths) - generated,
        "elapsed_seconds": time.perf_counter() - start,
        "runtime_state_after_first_new_prediction": initial_state,
        "runtime_state_unchanged_after_first_new_prediction": initial_state is not None,
        "runtime_state_scope": "after first new prediction to last; model setup/fusion precedes first snapshot",
        "command": command(),
    }


def evaluate_groups(predictions: list, names: dict) -> list[dict]:
    return [_evaluate_predictions((predictions[index] for index in group), names)
            for group in group_indices(len(predictions))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true", help="Generate only the two missing validation passes")
    args = parser.parse_args()
    for path, expected in PINNED.items():
        require_hash(path, expected)
    require_hash(WEIGHTS, CHECKPOINT_SHA)
    require_validation_lineage(json.loads(EVIDENCE.read_text()))
    names, directory = _dataset_details(DATA)
    paths = _numeric_paths(directory, ".npy")
    sizes, inventory = validation_inventory(paths)
    identity = _remap_full_cache(FULL_CACHE, paths)
    if tuple(identity) != SCALES:
        raise ValueError("Identity scale order changed")
    with FLIP_CACHE.open("rb") as handle:
        flip1024 = _remap_flip_cache(pickle.load(handle), paths)
    calibration, _ = load_audited_calibration(AUDIT)
    sources = [*[identity[scale] for scale in SCALES], flip1024]
    raw_a = list(_fused_prediction_sources(sources, paths, fusion_iou=0.65, max_det=300, support_gain=0.125))
    raw_a_metrics = _evaluate_predictions(iter(raw_a), names)
    require_control(raw_a_metrics["map50_95"], 0.7058779597408886)
    calibrated_a = _calibrate_predictions(raw_a, calibration)
    a_metrics = _evaluate_predictions(iter(calibrated_a), names)
    require_control(a_metrics["map50_95"], EXPECTED_A_CALIBRATED)
    a_groups = evaluate_groups(calibrated_a, names)
    print(f"A_CONTROL_PASS calibrated_map={a_metrics['map50_95']}", flush=True)
    generations, model_holder = {}, []
    for scale in NEW_SCALES:
        records, generation = cached_validation_flip(
            scale, paths, sizes, inventory, generate=args.generate, model_holder=model_holder,
        )
        sources.append(records)
        generations[str(scale)] = generation
    raw = list(_fused_prediction_sources(sources, paths, fusion_iou=0.65, max_det=300, support_gain=0.125))
    calibrated = _calibrate_predictions(raw, calibration)
    metrics = _evaluate_predictions(iter(calibrated), names)
    groups = evaluate_groups(calibrated, names)
    gain = metrics["map50_95"] - a_metrics["map50_95"]
    deltas = [candidate["map50_95"] - reference["map50_95"]
              for candidate, reference in zip(groups, a_groups, strict=True)]
    report = {
        "schema_version": 1, "candidate": "A_plus_hflip832_hflip1216",
        "validation_only": True, "training_performed": False,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "frozen_cache_checkpoint_lineage_verified": True,
        "production_checkpoint_not_loaded": True,
        "sources": [*[f"identity{scale}" for scale in SCALES], "hflip1024", "hflip832", "hflip1216"],
        "fusion_iou": 0.65, "support_gain": 0.125, "max_det": 300, "boxscale": 1.01,
        "input_inventory": inventory, "generations": generations,
        "source_hashes": {str(path.relative_to(ROOT)): value for path, value in PINNED.items()},
        "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in [
            Path(__file__), ROOT / "scripts/predict_submission.py", ROOT / "scripts/compare_inference_modes.py",
            ROOT / "scripts/eval_box_calibration.py", ROOT / "src/hsi_detection/box_calibration.py"]},
        "control": {"raw_A": raw_a_metrics, "calibrated_A": a_metrics, "groups_A": a_groups},
        "metrics": metrics, "groups": groups, "gain_vs_A": gain, "group_deltas_vs_A": deltas,
        "gate": {"minimum_gain": 0.001, "minimum_positive_groups": 2, "worst_group_floor": -0.0005},
        "passes_gate": passes_stability_gate(gain, deltas),
        "limitation": "Previously tuned fixed validation set; group consistency is not an untouched holdout guarantee.",
        "command": command(),
    }
    output = WORK / ("multiscale_flip_validation.json" if args.generate else "multiscale_flip_recheck.json")
    write_json(output, report)
    print(json.dumps({"map": metrics["map50_95"], "gain_vs_A": gain,
                      "group_deltas": deltas, "passes_gate": report["passes_gate"],
                      "new_inference_images": sum(item["new_inference_images"] for item in generations.values()),
                      "report": str(output)}), flush=True)


if __name__ == "__main__":
    main()
