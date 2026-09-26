"""Produce only the two preregistered Phase 2 flip candidates from one checkpoint.

The legacy test cache is hash-pinned and must reproduce the frozen baseline.
Missing ranking scales are cached once. Every new prediction is saved per image,
so a resumed run never repeats a successfully persisted model prediction.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import ultralytics

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibrate_submission_boxes import calibrate_submission_frame, load_audited_calibration
from merge_phase2_submission import _image_sizes, _sha256, _validated_submission
from predict_submission import (
    PredictionArrays, _checkpoint_channels, _collect_horizontal_flip_predictions,
    _fused_prediction_sources, _numeric_paths, _prediction_arrays,
)
from hsi_detection.box_calibration import BoxCalibration
from hsi_detection.submission import SUBMISSION_COLUMNS, clip_xyxy, validate_submission_frame


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "kaggle_remote/outputs/full/kaggle_full_yolo26m_e30/last.pt"
CHECKPOINT_SHA = "8E4BFBF7AC36BAA55274C464EF3A8BC6581C1450601AB4DFB97D3D8FAA4254C3"
LEGACY_CACHE = ROOT / "artifacts/ensemble_cache/m_full.pkl"
LEGACY_SHA = "9BEDAB0E8500497BE883007A0EDA34DF9B7E7FD9F4B9FEB8A99B69E3409BB373"
AUDIT = ROOT / "artifacts/ensemble/single_m_box_calibration_oof_20260922.json"
AUDIT_SHA = "056C8DD8A9274B3A4F49B8D5A9160BC4B42B021EA54EAD09A44E40F6B87844E2"
EVIDENCE = ROOT / "artifacts/inference_mode_validation/m_ablation_flip_only_20260921.json"
EVIDENCE_SHA = "49FA5146E4C1392F71C2D9BFE2B3609F742D3F85B2AAC29D9A855D7A4650F904"
BASELINE = ROOT / "submissions/submission_phase2_single_m_hsi16_ms7_f074_sg0125_boxscale101.csv"
BASELINE_SHA = "1744A3545E89F0E695EE106B177C582C7158D686D675940CFCAD3A72956C5E23"
TEST_BASELINE = ROOT / "submissions/submission_single_m_hsi16_ms7_f074_sg0125_boxscale101.csv"
TEST_BASELINE_SHA = "C01214E6B250075C7DA522F2E9E6F89C829BF94AE1EB2F57CF69618FDC995A86"
RANKING_BASELINE = ROOT / "submissions/phase2_ranking_single_m_hsi16_ms7_f074_sg0125_raw.csv"
RANKING_BASELINE_SHA = "6DC2D55513BAFF1FDD94EC613742721DA2345217DDB0347F79278919CBE1FB67"
IMAGES = {
    "test": ROOT / "data/processed/hsi16_shared_p005_995/images/test",
    "ranking": ROOT / "data/processed/phase2_ranking/images",
}
SCALES = (832, 896, 960, 1024, 1088, 1152, 1216)
CANDIDATES = {"A": 0.65, "B": 0.82}
WORK = ROOT / "artifacts/phase2_flip_20260926"
SETTINGS = {"batch": 1, "device": "0", "quantize": 16, "conf": 0.0001,
            "iou": 0.7, "max_det": 300, "verbose": False, "augment": False}


def command() -> str:
    return subprocess.list2cmdline([sys.executable, *sys.argv])


def require_hash(path: Path, expected: str) -> str:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch at {path}: {actual} != {expected}")
    return actual


def write_new(path: Path, payload: bytes) -> None:
    """Never silently replace an existing artifact or a partial write."""
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"Refusing to replace different artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
    os.replace(temporary, path)


def write_json(path: Path, payload: object) -> None:
    write_new(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def validate_record(record: PredictionArrays, shape: tuple[int, int]) -> None:
    boxes, classes, scores = record.boxes, record.classes, record.confidences
    if tuple(record.orig_shape) != shape or boxes.shape != (len(classes), 4):
        raise ValueError("Cache shape mismatch")
    if classes.shape != scores.shape or classes.ndim != 1 or len(boxes) > 300:
        raise ValueError("Cache array lengths or max_det mismatch")
    if not all(np.isfinite(a).all() for a in (boxes, classes, scores)):
        raise ValueError("Cache contains non-finite predictions")
    if not ((classes == np.floor(classes)) & (classes >= 0) & (classes < 18)).all():
        raise ValueError("Cache contains invalid class IDs")
    if not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError("Cache contains invalid confidence")
    h, w = shape
    if not ((boxes[:, 0] >= 0) & (boxes[:, 1] >= 0) & (boxes[:, 2] <= w)
            & (boxes[:, 3] <= h) & (boxes[:, 0] <= boxes[:, 2])
            & (boxes[:, 1] <= boxes[:, 3])).all():
        raise ValueError("Cache contains invalid geometry")
    # Raw model output can contain boundary-clipped zero-area boxes. Preserve
    # them here: the existing _box_vote filters them before clustering. Final
    # submission validation continues to require strictly positive box area.


def input_inventory(split: str) -> tuple[list[Path], dict, dict]:
    paths = _numeric_paths(IMAGES[split], ".npy")
    sizes = _image_sizes(IMAGES[split], expected_count=1000, label=split)
    if len(paths) != 1000 or {int(p.stem) for p in paths} != set(sizes):
        raise ValueError("NPY and PNG coverage mismatch")
    files = []
    for path in paths:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        w, h = sizes[int(path.stem)]
        if array.shape != (h, w, 16) or array.dtype != np.uint8:
            raise ValueError(f"Invalid 16-channel uint8 input: {path}")
        files.append({"path": str(path.resolve()), "bytes": path.stat().st_size,
                      "sha256": _sha256(path), "shape": list(array.shape)})
    inventory = {"split": split, "images": len(paths), "files": files}
    write_json(WORK / f"{split}_inputs.json", inventory)
    return paths, sizes, {"path": str(WORK / f"{split}_inputs.json"),
                          "sha256": _sha256(WORK / f"{split}_inputs.json")}


def remap_legacy(paths: list[Path], sizes: dict) -> dict:
    require_hash(LEGACY_CACHE, LEGACY_SHA)
    # This exact local, trusted pickle is pinned before deserialization.
    with LEGACY_CACHE.open("rb") as handle:
        stored = pickle.load(handle)
    if tuple(stored) != SCALES:
        raise ValueError("Legacy scale order mismatch")
    result = {}
    for scale, records in stored.items():
        by_stem = {Path(p).stem: r for p, r in records.items()}
        if len(by_stem) != len(records) or set(by_stem) != {p.stem for p in paths}:
            raise ValueError("Legacy cache coverage mismatch")
        result[scale] = {}
        for path in paths:
            old = by_stem[path.stem]
            record = PredictionArrays(path, old.boxes, old.classes, old.confidences,
                                      old.orig_shape)
            w, h = sizes[int(path.stem)]
            validate_record(record, (h, w))
            result[scale][path] = record
    return result


def save_record(path: Path, record: PredictionArrays) -> None:
    buffer = io.BytesIO()
    np.savez(buffer, boxes=record.boxes, classes=record.classes,
             confidences=record.confidences, orig_shape=np.array(record.orig_shape))
    write_new(path, buffer.getvalue())


def load_record(path: Path, image: Path, sizes: dict) -> PredictionArrays:
    with np.load(path, allow_pickle=False) as data:
        record = PredictionArrays(image, data["boxes"], data["classes"],
                                  data["confidences"], tuple(data["orig_shape"].tolist()))
    w, h = sizes[int(image.stem)]
    validate_record(record, (h, w))
    return record


def state_hash(model: object) -> str:
    module = model.predictor.model.model
    if any(m.training for m in module.modules()):
        raise RuntimeError("Inference model entered training mode")
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest().upper()


def cached_source(split: str, source: str, paths: list[Path], sizes: dict,
                  inventory: dict, *, generate: bool, model_holder: list) -> tuple[dict, dict]:
    flip = source == "hflip1024"
    scale = 1024 if flip else int(source)
    if (split == "test" and not flip) or (not flip and scale not in SCALES):
        raise ValueError("Only missing ranking scales or the fixed flip may be generated")
    directory = WORK / "cache" / split / source
    contract = {"checkpoint_sha256": CHECKPOINT_SHA, "input_inventory": inventory,
                "settings": {**SETTINGS, "imgsz": scale}, "channels": 16,
                "transform": "horizontal_flip_all_16_channels" if flip else "identity",
                "inverse_box": "x1=width-x2,x2=width-x1" if flip else "identity",
                "torch": torch.__version__, "ultralytics": ultralytics.__version__}
    contract_path = directory / "contract.json"
    if generate:
        write_json(contract_path, contract)
    if not contract_path.is_file() or json.loads(contract_path.read_text()) != contract:
        raise ValueError(f"Cache contract mismatch or missing: {directory}")
    manifest_path = directory / "manifest.json"
    existing_hashes = None
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        existing_hashes = {item["image_id"]: item["sha256"] for item in existing["records"]}
        if existing["contract"] != contract or set(existing_hashes) != {int(p.stem) for p in paths}:
            raise ValueError("Existing cache manifest contract/coverage mismatch")
    records, generated, items = {}, 0, []
    initial_state = None
    for index, image in enumerate(paths, 1):
        cache_path = directory / f"{image.stem}.npz"
        if existing_hashes is not None:
            require_hash(cache_path, existing_hashes[int(image.stem)])
        if not cache_path.exists():
            if not generate:
                raise FileNotFoundError(cache_path)
            if not model_holder:
                require_hash(WEIGHTS, CHECKPOINT_SHA)
                model = ultralytics.YOLO(str(WEIGHTS))
                if _checkpoint_channels(model) != 16:
                    raise ValueError("Production checkpoint must have 16 channels")
                model.model.eval()
                model_holder.append(model)
            model = model_holder[0]
            with torch.inference_mode():
                if flip:
                    record = _collect_horizontal_flip_predictions(
                        model, [image], channels=16, batch_size=1,
                        predict_kwargs={**SETTINGS, "imgsz": scale})[image]
                else:
                    record = next(_prediction_arrays(
                        model, [image], image_directory=IMAGES[split], input_format="npy",
                        channels=16, batch_size=1,
                        predict_kwargs={**SETTINGS, "imgsz": scale}))
            w, h = sizes[int(image.stem)]
            validate_record(record, (h, w))
            save_record(cache_path, record)
            generated += 1
            if initial_state is None:
                initial_state = state_hash(model)
        records[image] = load_record(cache_path, image, sizes)
        boxes = records[image].boxes
        items.append({"image_id": int(image.stem), "sha256": _sha256(cache_path),
                      "boxes": len(boxes), "zero_area_boxes": int(np.count_nonzero(
                          (boxes[:, 0] == boxes[:, 2]) | (boxes[:, 1] == boxes[:, 3])))})
        if index % 100 == 0:
            print(f"{split}/{source}: {index}/{len(paths)} cached, {generated} new", flush=True)
    if initial_state is not None and state_hash(model_holder[0]) != initial_state:
        raise RuntimeError("Model state changed during inference")
    require_hash(WEIGHTS, CHECKPOINT_SHA)
    payload = {"contract": contract, "records": items}
    write_json(directory / "manifest.json", payload)
    if generated:
        write_json(directory / "generation.json", {
            "command": command(), "generated_this_run": generated,
            "state_sha256_after_first_prediction": initial_state,
            "state_unchanged": True, "model_training": False})
    generation_path = directory / "generation.json"
    generation = json.loads(generation_path.read_text())
    return records, {"path": str(directory / "manifest.json"),
                     "sha256": _sha256(directory / "manifest.json"),
                     "generation": {"path": str(generation_path),
                                    "sha256": _sha256(generation_path),
                                    "command": generation["command"]}}


def prediction_frame(sources: list[dict], paths: list[Path], iou: float) -> pd.DataFrame:
    rows = []
    for record in _fused_prediction_sources(sources, paths, fusion_iou=iou,
                                            support_gain=0.125, max_det=300):
        h, w = record.orig_shape
        for box, cls, score in zip(record.boxes, record.classes, record.confidences, strict=True):
            clipped = clip_xyxy(box.tolist(), width=w, height=h)
            if clipped is None:
                raise ValueError("Fusion produced invalid geometry")
            rows.append([len(rows), int(record.path.stem), int(cls), float(score), *clipped])
    return pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)


def write_frame(path: Path, frame: pd.DataFrame, sizes: dict) -> dict:
    issues = validate_submission_frame(frame, sizes, 18)
    if issues or set(frame.image_id) != set(sizes):
        raise ValueError(f"Invalid submission or coverage: {issues[:10]}")
    for name in ("id", "image_id", "class_id"):
        if not np.equal(frame[name], np.floor(frame[name])).all():
            raise ValueError(f"Noninteger {name}")
    write_new(path, frame.to_csv(index=False).encode())
    reread, ids = _validated_submission(path, sizes, class_count=18, label=path.name)
    return {"path": str(path), "output_sha256": _sha256(path), "bytes": path.stat().st_size,
            "rows": len(reread), "images": len(ids), "columns": SUBMISSION_COLUMNS,
            "id_is_consecutive": True, "validation_issues": []}


def baseline_control(frame: pd.DataFrame, baseline: Path, expected_sha: str) -> dict:
    require_hash(baseline, expected_sha)
    reference = pd.read_csv(baseline)
    # CSV float parsing may differ in the last binary digit. No detection/ordering
    # change is accepted, and tolerance is far below a float32 coordinate ULP.
    if frame.shape != reference.shape or not np.allclose(
            frame.to_numpy(), reference.to_numpy(), rtol=0, atol=1e-10):
        raise ValueError(f"Baseline reproduction failed: {baseline.name}")
    return {"baseline": str(baseline), "sha256": expected_sha, "rows": len(frame),
            "maximum_absolute_difference": float(np.max(np.abs(frame.to_numpy() - reference.to_numpy()))),
            "passed": True, "tolerance": 1e-10}


def verify_fixed_evidence() -> dict:
    for path, digest in ((WEIGHTS, CHECKPOINT_SHA), (AUDIT, AUDIT_SHA),
                         (EVIDENCE, EVIDENCE_SHA), (BASELINE, BASELINE_SHA)):
        require_hash(path, digest)
    calibration, _ = load_audited_calibration(AUDIT)
    if calibration != BoxCalibration(width_scale=1.01, height_scale=1.01):
        raise ValueError("Only audited global boxscale101 is allowed")
    old = json.loads(EVIDENCE.read_text())
    if old["control"]["actual_full_map"] != 0.705708182597841 or not old["control"]["passed"]:
        raise ValueError("Fixed-validation cache control mismatch")
    selected = {}
    for candidate, iou in CANDIDATES.items():
        item = old["full_plus_flip"][f"iou_{iou:.3f}_support_0.125"]
        expected = 0.7058779597408886 if candidate == "A" else 0.705812158799068
        if item["map50_95"] != expected:
            raise ValueError("Fixed-validation candidate mismatch")
        selected[candidate] = {key: item[key] for key in
                               ("map50_95", "gain_vs_supported_full", "predictions")}
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cache", "build"))
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), default="A")
    args = parser.parse_args()
    evidence = verify_fixed_evidence()
    calibration, _ = load_audited_calibration(AUDIT)
    model_holder, splits = [], {}
    for split in ("test", "ranking"):
        paths, sizes, inventory = input_inventory(split)
        manifests = []
        if split == "test":
            full = remap_legacy(paths, sizes)
            manifests.append({"path": str(LEGACY_CACHE), "sha256": LEGACY_SHA})
        else:
            full = {}
            for scale in SCALES:
                full[scale], manifest = cached_source(
                    split, str(scale), paths, sizes, inventory,
                    generate=args.mode == "cache", model_holder=model_holder)
                manifests.append(manifest)
        sources = [full[scale] for scale in SCALES]
        control_frame = prediction_frame(sources, paths, 0.74)
        if split == "test":
            control_frame = calibrate_submission_frame(control_frame, sizes, calibration)
            control = baseline_control(control_frame, TEST_BASELINE, TEST_BASELINE_SHA)
        else:
            control = baseline_control(control_frame, RANKING_BASELINE, RANKING_BASELINE_SHA)
        write_json(WORK / f"{split}_baseline_control.json", control)
        print(f"{split}: baseline reproduction PASS ({control['rows']} rows)", flush=True)
        flip, manifest = cached_source(split, "hflip1024", paths, sizes, inventory,
                                       generate=args.mode == "cache", model_holder=model_holder)
        manifests.append(manifest)
        splits[split] = (paths, sizes, inventory, [*sources, flip], manifests, control)
    if set(splits["test"][1]) & set(splits["ranking"][1]):
        raise ValueError("test/ranking overlap")
    if args.mode == "cache":
        print("CACHE_COMPLETE: test and ranking sources persisted; no candidate was submitted", flush=True)
        return
    candidate = args.candidate
    slug = f"phase2_single_m_hsi16_ms7_hflip1024_f{int(CANDIDATES[candidate]*100):03d}_sg0125_boxscale101"
    manifest = {"candidate": candidate, "command": command(), "checkpoint": str(WEIGHTS),
                "checkpoint_sha256": CHECKPOINT_SHA, "single_checkpoint": True,
                "inference_only": True, "training_or_adaptation": False,
                "scales": list(SCALES), "flip_imgsz": 1024,
                "fusion_iou": CANDIDATES[candidate], "support_gain": 0.125,
                "max_det": 300, "calibration": asdict(calibration),
                "state_check_scope": "Model is set to eval before inference; runtime state hashes compare after the first forward of each generated source with its final forward, after Ultralytics setup/fusion.",
                "calibration_audit_sha256": AUDIT_SHA,
                "fixed_validation": evidence[candidate], "evidence_sha256": EVIDENCE_SHA,
                "test_ranking_overlap": 0, "splits": {}}
    manifest["source_code_sha256"] = {
        name: _sha256(ROOT / name) for name in (
            "scripts/produce_phase2_flip.py", "scripts/predict_submission.py",
            "scripts/calibrate_submission_boxes.py", "scripts/merge_phase2_submission.py",
            "src/hsi_detection/box_calibration.py", "src/hsi_detection/submission.py")}
    frames, all_sizes = [], {}
    for split, (paths, sizes, inventory, sources, caches, control) in splits.items():
        raw = prediction_frame(sources, paths, CANDIDATES[candidate])
        raw_info = write_frame(ROOT / f"submissions/{slug}_{split}_raw.csv", raw, sizes)
        # Match the audited CLI pipeline's CSV round trip before calibration.
        calibrated = calibrate_submission_frame(pd.read_csv(raw_info["path"]), sizes, calibration)
        calibrated_info = write_frame(ROOT / f"submissions/{slug}_{split}_calibrated.csv", calibrated, sizes)
        split_manifest = {"checkpoint_sha256": CHECKPOINT_SHA, "command": command(),
                          "input_inventory": inventory, "caches": caches,
                          "baseline_control": control, "raw": raw_info,
                          "calibrated": calibrated_info, "output_sha256": calibrated_info["output_sha256"]}
        write_json(WORK / f"{candidate}_{split}_manifest.json", split_manifest)
        manifest["splits"][split] = split_manifest
        frames.append(pd.read_csv(calibrated_info["path"]))
        all_sizes.update(sizes)
    combined = pd.concat(frames, ignore_index=True)
    combined["id"] = np.arange(len(combined), dtype=np.int64)
    manifest["combined"] = write_frame(ROOT / f"submissions/submission_{slug}.csv", combined, all_sizes)
    write_json(WORK / f"{candidate}_manifest.json", manifest)
    print(json.dumps(manifest["combined"], indent=2), flush=True)


if __name__ == "__main__":
    main()
