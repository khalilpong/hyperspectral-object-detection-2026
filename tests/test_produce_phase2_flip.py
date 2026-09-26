from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import produce_phase2_flip as production


def record(path: Path, x: float = 1.0) -> production.PredictionArrays:
    return production.PredictionArrays(
        path, np.array([[x, 1, 8, 7]], dtype=np.float32),
        np.array([2], dtype=np.float32), np.array([0.7], dtype=np.float32), (8, 10))


def test_checkpoint_guard_rejects_another_checkpoint(tmp_path):
    weights = tmp_path / "last.pt"
    weights.write_bytes(b"another trained checkpoint")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        production.require_hash(weights, production.CHECKPOINT_SHA)


def test_cached_source_resumes_without_predicting_and_rejects_changed_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(production, "WORK", tmp_path / "work")
    real_require_hash = production.require_hash
    monkeypatch.setattr(production, "require_hash", lambda *args: production.CHECKPOINT_SHA)
    monkeypatch.setattr(production, "state_hash", lambda model: "fixed-model-state")
    paths = [tmp_path / "1.npy", tmp_path / "2.npy"]
    sizes = {1: (10, 8), 2: (10, 8)}
    calls = []

    def predict(model, images, **kwargs):
        calls.extend(images)
        assert kwargs["channels"] == 16
        assert kwargs["predict_kwargs"]["imgsz"] == 1024
        return {p: record(p) for p in images}

    monkeypatch.setattr(production, "_collect_horizontal_flip_predictions", predict)
    holder = [SimpleNamespace()]
    inventory = {"sha256": "input-v1"}
    first, first_manifest = production.cached_source(
        "test", "hflip1024", paths, sizes, inventory, generate=True, model_holder=holder)
    second, second_manifest = production.cached_source(
        "test", "hflip1024", paths, sizes, inventory, generate=True, model_holder=[])
    assert calls == paths  # second run has no model and cannot run inference
    assert first_manifest == second_manifest
    for path in paths:
        np.testing.assert_array_equal(first[path].boxes, second[path].boxes)
    with pytest.raises(FileExistsError, match="different artifact"):
        production.cached_source("test", "hflip1024", paths, sizes,
                                 {"sha256": "changed-input"}, generate=True, model_holder=[])
    # Valid geometry alone must not allow a changed persisted prediction.
    changed_cache = tmp_path / "work/cache/test/hflip1024/1.npz"
    item = record(paths[0], x=1.1)
    np.savez(changed_cache, boxes=item.boxes, classes=item.classes,
             confidences=item.confidences, orig_shape=item.orig_shape)
    monkeypatch.setattr(production, "require_hash", real_require_hash)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        production.cached_source("test", "hflip1024", paths, sizes,
                                 inventory, generate=False, model_holder=[])
    assert calls == paths


def test_cached_source_build_refuses_missing_predictions(tmp_path, monkeypatch):
    monkeypatch.setattr(production, "WORK", tmp_path)
    with pytest.raises(ValueError, match="Cache contract"):
        production.cached_source("ranking", "832", [], {}, {}, generate=False, model_holder=[])


def test_fusion_output_is_deterministic_and_manifests_cover_written_file(tmp_path):
    path = tmp_path / "1.npy"
    sources = [{path: record(path, 1 + index * 0.01)} for index in range(8)]
    first = production.prediction_frame(sources, [path], production.CANDIDATES["A"])
    second = production.prediction_frame(sources, [path], production.CANDIDATES["A"])
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    output = tmp_path / "output.csv"
    manifest = production.write_frame(output, first, {1: (10, 8)})
    assert manifest["rows"] == 1
    assert manifest["images"] == 1
    assert manifest["validation_issues"] == []
    assert manifest["output_sha256"] == production._sha256(output)
    assert manifest == production.write_frame(output, second, {1: (10, 8)})
    with pytest.raises(ValueError, match="coverage"):
        production.write_frame(tmp_path / "missing.csv", first, {1: (10, 8), 2: (10, 8)})


def test_baseline_control_rejects_changed_geometry(tmp_path):
    path = tmp_path / "1.npy"
    frame = production.prediction_frame([{path: record(path)}] * 7, [path], 0.74)
    baseline = tmp_path / "baseline.csv"
    frame.to_csv(baseline, index=False)
    sha = production._sha256(baseline)
    assert production.baseline_control(frame, baseline, sha)["passed"]
    frame.loc[0, "x1"] += 0.001
    with pytest.raises(ValueError, match="reproduction failed"):
        production.baseline_control(frame, baseline, sha)


def test_record_validation_rejects_noninteger_class_and_invalid_geometry(tmp_path):
    item = record(tmp_path / "1.npy")
    item.classes[0] = 1.5
    with pytest.raises(ValueError, match="class IDs"):
        production.validate_record(item, (8, 10))
    item.classes[0] = 1
    item.boxes[0, 2] = 11
    with pytest.raises(ValueError, match="geometry"):
        production.validate_record(item, (8, 10))


def test_raw_boundary_zero_area_boxes_are_filtered_by_existing_fusion(tmp_path):
    path = tmp_path / "1.npy"
    raw = record(path)
    raw.boxes[0, 1] = raw.boxes[0, 3] = 8
    production.validate_record(raw, (8, 10))
    output = production.prediction_frame([{path: raw}, {path: record(path)}], [path], 0.65)
    assert len(output) == 1
    assert output.loc[0, "y1"] == 1
    assert output.loc[0, "y2"] == 7
