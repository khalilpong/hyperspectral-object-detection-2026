from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import eval_multiscale_flip_last_day as experiment
from scripts.predict_submission import PredictionArrays


@pytest.mark.parametrize("changed", [None, "weights_sha256", "full_cache_sha256", "flip_cache_sha256"])
def test_validation_lineage_rejects_mixed_checkpoint_or_cache(changed):
    evidence = {
        "weights_sha256": experiment.CHECKPOINT_SHA.lower(),
        "full_cache_sha256": experiment.PINNED[experiment.FULL_CACHE].lower(),
        "flip_cache_sha256": experiment.PINNED[experiment.FLIP_CACHE].lower(),
    }
    if changed is None:
        experiment.require_validation_lineage(evidence)
    else:
        evidence[changed] = "0" * 64
        with pytest.raises(ValueError, match=changed):
            experiment.require_validation_lineage(evidence)


def test_stability_gate_requires_gain_and_group_consistency():
    assert experiment.passes_stability_gate(0.0011, [0.001, 0.002, -0.0004])
    assert not experiment.passes_stability_gate(0.0009, [0.001] * 3)
    assert not experiment.passes_stability_gate(0.002, [0.005, -0.001, -0.001])
    assert not experiment.passes_stability_gate(0.002, [0.002, 0.002, -0.0006])
    assert not experiment.passes_stability_gate(float("nan"), [0.002] * 3)


def test_validation_inventory_rejects_production_count_and_unknown_scale():
    with pytest.raises(ValueError, match="600 validation"):
        experiment.validation_inventory([Path("images/test/1.npy")])
    with pytest.raises(ValueError, match="preregistered"):
        experiment.cached_validation_flip(1024, [], {}, {}, generate=True, model_holder=[])


def test_cached_flip_infers_once_and_rejects_modified_record(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "WORK", tmp_path)
    real_require_hash = experiment.require_hash
    monkeypatch.setattr(experiment, "require_hash", lambda *args: "fixed")
    monkeypatch.setattr(experiment, "state_hash", lambda model: "stable-state")
    paths = [tmp_path / "1.npy", tmp_path / "2.npy"]
    sizes = {1: (10, 8), 2: (10, 8)}
    calls = []

    def predict(model, images, **kwargs):
        calls.extend(images)
        assert kwargs["predict_kwargs"]["imgsz"] == 832
        assert kwargs["channels"] == 16
        return {path: PredictionArrays(path, np.array([[1, 1, 8, 7]], np.float32),
                                      np.array([2], np.float32), np.array([0.7], np.float32), (8, 10))
                for path in images}

    monkeypatch.setattr(experiment, "_collect_horizontal_flip_predictions", predict)
    first, meta = experiment.cached_validation_flip(832, paths, sizes, {"sha256": "input-v1"},
                                                     generate=True, model_holder=[SimpleNamespace()])
    second, reused = experiment.cached_validation_flip(832, paths, sizes, {"sha256": "input-v1"},
                                                       generate=False, model_holder=[])
    assert calls == paths
    assert meta["new_inference_images"] == 2 and reused["new_inference_images"] == 0
    for path in paths:
        np.testing.assert_array_equal(first[path].boxes, second[path].boxes)
    with pytest.raises(FileExistsError, match="different artifact"):
        experiment.cached_validation_flip(832, paths, sizes, {"sha256": "input-v2"},
                                          generate=True, model_holder=[])
    changed = tmp_path / "validation_flip_cache/832/1.npz"
    changed.write_bytes(b"modified prediction")
    monkeypatch.setattr(experiment, "require_hash", lambda path, expected:
                        "fixed" if path == experiment.WEIGHTS else real_require_hash(path, expected))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        experiment.cached_validation_flip(832, paths, sizes, {"sha256": "input-v1"},
                                          generate=False, model_holder=[])
    assert calls == paths
