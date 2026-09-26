from pathlib import Path

import pytest

from scripts import eval_individual_flip_last_day as experiment


def test_candidates_do_not_accumulate_other_flip_source():
    control = [object() for _ in range(8)]
    original = tuple(control)
    extra = {832: object(), 1216: object()}
    first = experiment.individual_sources(control, extra, 832)
    second = experiment.individual_sources(control, extra, 1216)
    assert tuple(control) == original
    assert first[:-1] == second[:-1] == control
    assert len(first) == len(second) == 9
    assert first[-1] is extra[832] and extra[1216] not in first
    assert second[-1] is extra[1216] and extra[832] not in second


def test_missing_cache_never_initializes_model(tmp_path, monkeypatch):
    base = experiment.base
    monkeypatch.setattr(base, "WORK", tmp_path)
    monkeypatch.setattr(base, "require_hash", lambda *args: None)

    def forbidden(*args, **kwargs):
        pytest.fail("Cache-only assessment must not initialize a model")

    monkeypatch.setattr(base.ultralytics, "YOLO", forbidden)
    with pytest.raises(ValueError, match="contract mismatch or missing"):
        base.cached_validation_flip(832, [Path("1.npy")], {1: (10, 8)}, {},
                                    generate=False, model_holder=[])
