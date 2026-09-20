from pathlib import Path
import pickle

import numpy as np
import pytest

from scripts.eval_tiled_inference import _remap_full_cache, _remap_tile_cache
from scripts.predict_submission import PredictionArrays


def _record(path: Path) -> PredictionArrays:
    return PredictionArrays(
        path=path,
        boxes=np.asarray([[1, 2, 3, 4]], dtype=np.float32),
        classes=np.asarray([2], dtype=np.float32),
        confidences=np.asarray([0.5], dtype=np.float32),
        orig_shape=(10, 20),
    )


def test_remap_full_cache_uses_current_validation_paths(tmp_path: Path) -> None:
    old_path = Path("old/7.npy")
    cache_path = tmp_path / "full.pkl"
    with cache_path.open("wb") as handle:
        pickle.dump({832: {old_path: _record(old_path)}, 1024: {old_path: _record(old_path)}}, handle)
    current_path = tmp_path / "images" / "val" / "7.npy"

    remapped = _remap_full_cache(cache_path, [current_path])

    assert tuple(remapped) == (832, 1024)
    assert remapped[832][current_path].path == current_path
    assert remapped[832][current_path].boxes.tolist() == [[1.0, 2.0, 3.0, 4.0]]


def test_remap_tile_cache_rejects_missing_images(tmp_path: Path) -> None:
    old_path = Path("old/7.npy")
    with pytest.raises(ValueError, match="expected 2"):
        _remap_tile_cache(
            {old_path: [_record(old_path)]},
            [tmp_path / "7.npy", tmp_path / "8.npy"],
        )
