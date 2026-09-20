from pathlib import Path

import numpy as np
import pytest

from scripts.eval_flip_tta import _remap_flip_cache
from scripts.predict_submission import PredictionArrays


def _record(path: Path) -> PredictionArrays:
    return PredictionArrays(
        path=path,
        boxes=np.asarray([[1, 2, 3, 4]], dtype=np.float32),
        classes=np.asarray([2], dtype=np.float32),
        confidences=np.asarray([0.5], dtype=np.float32),
        orig_shape=(10, 20),
    )


def test_remap_flip_cache_uses_current_validation_paths(tmp_path: Path) -> None:
    old_path = Path("old/7.npy")
    current_path = tmp_path / "images" / "val" / "7.npy"

    remapped = _remap_flip_cache({old_path: _record(old_path)}, [current_path])

    assert remapped[current_path].path == current_path
    assert remapped[current_path].boxes.tolist() == [[1.0, 2.0, 3.0, 4.0]]


def test_remap_flip_cache_rejects_duplicate_stems(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate image stem"):
        _remap_flip_cache(
            {
                Path("old/7.npy"): _record(Path("old/7.npy")),
                Path("other/7.npy"): _record(Path("other/7.npy")),
            },
            [tmp_path / "7.npy"],
        )
