from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest


PROJECT_ROOT = Path(__file__).parents[1]
PREPARE_SCRIPT = PROJECT_ROOT / "scripts" / "prepare_phase_aware_multispectral.py"


def _load_prepare_module():
    spec = importlib.util.spec_from_file_location("prepare_phase_aware_under_test", PREPARE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase_writer_creates_valid_npy_preview_and_resumes(tmp_path: Path) -> None:
    module = _load_prepare_module()
    source = tmp_path / "source.png"
    rows, columns = np.indices((8, 12), dtype=np.uint16)
    mosaic = rows * np.uint16(100) + columns
    Image.fromarray(mosaic).save(source)
    destination = tmp_path / "dataset" / "images" / "train" / "1.png"

    first = module._write_phase_aware_image(
        source,
        destination,
        lower_percentile=0,
        upper_percentile=100,
        target_long_edge=12,
    )

    encoded = np.load(destination.with_suffix(".npy"), allow_pickle=False)
    assert first["skipped"] is False
    assert encoded.shape == (8, 12, 16)
    assert encoded.dtype == np.uint8
    with Image.open(destination) as preview:
        assert preview.mode == "RGB"
        assert preview.size == (12, 8)
    assert first["bytes"] == destination.with_suffix(".npy").stat().st_size

    second = module._write_phase_aware_image(
        source,
        destination,
        lower_percentile=0,
        upper_percentile=100,
        target_long_edge=12,
    )
    assert second["skipped"] is True
    assert second["low"] is None
    assert second["high"] is None


def test_phase_writer_rebuilds_an_invalid_existing_array(tmp_path: Path) -> None:
    module = _load_prepare_module()
    source = tmp_path / "source.png"
    Image.fromarray(np.arange(8 * 12, dtype=np.uint16).reshape(8, 12)).save(source)
    destination = tmp_path / "dataset" / "images" / "val" / "2.png"
    destination.parent.mkdir(parents=True)
    np.save(destination.with_suffix(".npy"), np.zeros((1, 1, 16), dtype=np.uint8))
    Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8)).save(destination)

    result = module._write_phase_aware_image(
        source,
        destination,
        lower_percentile=0.5,
        upper_percentile=99.5,
        target_long_edge=12,
    )

    assert result["skipped"] is False
    assert np.load(destination.with_suffix(".npy"), allow_pickle=False).shape == (8, 12, 16)


def test_preparation_contract_prevents_mixed_encodings(tmp_path: Path) -> None:
    module = _load_prepare_module()
    output = tmp_path / "dataset"
    contract = module._contract(
        target_long_edge=1024,
        lower_percentile=0.5,
        upper_percentile=99.5,
        limit=0,
    )

    module._ensure_contract(output, contract)
    module._ensure_contract(output, contract)
    assert json.loads((output / "preparation_config.json").read_text(encoding="utf-8")) == contract

    changed = dict(contract, target_long_edge=960)
    with pytest.raises(ValueError, match="different phase-aware preparation contract"):
        module._ensure_contract(output, changed)


def test_manifest_contract_rejects_duplicate_ids(tmp_path: Path) -> None:
    module = _load_prepare_module()
    manifest = tmp_path / "split_manifest.csv"
    manifest.write_text("image_id,split\n1,train\n1,val\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate image_id"):
        module._load_manifest(manifest)
