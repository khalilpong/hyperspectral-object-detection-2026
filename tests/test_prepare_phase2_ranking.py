from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image
import pytest

from scripts import prepare_phase2_ranking


def _png_bytes(offset: int = 0) -> bytes:
    mosaic = np.arange(64, dtype=np.uint16).reshape(8, 8) + offset
    payload = BytesIO()
    Image.fromarray(mosaic).save(payload, format="PNG")
    return payload.getvalue()


def test_prepare_ranking_zip_writes_exact_hsi16_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = tmp_path / "ranking_images.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("ranking/VIS/100.png", _png_bytes())
        archive.writestr("ranking/VIS/101.png", _png_bytes(100))
    output = tmp_path / "prepared"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_phase2_ranking.py",
            "--zip",
            str(archive_path),
            "--output",
            str(output),
            "--expected-count",
            "2",
            "--workers",
            "1",
            "--lower-percentile",
            "0",
            "--upper-percentile",
            "100",
        ],
    )

    prepare_phase2_ranking.main()

    assert np.load(output / "images" / "100.npy").shape == (2, 2, 16)
    assert np.load(output / "images" / "100.npy").dtype == np.uint8
    with Image.open(output / "images" / "101.png") as preview:
        assert preview.size == (2, 2)
    report = json.loads(
        (output / "ranking_preparation_report.json").read_text(encoding="utf-8")
    )
    assert report["contract"]["inference_only"] is True
    assert report["contract"]["member_count"] == 2
    assert report["encoding"]["band_order"] == list(
        prepare_phase2_ranking.DEFAULT_BAND_ORDER
    )
    assert report["images"]["unique_ids"] == 2


@pytest.mark.parametrize(
    "members, message",
    [
        (["../100.png", "101.png"], "Unsafe ZIP member"),
        (["a/100.png", "b/0100.png"], "Duplicate ranking image_id"),
        (["100.png", "notes.txt"], "PNG images only"),
    ],
)
def test_inspect_ranking_zip_rejects_unsafe_or_ambiguous_members(
    tmp_path: Path, members: list[str], message: str
) -> None:
    archive_path = tmp_path / "ranking_images.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for member in members:
            archive.writestr(member, _png_bytes())

    with pytest.raises(ValueError, match=message):
        prepare_phase2_ranking.inspect_ranking_zip(
            archive_path,
            expected_count=2,
            max_uncompressed_bytes=1024 * 1024,
        )
