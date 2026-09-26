from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
from PIL import Image
import pytest

from scripts import download_phase2_ranking


def _png_bytes(offset: int = 0) -> bytes:
    image = np.arange(64, dtype=np.uint16).reshape(8, 8) + offset
    payload = BytesIO()
    Image.fromarray(image).save(payload, format="PNG")
    return payload.getvalue()


class _FakeApi:
    def __init__(self, pages: dict[str | None, SimpleNamespace]) -> None:
        self.pages = pages
        self.calls: list[tuple[str, str | None, int]] = []

    def competition_list_files(
        self, competition: str, *, page_token: str | None, page_size: int
    ) -> SimpleNamespace:
        self.calls.append((competition, page_token, page_size))
        return self.pages[page_token]


def _file(name: str, size: int) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        total_bytes=size,
        creation_date="2026-09-25T09:07:07Z",
    )


def test_collect_ranking_files_paginates_and_filters() -> None:
    prefix = download_phase2_ranking.RANKING_PREFIX
    api = _FakeApi(
        {
            None: SimpleNamespace(
                files=[_file("class.txt", 10), _file(f"{prefix}101.png", 20)],
                next_page_token="next",
            ),
            "next": SimpleNamespace(
                files=[_file(f"{prefix}100.png", 19)],
                next_page_token=None,
            ),
        }
    )

    files = download_phase2_ranking.collect_ranking_files(
        api, "competition", expected_count=2
    )

    assert [item["image_id"] for item in files] == [100, 101]
    assert [item["bytes"] for item in files] == [19, 20]
    assert api.calls == [
        ("competition", None, 200),
        ("competition", "next", 200),
    ]


def test_collect_ranking_files_rejects_duplicate_numeric_ids() -> None:
    prefix = download_phase2_ranking.RANKING_PREFIX
    api = _FakeApi(
        {
            None: SimpleNamespace(
                files=[_file(f"{prefix}100.png", 20), _file(f"{prefix}0100.png", 20)],
                next_page_token=None,
            )
        }
    )

    with pytest.raises(ValueError, match="duplicate numeric image IDs"):
        download_phase2_ranking.collect_ranking_files(
            api, "competition", expected_count=2
        )


def test_verify_and_zip_exact_16bit_grayscale_files(tmp_path: Path) -> None:
    prefix = download_phase2_ranking.RANKING_PREFIX
    output = tmp_path / "download"
    output.mkdir()
    payloads = {"100.png": _png_bytes(), "101.png": _png_bytes(100)}
    entries = []
    for image_id, (name, payload) in enumerate(payloads.items(), start=100):
        (output / name).write_bytes(payload)
        entries.append(
            {
                "image_id": image_id,
                "name": f"{prefix}{name}",
                "bytes": len(payload),
                "creation_date": "2026-09-25T09:07:07Z",
            }
        )

    verified = download_phase2_ranking.verify_downloaded_files(entries, output)
    archive_path = tmp_path / "ranking.zip"
    download_phase2_ranking.build_ranking_zip(entries, output, archive_path)

    assert [item["bit_depth"] for item in verified] == [16, 16]
    assert all(len(str(item["sha256"])) == 64 for item in verified)
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.namelist() == [str(entry["name"]) for entry in entries]


def test_verify_rejects_extra_png(tmp_path: Path) -> None:
    prefix = download_phase2_ranking.RANKING_PREFIX
    output = tmp_path / "download"
    output.mkdir()
    payload = _png_bytes()
    (output / "100.png").write_bytes(payload)
    (output / "999.png").write_bytes(payload)
    entries = [
        {
            "image_id": 100,
            "name": f"{prefix}100.png",
            "bytes": len(payload),
            "creation_date": None,
        }
    ]

    with pytest.raises(ValueError, match="differs from official list"):
        download_phase2_ranking.verify_downloaded_files(entries, output)


def test_download_one_retries_rate_limited_signed_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = download_phase2_ranking.RANKING_PREFIX
    payload = _png_bytes()
    output = tmp_path / "download"
    output.mkdir()

    class RateLimitError(RuntimeError):
        response = SimpleNamespace(status_code=429, headers={"Retry-After": "0"})

    class FakeDownloadApi:
        def __init__(self) -> None:
            self.calls = 0

        def competition_download_file(
            self,
            competition: str,
            file_name: str,
            *,
            path: str,
            force: bool,
            quiet: bool,
        ) -> None:
            self.calls += 1
            if self.calls == 1:
                raise RateLimitError("429 Too Many Requests")
            (Path(path) / Path(file_name).name).write_bytes(payload)

    api = FakeDownloadApi()
    monkeypatch.setattr(download_phase2_ranking, "_thread_api", lambda: api)
    entry = {
        "image_id": 100,
        "name": f"{prefix}100.png",
        "bytes": len(payload),
        "creation_date": None,
    }

    disposition = download_phase2_ranking._download_one(
        "competition", entry, output
    )

    assert disposition == "downloaded"
    assert api.calls == 2
