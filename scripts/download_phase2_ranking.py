from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
import threading
import time
from typing import Any
import zipfile


DEFAULT_COMPETITION = "hyperspectral-object-detection-challenge-2026"
RANKING_PREFIX = "data_ranking/data_ranking/VIS/"
SCHEMA_VERSION = 1
_RANKING_PATH = re.compile(
    r"^data_ranking/data_ranking/VIS/(?P<image_id>[0-9]+)\.png$"
)
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_THREAD_LOCAL = threading.local()
SIGNED_URL_MAX_ATTEMPTS = 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _json_text(value: object) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def collect_ranking_files(
    api: Any,
    competition: str,
    *,
    expected_count: int,
    page_size: int = 200,
) -> list[dict[str, int | str | None]]:
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    if not 1 <= page_size <= 200:
        raise ValueError("page_size must be in 1..200")

    page_token: str | None = None
    entries: list[dict[str, int | str | None]] = []
    seen_tokens: set[str] = set()
    while True:
        page = api.competition_list_files(
            competition,
            page_token=page_token,
            page_size=page_size,
        )
        for item in page.files or []:
            name = str(item.name)
            match = _RANKING_PATH.fullmatch(name)
            if match is None:
                continue
            total_bytes = int(item.total_bytes)
            if total_bytes <= 0:
                raise ValueError(f"Official ranking file is empty: {name}")
            entries.append(
                {
                    "image_id": int(match.group("image_id")),
                    "name": name,
                    "bytes": total_bytes,
                    "creation_date": _json_text(getattr(item, "creation_date", None)),
                }
            )

        next_token = getattr(page, "next_page_token", None)
        if not next_token:
            break
        next_token = str(next_token)
        if next_token in seen_tokens:
            raise RuntimeError("Kaggle file-list pagination repeated a page token")
        seen_tokens.add(next_token)
        page_token = next_token

    if len(entries) != expected_count:
        raise ValueError(
            f"Expected exactly {expected_count} official ranking PNGs, found {len(entries)}"
        )
    names = [str(item["name"]) for item in entries]
    image_ids = [int(item["image_id"]) for item in entries]
    basenames = [PurePosixPath(name).name for name in names]
    if len(set(names)) != len(names):
        raise ValueError("Official ranking file list has duplicate paths")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Official ranking file list has duplicate numeric image IDs")
    if len(set(name.casefold() for name in basenames)) != len(basenames):
        raise ValueError("Official ranking file list has colliding local basenames")
    return sorted(entries, key=lambda item: int(item["image_id"]))


def _inspect_png_header(path: Path) -> dict[str, int]:
    with path.open("rb") as handle:
        header = handle.read(29)
    if len(header) != 29 or header[:8] != _PNG_SIGNATURE:
        raise ValueError(f"Not a PNG file: {path}")
    if int.from_bytes(header[8:12], "big") != 13 or header[12:16] != b"IHDR":
        raise ValueError(f"PNG is missing the standard IHDR header: {path}")
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    bit_depth = header[24]
    color_type = header[25]
    if width <= 0 or height <= 0:
        raise ValueError(f"PNG has invalid dimensions: {path}")
    if bit_depth != 16 or color_type != 0:
        raise ValueError(
            f"Ranking PNG must be 16-bit grayscale; got depth={bit_depth}, "
            f"color_type={color_type}: {path}"
        )
    return {
        "width": width,
        "height": height,
        "bit_depth": bit_depth,
        "color_type": color_type,
    }


def _load_kaggle_api() -> Any:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as error:
        raise RuntimeError(
            "The Kaggle Python package is required. Run this script with the Python "
            "installation that provides the authenticated `kaggle` CLI."
        ) from error
    api = KaggleApi()
    api.authenticate()
    return api


def _thread_api() -> Any:
    api = getattr(_THREAD_LOCAL, "api", None)
    if api is None:
        api = _load_kaggle_api()
        _THREAD_LOCAL.api = api
    return api


def _rate_limit_delay(error: Exception, attempt: int) -> float | None:
    response = getattr(error, "response", None)
    if getattr(response, "status_code", None) != 429:
        return None
    headers = getattr(response, "headers", {}) or {}
    retry_after = headers.get("Retry-After")
    if retry_after is not None:
        try:
            return min(max(float(retry_after), 0.0), 300.0)
        except (TypeError, ValueError):
            pass
    return min(2 ** (attempt - 1) + random.random(), 60.0)


def _download_one(
    competition: str,
    entry: dict[str, int | str | None],
    output_dir: Path,
) -> str:
    destination = output_dir / PurePosixPath(str(entry["name"])).name
    expected_bytes = int(entry["bytes"])
    if destination.is_file() and destination.stat().st_size == expected_bytes:
        try:
            _inspect_png_header(destination)
            return "reused"
        except ValueError:
            force = True
    else:
        force = destination.exists() and destination.stat().st_size > expected_bytes

    for attempt in range(1, SIGNED_URL_MAX_ATTEMPTS + 1):
        try:
            _thread_api().competition_download_file(
                competition,
                str(entry["name"]),
                path=str(output_dir),
                force=force,
                quiet=True,
            )
            break
        except Exception as error:
            delay = _rate_limit_delay(error, attempt)
            if delay is None or attempt == SIGNED_URL_MAX_ATTEMPTS:
                raise
            print(
                f"DOWNLOAD_RATE_LIMIT: {entry['name']} attempt "
                f"{attempt}/{SIGNED_URL_MAX_ATTEMPTS}; retrying in {delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)
    if not destination.is_file():
        raise RuntimeError(f"Kaggle download did not create {destination}")
    actual_bytes = destination.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"Downloaded size mismatch for {destination.name}: "
            f"expected {expected_bytes}, got {actual_bytes}"
        )
    _inspect_png_header(destination)
    return "downloaded"


def download_ranking_files(
    competition: str,
    entries: list[dict[str, int | str | None]],
    output_dir: Path,
    *,
    workers: int,
) -> dict[str, int]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {"downloaded": 0, "reused": 0}
    with ThreadPoolExecutor(max_workers=min(workers, len(entries))) as executor:
        futures = {
            executor.submit(_download_one, competition, entry, output_dir): entry
            for entry in entries
        }
        complete = 0
        for future in as_completed(futures):
            entry = futures[future]
            try:
                disposition = future.result()
            except Exception as error:
                raise RuntimeError(
                    f"Failed to download official ranking file {entry['name']}"
                ) from error
            counts[disposition] += 1
            complete += 1
            if complete % 25 == 0 or complete == len(entries):
                print(f"DOWNLOAD_PROGRESS: {complete}/{len(entries)}")
    return counts


def verify_downloaded_files(
    entries: list[dict[str, int | str | None]],
    output_dir: Path,
) -> list[dict[str, int | str | None]]:
    expected_names = {
        PurePosixPath(str(entry["name"])).name.casefold() for entry in entries
    }
    actual_names = {path.name.casefold() for path in output_dir.glob("*.png")}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(
            f"Downloaded PNG set differs from official list; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    verified: list[dict[str, int | str | None]] = []
    for entry in entries:
        path = output_dir / PurePosixPath(str(entry["name"])).name
        expected_bytes = int(entry["bytes"])
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"Downloaded size differs from official metadata: {path}")
        image = _inspect_png_header(path)
        verified.append(
            {
                **entry,
                "local_path": str(path.resolve()),
                "sha256": _sha256(path),
                **image,
            }
        )
    return verified


def build_ranking_zip(
    entries: list[dict[str, int | str | None]],
    output_dir: Path,
    zip_path: Path,
) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = zip_path.with_suffix(zip_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as archive:
            for entry in entries:
                local = output_dir / PurePosixPath(str(entry["name"])).name
                archive.write(local, arcname=str(entry["name"]))
        os.replace(temporary, zip_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Selectively download and audit the 1000 inference-only Phase 2 "
            "ranking PNGs without downloading the full competition archive"
        )
    )
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--zip", dest="zip_path", type=Path)
    parser.add_argument("--expected-count", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")

    api = _load_kaggle_api()
    entries = collect_ranking_files(
        api,
        args.competition,
        expected_count=args.expected_count,
    )
    listed_at = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "competition": args.competition,
        "ranking_prefix": RANKING_PREFIX,
        "inference_only": True,
        "listed_at_utc": listed_at,
        "expected_count": args.expected_count,
        "official_total_bytes": sum(int(entry["bytes"]) for entry in entries),
        "output_dir": str(args.output_dir.resolve()),
        "status": "listed",
        "files": entries,
    }
    _write_json_atomic(args.manifest, manifest)
    counts = download_ranking_files(
        args.competition,
        entries,
        args.output_dir,
        workers=args.workers,
    )
    verified = verify_downloaded_files(entries, args.output_dir)
    manifest.update(
        {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "downloaded_this_run": counts["downloaded"],
            "reused_existing": counts["reused"],
            "verified_count": len(verified),
            "files": verified,
        }
    )
    if args.zip_path is not None:
        build_ranking_zip(entries, args.output_dir, args.zip_path)
        manifest["ranking_zip"] = {
            "path": str(args.zip_path.resolve()),
            "bytes": args.zip_path.stat().st_size,
            "sha256": _sha256(args.zip_path),
            "compression": "stored",
        }
    _write_json_atomic(args.manifest, manifest)
    print(
        f"RANKING_DOWNLOAD_OK: {len(verified)} files, "
        f"{manifest['official_total_bytes']} bytes"
    )


if __name__ == "__main__":
    main()
