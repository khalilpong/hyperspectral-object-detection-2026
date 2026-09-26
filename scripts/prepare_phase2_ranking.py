from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

import numpy as np
from PIL import Image

from hsi_detection.spectral import make_multispectral_uint8, x2cube


DEFAULT_BAND_ORDER = (5, 8, 13, 0, 1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15)
METHOD = "phase2_ranking_compact_4x4_unpack_shared_scale_v1"
SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _parse_band_order(text: str) -> tuple[int, ...]:
    try:
        bands = tuple(int(value.strip()) for value in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "band order must be 16 comma-separated integers"
        ) from error
    if len(bands) != 16 or set(bands) != set(range(16)):
        raise argparse.ArgumentTypeError(
            "band order must be a permutation of the physical bands 0..15"
        )
    return bands


def _safe_member_path(info: zipfile.ZipInfo) -> PurePosixPath:
    raw = info.filename.replace("\\", "/")
    if "\x00" in raw or raw.startswith("/"):
        raise ValueError(f"Unsafe ZIP member path: {info.filename!r}")
    parts = raw.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Unsafe ZIP member path: {info.filename!r}")
    if re.match(r"^[A-Za-z]:", parts[0]):
        raise ValueError(f"Unsafe ZIP member drive path: {info.filename!r}")
    mode = (info.external_attr >> 16) & 0xFFFF
    if mode and stat.S_ISLNK(mode):
        raise ValueError(f"ZIP member is a symbolic link: {info.filename!r}")
    if info.flag_bits & 0x1:
        raise ValueError(f"ZIP member is encrypted: {info.filename!r}")
    return PurePosixPath(*parts)


def inspect_ranking_zip(
    zip_path: Path,
    *,
    expected_count: int,
    max_uncompressed_bytes: int,
) -> tuple[list[dict[str, int | str]], str, int]:
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    if max_uncompressed_bytes <= 0:
        raise ValueError("max_uncompressed_bytes must be positive")

    entries: list[dict[str, int | str]] = []
    seen_paths: set[str] = set()
    seen_ids: set[int] = set()
    total_uncompressed = 0
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            member_path = _safe_member_path(info)
            normalized = member_path.as_posix()
            folded = normalized.casefold()
            if folded in seen_paths:
                raise ValueError(f"Duplicate ZIP member path: {normalized}")
            seen_paths.add(folded)
            if member_path.suffix.casefold() != ".png":
                raise ValueError(
                    f"Ranking ZIP must contain PNG images only; found {normalized}"
                )
            if not member_path.stem.isdigit():
                raise ValueError(f"Ranking image stem must be numeric: {normalized}")
            image_id = int(member_path.stem)
            if image_id in seen_ids:
                raise ValueError(f"Duplicate ranking image_id {image_id}")
            seen_ids.add(image_id)
            if info.file_size <= 0:
                raise ValueError(f"Ranking image is empty: {normalized}")
            total_uncompressed += info.file_size
            if total_uncompressed > max_uncompressed_bytes:
                raise ValueError(
                    "Ranking ZIP exceeds the configured uncompressed-size safety limit"
                )
            entries.append(
                {
                    "image_id": image_id,
                    "member": normalized,
                    "bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                    "crc32": f"{info.CRC:08X}",
                }
            )

    if len(entries) != expected_count:
        raise ValueError(
            f"Expected exactly {expected_count} ranking PNGs, found {len(entries)}"
        )
    entries.sort(key=lambda item: int(item["image_id"]))
    inventory_bytes = "\n".join(
        f"{item['image_id']}\t{item['member']}\t{item['bytes']}\t"
        f"{item['compressed_bytes']}\t{item['crc32']}"
        for item in entries
    ).encode("utf-8")
    inventory_sha256 = hashlib.sha256(inventory_bytes).hexdigest().upper()
    return entries, inventory_sha256, total_uncompressed


def _ensure_contract(output: Path, contract: dict[str, object]) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "ranking_preparation_config.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != contract:
            raise ValueError(
                "Output already has a different ranking preparation contract; "
                "use a new output directory"
            )
        return path
    image_root = output / "images"
    if image_root.exists() and any(image_root.iterdir()):
        raise ValueError(
            "Output contains images but no ranking_preparation_config.json; "
            "refusing an unsafe resume"
        )
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def _encode_member(
    archive: zipfile.ZipFile,
    entry: dict[str, int | str],
    image_root: Path,
    *,
    band_order: tuple[int, ...],
    lower_percentile: float,
    upper_percentile: float,
) -> dict[str, int | float | bool | str]:
    image_id = int(entry["image_id"])
    destination_png = image_root / f"{image_id}.png"
    destination_npy = image_root / f"{image_id}.npy"
    if destination_png.exists() and destination_npy.exists():
        try:
            existing = np.load(destination_npy, mmap_mode="r", allow_pickle=False)
            with Image.open(destination_png) as preview:
                preview_size = preview.size
            if (
                existing.ndim == 3
                and existing.shape[2] == len(band_order)
                and existing.dtype == np.uint8
                and preview_size == (existing.shape[1], existing.shape[0])
            ):
                return {
                    "image_id": image_id,
                    "member": str(entry["member"]),
                    "height": int(existing.shape[0]),
                    "width": int(existing.shape[1]),
                    "skipped": True,
                    "low": 0.0,
                    "high": 0.0,
                }
        except (OSError, ValueError):
            pass

    with archive.open(str(entry["member"])) as handle:
        payload = handle.read()
    with Image.open(BytesIO(payload)) as image:
        mosaic = np.asarray(image).copy()
    if mosaic.ndim != 2 or not np.issubdtype(mosaic.dtype, np.integer):
        raise ValueError(
            f"Ranking image {entry['member']} must be a single-channel integer PNG"
        )
    if np.iinfo(mosaic.dtype).bits < 16:
        raise ValueError(f"Ranking image {entry['member']} is not a 16-bit PNG")
    cube = x2cube(mosaic)
    selected = cube[:, :, band_order]
    low, high = np.percentile(selected, [lower_percentile, upper_percentile])
    encoded = make_multispectral_uint8(
        cube,
        band_order=band_order,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )
    image_root.mkdir(parents=True, exist_ok=True)
    temporary_npy = destination_npy.with_suffix(".npy.tmp")
    temporary_png = destination_png.with_suffix(".png.tmp")
    with temporary_npy.open("wb") as handle:
        np.save(handle, encoded, allow_pickle=False)
    Image.fromarray(encoded[:, :, :3], mode="RGB").save(
        temporary_png, format="PNG", compress_level=3
    )
    os.replace(temporary_npy, destination_npy)
    os.replace(temporary_png, destination_png)
    return {
        "image_id": image_id,
        "member": str(entry["member"]),
        "height": int(encoded.shape[0]),
        "width": int(encoded.shape[1]),
        "skipped": False,
        "low": float(low),
        "high": float(high),
    }


def _encode_chunk(
    zip_path: Path,
    entries: list[dict[str, int | str]],
    image_root: Path,
    *,
    band_order: tuple[int, ...],
    lower_percentile: float,
    upper_percentile: float,
) -> list[dict[str, int | float | bool | str]]:
    with zipfile.ZipFile(zip_path) as archive:
        return [
            _encode_member(
                archive,
                entry,
                image_root,
                band_order=band_order,
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
            )
            for entry in entries
        ]


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p50": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "p50": float(np.median(array)),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safely prepare the inference-only Phase 2 ranking ZIP as HSI16 NPY files"
    )
    parser.add_argument("--zip", dest="zip_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lower-percentile", type=float, default=0.5)
    parser.add_argument("--upper-percentile", type=float, default=99.5)
    parser.add_argument(
        "--band-order",
        type=_parse_band_order,
        default=DEFAULT_BAND_ORDER,
    )
    parser.add_argument("--max-uncompressed-gb", type=float, default=16.0)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if not 0 <= args.lower_percentile < args.upper_percentile <= 100:
        parser.error("percentiles must satisfy 0 <= lower < upper <= 100")
    if not args.zip_path.is_file():
        parser.error(f"ranking ZIP does not exist: {args.zip_path}")

    zip_path = args.zip_path.resolve()
    output = args.output.resolve()
    zip_sha256 = _sha256(zip_path)
    entries, inventory_sha256, total_uncompressed = inspect_ranking_zip(
        zip_path,
        expected_count=args.expected_count,
        max_uncompressed_bytes=int(args.max_uncompressed_gb * 1024**3),
    )
    contract: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "inference_only": True,
        "source_zip": str(zip_path),
        "source_zip_bytes": zip_path.stat().st_size,
        "source_zip_sha256": zip_sha256,
        "member_inventory_sha256": inventory_sha256,
        "member_count": len(entries),
        "uncompressed_bytes": total_uncompressed,
        "cell_size": 4,
        "mosaic_band_order": "row-major",
        "output_band_order": list(args.band_order),
        "channels": len(args.band_order),
        "encoding_dtype": "uint8",
        "encoding_scale": "per_image_shared_selected_band_bounds",
        "lower_percentile": args.lower_percentile,
        "upper_percentile": args.upper_percentile,
    }
    contract_path = _ensure_contract(output, contract)
    worker_count = min(args.workers, len(entries))
    chunks = [entries[index::worker_count] for index in range(worker_count)]
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        chunk_results = list(
            executor.map(
                lambda chunk: _encode_chunk(
                    zip_path,
                    chunk,
                    output / "images",
                    band_order=args.band_order,
                    lower_percentile=args.lower_percentile,
                    upper_percentile=args.upper_percentile,
                ),
                chunks,
            )
        )
    results = sorted(
        [item for chunk in chunk_results for item in chunk],
        key=lambda item: int(item["image_id"]),
    )
    expected_ids = {int(entry["image_id"]) for entry in entries}
    npy_ids = {int(path.stem) for path in (output / "images").glob("*.npy")}
    png_ids = {int(path.stem) for path in (output / "images").glob("*.png")}
    if npy_ids != expected_ids or png_ids != expected_ids:
        raise RuntimeError("Prepared ranking outputs do not exactly match the ZIP image IDs")

    generated = [item for item in results if not item["skipped"]]
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "preparation_config_sha256": _sha256(contract_path),
        "output": str(output / "images"),
        "images": {
            "total": len(results),
            "generated": len(generated),
            "resumed_existing": len(results) - len(generated),
            "id_min": min(expected_ids),
            "id_max": max(expected_ids),
            "unique_ids": len(expected_ids),
        },
        "encoding": {
            "channels": len(args.band_order),
            "band_order": list(args.band_order),
            "lower_percentile": args.lower_percentile,
            "upper_percentile": args.upper_percentile,
            "shared_scale_across_channels": True,
            "dtype": "uint8",
            "per_image_low": _summary([float(item["low"]) for item in generated]),
            "per_image_high": _summary([float(item["high"]) for item in generated]),
        },
    }
    report_path = output / "ranking_preparation_report.json"
    temporary_report = report_path.with_suffix(".json.tmp")
    temporary_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_report, report_path)
    print(json.dumps(report, indent=2))
    print(f"RANKING_PREPARATION_OK: {len(results)} inference-only images")


if __name__ == "__main__":
    main()
