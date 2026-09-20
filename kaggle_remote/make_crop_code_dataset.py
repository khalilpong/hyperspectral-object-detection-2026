"""Stage the minimal private Kaggle dataset needed by crop/tile variants."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUTPUT = HERE / "crop_code_dataset"
DATASET_ID = "zephyrpong/hsi-object-crop-code"
SOURCE_FILES = {
    "hsi_detection.tiling.py": PROJECT_ROOT / "src" / "hsi_detection" / "tiling.py",
    "scripts.prepare_object_crops.py": PROJECT_ROOT / "scripts" / "prepare_object_crops.py",
    "scripts.predict_submission.py": PROJECT_ROOT / "scripts" / "predict_submission.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def stage(output: Path = OUTPUT) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    expected = {*SOURCE_FILES, "dataset-metadata.json", "payload-manifest.json"}
    unexpected = sorted(path.name for path in output.iterdir() if path.name not in expected)
    if unexpected:
        raise RuntimeError(f"增量代码目录含未预期文件，拒绝打包：{unexpected}")

    manifest = {"dataset_id": DATASET_ID, "files": {}}
    for target_name, source in SOURCE_FILES.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        target = output / target_name
        shutil.copy2(source, target)
        manifest["files"][target_name] = {
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
            "source": source.relative_to(PROJECT_ROOT).as_posix(),
        }

    metadata = {
        "title": "hsi-object-crop-code",
        "id": DATASET_ID,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (output / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (output / "payload-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    result = stage()
    print(json.dumps(result, indent=2))
