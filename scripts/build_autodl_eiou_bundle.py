"""Build a self-auditing upload bundle for the AutoDL EIoU fixed run."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import time
import zipfile
from pathlib import Path


PROFILE_NAME = "yolo26m_eiou_fixed"
EXCLUDED_CODE_DATASET_FILES = {
    "dataset-metadata.json",
    "yolo11m.pt",
    "yolo26s.pt",
    "yolo26m-regmax16.yaml",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def collect_sources(repo_root: Path) -> dict[str, Path]:
    staging = repo_root / "kaggle_remote" / "code_dataset"
    sources = {
        path.name: path
        for path in sorted(staging.iterdir())
        if path.is_file() and path.name not in EXCLUDED_CODE_DATASET_FILES
    }
    sources.update(
        {
            "run_autodl_eiou.py": repo_root / "scripts" / "run_autodl_eiou.py",
            "scripts.check_fixed_split_gate.py": repo_root
            / "scripts"
            / "check_fixed_split_gate.py",
            "requirements-autodl-eiou.txt": repo_root
            / "requirements-autodl-eiou.txt",
        }
    )
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"bundle sources are missing: {missing}")
    expected = {
        "run_hsi_yolo26.py": repo_root / "kaggle_remote" / "run_hsi_yolo26.py",
        "scripts.train_baseline.py": repo_root / "scripts" / "train_baseline.py",
        "scripts.prepare_multispectral.py": repo_root
        / "scripts"
        / "prepare_multispectral.py",
    }
    drift = {
        name: {"staging": sha256(sources[name]), "source": sha256(source)}
        for name, source in expected.items()
        if sha256(sources[name]) != sha256(source)
    }
    if drift:
        raise RuntimeError(f"flattened code dataset is not synchronized: {drift}")
    return sources


def build_bundle(repo_root: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing bundle: {output}")
    sources = collect_sources(repo_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="autodl-eiou-") as temporary:
        stage = Path(temporary)
        for name, source in sources.items():
            shutil.copy2(source, stage / name)
        files = [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(stage.iterdir())
            if path.is_file()
        ]
        manifest = {
            "schema_version": 1,
            "profile": PROFILE_NAME,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "files": files,
        }
        (stage / "bundle_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(stage.iterdir()):
                archive.write(path, arcname=path.name)
    return {
        "path": str(output.resolve()),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "file_count": len(files) + 1,
        "profile": PROFILE_NAME,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_bundle(args.repo_root.resolve(), args.output.resolve()),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
