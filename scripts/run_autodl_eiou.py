"""Run the preregistered YOLO26m EIoU fixed experiment on a Linux GPU host.

This wrapper intentionally exposes paths and resource thresholds, not model
hyperparameters.  It validates an immutable code bundle, safely prepares the
competition archive, records the host environment, launches the existing
remote runner with the exact preregistered EIoU contract, applies the fixed
split gate, and creates a portable result archive.  It never starts a full-data
run and never submits to Kaggle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable


PROFILE_NAME = "yolo26m_eiou_fixed"
DEFAULT_RUN_NAME = "autodl_ablation_yolo26m_eiou_e30"
DEFAULT_BAND_ORDER = "5,8,13,0,1,2,3,4,6,7,9,10,11,12,14,15"
EXPECTED_YOLO26M_SHA256 = (
    "401CEA9AB23AD19246FF7744859816BC599F350E93C9DD30367B6F0A0745D0B7"
)
GATE = 0.70443
REQUIRED_BUNDLE_FILES = (
    "bundle_manifest.json",
    "run_hsi_yolo26.py",
    "scripts.check_fixed_split_gate.py",
    "scripts.train_baseline.py",
    "scripts.prepare_multispectral.py",
    "scripts.predict_submission.py",
    "scripts.check_submission.py",
    "hsi_detection.box_loss.py",
    "hsi_detection.layout.py",
    "hsi_detection.spectral.py",
    "split_manifest.csv",
    "yolo26m.pt",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def validate_bundle(code_root: Path) -> dict[str, object]:
    """Validate every file declared by the immutable AutoDL bundle manifest."""
    missing = [name for name in REQUIRED_BUNDLE_FILES if not (code_root / name).is_file()]
    if missing:
        raise RuntimeError(f"AutoDL bundle is missing required files: {missing}")

    manifest_path = code_root / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise RuntimeError("bundle_manifest.json has an unsupported schema")
    if manifest.get("profile") != PROFILE_NAME:
        raise RuntimeError(
            f"bundle profile mismatch: {manifest.get('profile')!r} != {PROFILE_NAME!r}"
        )
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("bundle manifest does not contain a non-empty file list")

    observed_paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError("bundle manifest file entry is not an object")
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative:
            raise RuntimeError("bundle manifest file entry has no path")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or "\\" in relative:
            raise RuntimeError(f"unsafe bundle manifest path: {relative!r}")
        path = code_root.joinpath(*pure.parts)
        if not path.is_file():
            raise RuntimeError(f"bundle file is missing: {relative}")
        actual_size = path.stat().st_size
        actual_sha = sha256(path)
        if entry.get("bytes") != actual_size or str(entry.get("sha256", "")).upper() != actual_sha:
            raise RuntimeError(
                f"bundle file drift: {relative} size={actual_size} sha256={actual_sha}"
            )
        if relative in observed_paths:
            raise RuntimeError(f"duplicate bundle manifest path: {relative}")
        observed_paths.add(relative)

    required_manifest_paths = set(REQUIRED_BUNDLE_FILES) - {"bundle_manifest.json"}
    missing_manifest_paths = sorted(required_manifest_paths - observed_paths)
    if missing_manifest_paths:
        raise RuntimeError(
            "bundle manifest does not cover required files: "
            f"{missing_manifest_paths}"
        )

    undeclared_files = sorted(
        path.name
        for path in code_root.iterdir()
        if path.is_file()
        and path.name != "bundle_manifest.json"
        and path.name not in observed_paths
    )
    if undeclared_files:
        raise RuntimeError(f"bundle contains undeclared files: {undeclared_files}")

    expected_weight = code_root / "yolo26m.pt"
    if sha256(expected_weight) != EXPECTED_YOLO26M_SHA256:
        raise RuntimeError("bundled yolo26m.pt does not match the preregistered checkpoint")
    return manifest


def _zip_member_path(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if (
        not normalized
        or normalized.startswith("/")
        or pure.is_absolute()
        or ".." in pure.parts
        or (pure.parts and ":" in pure.parts[0])
        or "\x00" in normalized
    ):
        raise RuntimeError(f"unsafe zip member path: {name!r}")
    return pure


def inspect_safe_zip(path: Path) -> dict[str, object]:
    """Reject traversal/symlink entries and describe a competition archive."""
    class_members: list[str] = []
    layout_members = {
        "annotations": False,
        "train_images": False,
        "test_images": False,
    }
    file_count = 0
    uncompressed_bytes = 0
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            pure = _zip_member_path(info.filename)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise RuntimeError(f"zip symlink entries are not allowed: {info.filename!r}")
            if info.is_dir():
                continue
            file_count += 1
            uncompressed_bytes += info.file_size
            if pure.name == "class.txt":
                class_members.append(info.filename)
            lowered = tuple(part.lower() for part in pure.parts)
            for key, tail in {
                "annotations": ("vis", "annotations"),
                "train_images": ("vis", "data_train"),
                "test_images": ("vis", "data_test"),
            }.items():
                if any(
                    lowered[index : index + 2] == tail
                    for index in range(max(0, len(lowered) - 1))
                ):
                    layout_members[key] = True
    if len(class_members) != 1:
        raise RuntimeError(
            f"competition zip must contain exactly one class.txt, found {class_members}"
        )
    missing_layout = [key for key, present in layout_members.items() if not present]
    if missing_layout:
        raise RuntimeError(
            f"competition zip is missing required VIS layout members: {missing_layout}"
        )
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "files": file_count,
        "uncompressed_bytes": uncompressed_bytes,
        "class_member": class_members[0],
        "layout_members": layout_members,
    }


def _find_unique(root: Path, predicate, label: str) -> Path:
    candidates = sorted(path for path in root.rglob("*") if predicate(path))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one {label} below {root}, found {[str(p) for p in candidates]}"
        )
    return candidates[0]


def audit_competition_layout(root: Path) -> dict[str, str]:
    """Validate the four raw-data anchors required by hsi_detection.layout."""
    class_file = _find_unique(root, lambda path: path.is_file() and path.name == "class.txt", "class.txt")

    def has_tail(path: Path, *tail: str) -> bool:
        parts = tuple(part.lower() for part in path.parts)
        lowered = tuple(part.lower() for part in tail)
        return len(parts) >= len(lowered) and parts[-len(lowered) :] == lowered

    annotations = _find_unique(
        root,
        lambda path: path.is_dir() and has_tail(path, "VIS", "Annotations"),
        "VIS/Annotations directory",
    )
    train_images = _find_unique(
        root,
        lambda path: path.is_dir() and has_tail(path, "VIS", "data_train"),
        "VIS/data_train directory",
    )
    test_images = _find_unique(
        root,
        lambda path: path.is_dir() and has_tail(path, "VIS", "data_test"),
        "VIS/data_test directory",
    )
    return {
        "root": str(class_file.parent.resolve()),
        "class_file": str(class_file.resolve()),
        "annotations": str(annotations.resolve()),
        "train_images": str(train_images.resolve()),
        "test_images": str(test_images.resolve()),
    }


def locate_extracted_competition_root(input_root: Path) -> Path | None:
    class_files = sorted(input_root.rglob("class.txt")) if input_root.exists() else []
    if not class_files:
        return None
    if len(class_files) != 1:
        raise RuntimeError(
            f"input root contains multiple class.txt files: {[str(path) for path in class_files]}"
        )
    return class_files[0].parent


def choose_competition_source(
    input_root: Path, work_root: Path, *, extract: bool
) -> tuple[Path | None, dict[str, object]]:
    """Use an extracted tree or validate and safely extract exactly one raw zip."""
    extracted_root = locate_extracted_competition_root(input_root)
    if extracted_root is not None:
        layout = audit_competition_layout(extracted_root)
        return extracted_root, {"kind": "directory", "layout": layout}

    archives = []
    for path in sorted(input_root.rglob("*.zip")):
        try:
            audit = inspect_safe_zip(path)
        except (RuntimeError, zipfile.BadZipFile):
            continue
        archives.append((path, audit))
    if len(archives) != 1:
        raise RuntimeError(
            "input root must contain one extracted competition tree or exactly one safe "
            f"competition zip; found {[str(path) for path, _ in archives]}"
        )
    archive_path, archive_audit = archives[0]
    if not extract:
        return None, {"kind": "zip", "archive": archive_audit, "extracted": False}

    destination = work_root / "competition_raw" / str(archive_audit["sha256"])[:12]
    marker = destination / ".autodl_raw_source.json"
    if destination.exists():
        if not marker.is_file():
            raise RuntimeError(
                f"refusing to reuse non-empty unverified extraction directory: {destination}"
            )
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        if recorded.get("sha256") != archive_audit["sha256"]:
            raise RuntimeError("existing extraction marker does not match the raw archive")
    else:
        destination.mkdir(parents=True, exist_ok=False)
        root_resolved = destination.resolve()
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                pure = _zip_member_path(info.filename)
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise RuntimeError(f"zip symlink entries are not allowed: {info.filename!r}")
                target = destination.joinpath(*pure.parts)
                resolved = target.resolve()
                if not resolved.is_relative_to(root_resolved):
                    raise RuntimeError(f"zip extraction escaped destination: {info.filename!r}")
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
        _write_json(marker, archive_audit)

    extracted_root = locate_extracted_competition_root(destination)
    if extracted_root is None:
        raise RuntimeError("safe extraction completed but class.txt is still missing")
    layout = audit_competition_layout(extracted_root)
    return extracted_root, {
        "kind": "zip",
        "archive": archive_audit,
        "extracted": True,
        "layout": layout,
    }


def exact_profile_environment(
    base: dict[str, str],
    *,
    code_root: Path,
    input_root: Path,
    competition_root: Path,
    work_root: Path,
    output_dir: Path,
    run_name: str,
) -> dict[str, str]:
    """Remove ambient HSI overrides and set the complete preregistered contract."""
    env = {key: value for key, value in base.items() if not key.startswith("HSI_")}
    env.update(
        {
            "HSI_MODE": "ablation",
            "HSI_ARCHITECTURE": "yolo",
            "HSI_MODEL": "yolo26m.pt",
            "HSI_MODEL_SOURCE_KIND": "checkpoint_native",
            "HSI_MODEL_YAML": "",
            "HSI_REG_MAX": "1",
            "HSI_EPOCHS": "30",
            "HSI_RUN_NAME": run_name,
            "HSI_ATTEMPTS": "8:2,6:2,4:2,4:0",
            "HSI_MULTISCALE": "0",
            "HSI_DATA": "hsi16",
            "HSI_BAND_ORDER": DEFAULT_BAND_ORDER,
            "HSI_SEED": "2026",
            "HSI_EXTRA_CHANNEL_INIT": "random",
            "HSI_SPECTRAL_STEM": "0",
            "HSI_LOWER_PERCENTILE": "0.5",
            "HSI_UPPER_PERCENTILE": "99.5",
            "HSI_CLS_PW": "0.0",
            "HSI_SCALE": "0.5",
            "HSI_DEGREES": "0",
            "HSI_DFL": "1.5",
            "HSI_BOX_IOU_LOSS": "eiou",
            "HSI_OPTIMIZER_RECIPE": "auto",
            "HSI_RTDETR_NUM_DENOISING": "100",
            "HSI_FUSION_IOU": "0.70",
            "HSI_SUPPORT_GAIN": "0.0",
            "HSI_PHASE_TARGET_LONG_EDGE": "1024",
            "HSI_OBJECT_CROPS": "0",
            "HSI_TILE_INFERENCE": "0",
            "HSI_CODE_ROOT": str(code_root.resolve()),
            "HSI_INPUT_ROOT": str(input_root.resolve()),
            "HSI_COMP_ROOT": str(competition_root.resolve()),
            "HSI_WORK_ROOT": str(work_root.resolve()),
            "HSI_OUT_DIR": str(output_dir.resolve()),
        }
    )
    return env


def _run_text(command: list[str], *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def collect_preflight(
    code_root: Path,
    input_root: Path,
    work_root: Path,
    output_root: Path,
    *,
    min_ram_gb: float,
    min_free_gb: float,
) -> dict[str, object]:
    """Collect and enforce the paid-host prerequisites before training starts."""
    for path in (input_root, work_root, output_root):
        path.mkdir(parents=True, exist_ok=True)
    manifest = validate_bundle(code_root)

    query = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if query.returncode != 0 or not query.stdout.strip():
        raise RuntimeError("nvidia-smi did not report an available NVIDIA GPU")
    gpu_rows = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    if len(gpu_rows) != 1:
        raise RuntimeError(f"this profile requires exactly one visible GPU, found {gpu_rows}")
    parts = [part.strip() for part in gpu_rows[0].split(",")]
    if len(parts) < 3:
        raise RuntimeError(f"unexpected nvidia-smi query output: {gpu_rows[0]!r}")
    gpu_memory_mib = int(parts[-2])
    if gpu_memory_mib < 22_000:
        raise RuntimeError(
            f"EIoU fixed profile requires a 24GB-class GPU; reported {gpu_memory_mib} MiB"
        )

    try:
        import psutil
        import torch
    except ImportError as error:
        raise RuntimeError(
            "preflight requires psutil and a CUDA-enabled torch installation"
        ) from error
    ram = psutil.virtual_memory()
    ram_gb = ram.total / 2**30
    if ram_gb < min_ram_gb:
        raise RuntimeError(
            f"host RAM {ram_gb:.1f} GiB is below the required {min_ram_gb:.1f} GiB"
        )
    free_gb = shutil.disk_usage(work_root).free / 2**30
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"work disk free space {free_gb:.1f} GiB is below {min_free_gb:.1f} GiB"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "PyTorch must see exactly one CUDA device before the paid run starts"
        )

    return {
        "schema_version": 1,
        "profile": PROFILE_NAME,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "cpus": os.cpu_count(),
            "ram_total_gib": round(ram_gb, 3),
            "ram_available_gib": round(ram.available / 2**30, 3),
            "work_free_gib": round(free_gb, 3),
        },
        "gpu": {
            "nvidia_smi_query": gpu_rows,
            "name": torch.cuda.get_device_name(0),
            "memory_mib": gpu_memory_mib,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
        "bundle_manifest_sha256": sha256(code_root / "bundle_manifest.json"),
        "bundle_created_utc": manifest.get("created_utc"),
        "thresholds": {"min_ram_gib": min_ram_gb, "min_work_free_gib": min_free_gb},
    }


def gate_command(code_root: Path, output_dir: Path, run_name: str) -> list[str]:
    keep = output_dir / run_name
    data_contract = output_dir / "data_contract"
    return [
        sys.executable,
        str(code_root / "scripts.check_fixed_split_gate.py"),
        "--status",
        str(output_dir / "status.json"),
        "--results",
        str(keep / "results.csv"),
        "--architecture",
        "yolo",
        "--model",
        "yolo26m.pt",
        "--expected-model-source-kind",
        "checkpoint_native",
        "--expected-pretrained-weights-sha256",
        EXPECTED_YOLO26M_SHA256,
        "--expected-reg-max",
        "1",
        "--expected-dfl",
        "1.5",
        "--expected-degrees",
        "0",
        "--expected-box-iou-loss",
        "eiou",
        "--expected-optimizer-recipe",
        "auto",
        "--optimizer-contract",
        str(keep / "optimizer_contract.json"),
        "--model-contract",
        str(keep / "model_contract.json"),
        "--expected-band-order",
        DEFAULT_BAND_ORDER,
        "--preparation-config",
        str(data_contract / "preparation_config.json"),
        "--preparation-report",
        str(data_contract / "preparation_report.json"),
        "--dataset-yaml",
        str(data_contract / "dataset.yaml"),
        "--split-manifest",
        str(data_contract / "split_manifest.csv"),
        "--gate",
        f"{GATE:.5f}",
        "--output",
        str(output_dir / "fixed_gate.json"),
    ]


def _run_tee(command: list[str], log_path: Path, *, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def result_manifest(output_dir: Path) -> dict[str, object]:
    files = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "result_manifest.json":
            files.append(
                {
                    "path": path.relative_to(output_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    return {
        "schema_version": 1,
        "profile": PROFILE_NAME,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": files,
    }


def archive_results(output_dir: Path, archive_path: Path) -> dict[str, object]:
    if archive_path.exists():
        raise RuntimeError(f"refusing to overwrite existing result archive: {archive_path}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(output_dir, arcname=output_dir.name, recursive=True)
    return {
        "path": str(archive_path.resolve()),
        "bytes": archive_path.stat().st_size,
        "sha256": sha256(archive_path),
    }


def _safe_run_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", value):
        raise argparse.ArgumentTypeError("run name must be 1-96 safe filename characters")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", type=_safe_run_name, default=DEFAULT_RUN_NAME)
    parser.add_argument("--min-ram-gb", type=float, default=48.0)
    parser.add_argument("--min-free-gb", type=float, default=45.0)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate hardware, bundle, and raw source without extracting or training",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code_root = args.code_root.resolve()
    input_root = args.input_root.resolve()
    work_root = args.work_root.resolve()
    output_root = args.output_root.resolve()
    output_dir = (
        output_root / "_preflight" / args.run_name
        if args.preflight_only
        else output_root / args.run_name
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output directory: {output_dir}")

    preflight = collect_preflight(
        code_root,
        input_root,
        work_root,
        output_root,
        min_ram_gb=args.min_ram_gb,
        min_free_gb=args.min_free_gb,
    )
    competition_root, raw_source = choose_competition_source(
        input_root, work_root, extract=not args.preflight_only
    )
    preflight["raw_source"] = raw_source
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = output_dir / "autodl_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    _write_json(audit_dir / "preflight.json", preflight)
    shutil.copy2(code_root / "bundle_manifest.json", audit_dir / "bundle_manifest.json")
    full_smi = _run_text(["nvidia-smi", "-q"])
    (audit_dir / "nvidia-smi.txt").write_text(
        full_smi.stdout + full_smi.stderr, encoding="utf-8"
    )
    freeze = _run_text([sys.executable, "-m", "pip", "freeze"], timeout=120)
    (audit_dir / "pip-freeze.txt").write_text(
        freeze.stdout + freeze.stderr, encoding="utf-8"
    )

    if args.preflight_only:
        _write_json(output_dir / "result_manifest.json", result_manifest(output_dir))
        print(json.dumps(preflight, indent=2, ensure_ascii=False))
        print("PREFLIGHT_OK: no extraction, training, full-data run, or submission was performed")
        return 0

    if competition_root is None:
        raise RuntimeError("competition root was not resolved after safe extraction")
    env = exact_profile_environment(
        dict(os.environ),
        code_root=code_root,
        input_root=input_root,
        competition_root=competition_root,
        work_root=work_root,
        output_dir=output_dir,
        run_name=args.run_name,
    )
    _write_json(
        audit_dir / "profile_contract.json",
        {key: value for key, value in sorted(env.items()) if key.startswith("HSI_")},
    )

    return_code = 1
    gate_return_code: int | None = None
    try:
        return_code = _run_tee(
            [sys.executable, str(code_root / "run_hsi_yolo26.py")],
            audit_dir / "wrapper_runner.log",
            env=env,
        )
        if return_code == 0:
            gate = gate_command(code_root, output_dir, args.run_name)
            gate_return_code = _run_tee(
                gate, audit_dir / "wrapper_gate.log", env=dict(os.environ)
            )
            if gate_return_code != 0:
                raise RuntimeError(
                    f"fixed-split gate audit failed with return code {gate_return_code}"
                )
    finally:
        wrapper_status = {
            "profile": PROFILE_NAME,
            "runner_return_code": return_code,
            "gate_return_code": gate_return_code,
            "full_data_started": False,
            "kaggle_submitted": False,
        }
        _write_json(audit_dir / "wrapper_status.json", wrapper_status)
        _write_json(output_dir / "result_manifest.json", result_manifest(output_dir))
        archive_info = archive_results(
            output_dir, output_root / f"{args.run_name}.tar.gz"
        )
        _write_json(output_root / f"{args.run_name}.archive.json", archive_info)
        print("RESULT_ARCHIVE=" + json.dumps(archive_info, ensure_ascii=False))

    if return_code != 0:
        raise RuntimeError(f"fixed runner failed with return code {return_code}")
    gate_report = json.loads((output_dir / "fixed_gate.json").read_text(encoding="utf-8"))
    print(
        "FIXED_GATE_RESULT="
        + json.dumps(
            {
                "best_map50_95": gate_report["best_map50_95"],
                "passes_full_data_gate": gate_report["passes_full_data_gate"],
                "decision": gate_report["decision"],
            },
            ensure_ascii=False,
        )
    )
    print("STOP: this wrapper never starts full-data training or submits to Kaggle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
