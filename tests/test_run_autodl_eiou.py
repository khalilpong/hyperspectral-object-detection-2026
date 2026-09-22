import json
import zipfile
from pathlib import Path

import pytest

from scripts import run_autodl_eiou as autodl


def _minimal_layout(root: Path) -> Path:
    competition = root / "competition"
    (competition / "data_train" / "data_train" / "Annotations" / "VIS").mkdir(
        parents=True
    )
    (competition / "data_train" / "data_train" / "VIS").mkdir(parents=True)
    (competition / "data_test" / "data_test" / "VIS").mkdir(parents=True)
    (competition / "class.txt").write_text("class\n", encoding="utf-8")
    return competition


def test_exact_profile_environment_strips_ambient_hsi(tmp_path):
    paths = [tmp_path / name for name in ("code", "input", "competition", "work", "out")]
    env = autodl.exact_profile_environment(
        {"PATH": "kept", "HSI_BOX_IOU_LOSS": "ciou", "HSI_EPOCHS": "999"},
        code_root=paths[0],
        input_root=paths[1],
        competition_root=paths[2],
        work_root=paths[3],
        output_dir=paths[4],
        run_name="test_eiou",
    )
    assert env["PATH"] == "kept"
    assert env["HSI_BOX_IOU_LOSS"] == "eiou"
    assert env["HSI_EPOCHS"] == "30"
    assert env["HSI_ATTEMPTS"] == "8:2,6:2,4:2,4:0"
    assert env["HSI_BAND_ORDER"] == autodl.DEFAULT_BAND_ORDER
    assert env["HSI_RUN_NAME"] == "test_eiou"


def test_safe_zip_rejects_traversal(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("class.txt", "classes")
        handle.writestr("../escape.txt", "bad")
    with pytest.raises(RuntimeError, match="unsafe zip member"):
        autodl.inspect_safe_zip(archive)


def test_safe_zip_is_audited_without_extraction(tmp_path):
    source = _minimal_layout(tmp_path / "source")
    input_root = tmp_path / "input"
    input_root.mkdir()
    archive = input_root / "competition.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                handle.write(path, arcname=path.relative_to(source).as_posix())
        for relative in (
            "data_train/data_train/Annotations/VIS/example.xml",
            "data_train/data_train/VIS/example.png",
            "data_test/data_test/VIS/example.png",
        ):
            handle.writestr(relative, "x")
    competition_root, audit = autodl.choose_competition_source(
        input_root, tmp_path / "work", extract=False
    )
    assert competition_root is None
    assert audit["kind"] == "zip"
    assert audit["extracted"] is False
    assert audit["archive"]["class_member"] == "class.txt"
    assert all(audit["archive"]["layout_members"].values())


def test_safe_zip_extracts_to_hashed_work_directory(tmp_path):
    source = _minimal_layout(tmp_path / "source")
    for relative in (
        "data_train/data_train/Annotations/VIS/example.xml",
        "data_train/data_train/VIS/example.png",
        "data_test/data_test/VIS/example.png",
    ):
        path = source / relative
        path.write_text("x", encoding="utf-8")
    input_root = tmp_path / "input"
    input_root.mkdir()
    archive = input_root / "competition.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                handle.write(path, arcname=path.relative_to(source).as_posix())

    resolved, audit = autodl.choose_competition_source(
        input_root, tmp_path / "work", extract=True
    )
    assert resolved is not None
    assert audit["extracted"] is True
    assert (resolved / "class.txt").is_file()
    marker = resolved / ".autodl_raw_source.json"
    assert marker.is_file()
    assert json.loads(marker.read_text(encoding="utf-8"))["sha256"] == autodl.sha256(
        archive
    )


def test_extracted_layout_is_resolved(tmp_path):
    competition = _minimal_layout(tmp_path)
    resolved, audit = autodl.choose_competition_source(
        tmp_path, tmp_path / "work", extract=False
    )
    assert resolved == competition
    assert audit["kind"] == "directory"
    assert audit["layout"]["root"] == str(competition.resolve())


def test_reversed_vis_layout_is_rejected(tmp_path):
    archive = tmp_path / "reversed.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("class.txt", "class\n")
        handle.writestr("VIS/Annotations/example.xml", "x")
        handle.writestr("VIS/data_train/example.png", "x")
        handle.writestr("VIS/data_test/example.png", "x")

    with pytest.raises(RuntimeError, match="exactly one annotation"):
        autodl.inspect_safe_zip(archive)


def test_bundle_manifest_detects_drift(tmp_path, monkeypatch):
    code = tmp_path / "code"
    code.mkdir()
    monkeypatch.setattr(autodl, "REQUIRED_BUNDLE_FILES", ("bundle_manifest.json", "x.py"))
    (code / "x.py").write_text("one\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "profile": autodl.PROFILE_NAME,
        "files": [
            {
                "path": "x.py",
                "bytes": (code / "x.py").stat().st_size,
                "sha256": autodl.sha256(code / "x.py"),
            }
        ],
    }
    (code / "bundle_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(autodl, "EXPECTED_YOLO26M_SHA256", "0" * 64)
    # The real validator also requires the checkpoint; use a focused drift assertion.
    (code / "x.py").write_text("two\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="bundle file drift"):
        autodl.validate_bundle(code)


def test_bundle_manifest_must_cover_every_required_file(tmp_path, monkeypatch):
    code = tmp_path / "code"
    code.mkdir()
    required = ("bundle_manifest.json", "x.py", "yolo26m.pt")
    monkeypatch.setattr(autodl, "REQUIRED_BUNDLE_FILES", required)
    (code / "x.py").write_text("required\n", encoding="utf-8")
    weight = code / "yolo26m.pt"
    weight.write_bytes(b"weight")
    monkeypatch.setattr(autodl, "EXPECTED_YOLO26M_SHA256", autodl.sha256(weight))
    manifest = {
        "schema_version": 1,
        "profile": autodl.PROFILE_NAME,
        "files": [
            {
                "path": "yolo26m.pt",
                "bytes": weight.stat().st_size,
                "sha256": autodl.sha256(weight),
            }
        ],
    }
    (code / "bundle_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="does not cover required files"):
        autodl.validate_bundle(code)


def test_bundle_manifest_rejects_undeclared_top_level_file(tmp_path, monkeypatch):
    code = tmp_path / "code"
    code.mkdir()
    required = ("bundle_manifest.json", "yolo26m.pt")
    monkeypatch.setattr(autodl, "REQUIRED_BUNDLE_FILES", required)
    weight = code / "yolo26m.pt"
    weight.write_bytes(b"weight")
    monkeypatch.setattr(autodl, "EXPECTED_YOLO26M_SHA256", autodl.sha256(weight))
    manifest = {
        "schema_version": 1,
        "profile": autodl.PROFILE_NAME,
        "files": [
            {
                "path": "yolo26m.pt",
                "bytes": weight.stat().st_size,
                "sha256": autodl.sha256(weight),
            }
        ],
    }
    (code / "bundle_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (code / "unexpected.py").write_text("unexpected\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="undeclared files"):
        autodl.validate_bundle(code)


def test_gate_command_locks_eiou_and_preparation_artifacts(tmp_path):
    command = autodl.gate_command(tmp_path / "code", tmp_path / "out", "fixed")
    joined = " ".join(command)
    assert "--expected-box-iou-loss eiou" in joined
    assert autodl.EXPECTED_YOLO26M_SHA256 in command
    assert "--expected-band-order " + autodl.DEFAULT_BAND_ORDER in joined
    assert str(tmp_path / "out" / "data_contract" / "preparation_report.json") in command


def test_archive_results_refuses_overwrite(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "status.json").write_text("{}", encoding="utf-8")
    archive = tmp_path / "run.tar.gz"
    info = autodl.archive_results(output, archive)
    assert info["sha256"] == autodl.sha256(archive)
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        autodl.archive_results(output, archive)
