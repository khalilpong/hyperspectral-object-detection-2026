from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]


def test_multispectral_runner_uses_memory_safe_workers_for_new_and_resumed_runs() -> None:
    script = (PROJECT_ROOT / "scripts" / "run_multispectral_followup.ps1").read_text(
        encoding="utf-8"
    )

    assert "[int]$Workers = 0" in script
    assert script.count('"--workers", [string]$Workers') == 2


def test_multispectral_runner_accepts_isolated_run_name_and_epoch_budget() -> None:
    script = (PROJECT_ROOT / "scripts" / "run_multispectral_followup.ps1").read_text(
        encoding="utf-8"
    )

    assert '[string]$RunName = "ablation_s1024_b4_hsi16_shared_p005_995_r1"' in script
    assert "[int]$TargetEpochs = 30" in script
    assert script.count('$RunDir = Join-Path $ProjectRoot ("runs\\" + $RunName)') == 1
    assert '"--epochs", [string]$TargetEpochs' in script


def test_multispectral_runner_preserves_complete_native_tracebacks() -> None:
    script = (PROJECT_ROOT / "scripts" / "run_multispectral_followup.ps1").read_text(
        encoding="utf-8"
    )

    continue_line = '$ErrorActionPreference = "Continue"'
    restore_line = "$ErrorActionPreference = $previousErrorActionPreference"
    tee_line = "Tee-Object -FilePath $logPath -Append"
    assert script.index(continue_line) < script.index(tee_line) < script.index(restore_line)


def test_hsi_followups_disable_spawn_workers() -> None:
    geometry = (PROJECT_ROOT / "scripts" / "run_geometry_followup.ps1").read_text(
        encoding="utf-8"
    )
    final = (
        PROJECT_ROOT / "scripts" / "run_final_candidate_followup.ps1"
    ).read_text(encoding="utf-8")

    hsi_geometry = geometry.split(
        'name = "ablation_s1024_b3_hsi16_shared_p005_995_p2_e60_r1"', 1
    )[1].split('name = "ablation_s1024_b4_b5-8-13_multiscale025_e60_r1"', 1)[0]
    hsi_p2_final = final.split('name = "hsi16_shared_p005_995_p2_e60"', 1)[1].split(
        'name = "pseudo_rgb_multiscale025_e60"', 1
    )[0]
    hsi_final = final.split('name = "hsi16_shared_p005_995"', 1)[1].split(
        'name = "pseudo_rgb_scale075"', 1
    )[0]

    assert "workers = 0" in hsi_geometry
    assert "workers = 0" in hsi_p2_final
    assert "workers = 0" in hsi_final
    assert geometry.count('"--workers", [string]$job.workers') == 2
    assert final.count('"--workers", [string]$selectedCandidate.workers') == 3


def test_background_runners_do_not_truncate_native_stderr() -> None:
    expected_tee_counts = {
        "run_background_experiments.ps1": 1,
        "run_multispectral_followup.ps1": 1,
        "run_geometry_followup.ps1": 1,
        "run_final_candidate_followup.ps1": 2,
        "run_submission_followup.ps1": 2,
    }

    for filename, expected_count in expected_tee_counts.items():
        script = (PROJECT_ROOT / "scripts" / filename).read_text(encoding="utf-8")
        assert script.count('$ErrorActionPreference = "Continue"') == expected_count
        assert (
            script.count("$ErrorActionPreference = $previousErrorActionPreference")
            == expected_count
        )
        assert script.count("Tee-Object -FilePath") == expected_count


def test_ensemble_builder_isolates_optional_p1_p99_member() -> None:
    script = (PROJECT_ROOT / "scripts" / "build_ensemble_submission.sh").read_text(
        encoding="utf-8"
    )

    assert 'W_I="${W_I:-0.4}"' in script
    assert 'if [[ -n "${P1_WEIGHTS:-}" ]]; then' in script
    assert "m_p010_990_full=${P1_WEIGHTS}@${W_I}" in script
    assert "#data/processed/hsi16_shared_p010_990" in script
    assert '"${P1_MODEL[@]}"' in script
