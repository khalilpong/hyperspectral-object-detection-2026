"""Audit a downloaded fixed-split Kaggle run and apply the full-data gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


METRIC_COLUMN = "metrics/mAP50-95(B)"
MAP50_COLUMN = "metrics/mAP50(B)"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _expect_equal(config: dict[str, object], expected: dict[str, object]) -> None:
    mismatches = {
        key: {"observed": config.get(key), "expected": value}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Fixed-split config mismatch: {mismatches}")


def audit_fixed_split(
    *,
    status_path: Path,
    results_path: Path,
    expected_dfl: float,
    gate: float,
) -> dict[str, object]:
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("result") != "success":
        raise ValueError(f"Run is not successful: result={status.get('result')!r}")

    config = status.get("config")
    if not isinstance(config, dict):
        raise ValueError("status.json is missing a config object")
    _expect_equal(
        config,
        {
            "MODE": "ablation",
            "MODEL": "yolo26m.pt",
            "EPOCHS": 30,
            "MULTISCALE": False,
            "DATA": "hsi16",
            "SEED": 2026,
            "EXTRA_CHANNEL_INIT": "random",
            "SPECTRAL_STEM": False,
            "LOWER_PERCENTILE": 0.5,
            "UPPER_PERCENTILE": 99.5,
            "CLS_PW": 0.0,
            "SCALE": 0.5,
            "DFL": expected_dfl,
            "PHASE_TARGET_LONG_EDGE": 1024,
            "OBJECT_CROPS": False,
            "TILE_INFERENCE": False,
            "IMGSZ": 1024,
        },
    )

    steps = status.get("steps")
    if not isinstance(steps, dict):
        raise ValueError("status.json is missing a steps object")
    attempts = [
        value
        for key, value in steps.items()
        if key.startswith("train_attempt_") and isinstance(value, dict)
    ]
    successful_attempts = [
        attempt
        for attempt in attempts
        if attempt.get("returncode") == 0
        and attempt.get("cuda_oom") is False
        and attempt.get("shm_error") is False
    ]
    if len(successful_attempts) != 1:
        raise ValueError(
            "Expected exactly one successful non-OOM/non-SHM training attempt, "
            f"found {len(successful_attempts)}"
        )

    with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 30:
        raise ValueError(f"Expected 30 epoch rows, found {len(rows)}")
    required = {"epoch", METRIC_COLUMN, MAP50_COLUMN}
    if not required.issubset(rows[0]):
        raise ValueError(f"results.csv is missing columns: {sorted(required - rows[0].keys())}")

    epochs = [int(float(row["epoch"])) for row in rows]
    if epochs != list(range(1, 31)):
        raise ValueError(f"Expected epochs 1..30, found {epochs}")
    map50_95 = [float(row[METRIC_COLUMN]) for row in rows]
    map50 = [float(row[MAP50_COLUMN]) for row in rows]
    if not all(math.isfinite(value) for value in (*map50_95, *map50)):
        raise ValueError("results.csv contains non-finite validation metrics")

    best_index = max(range(len(rows)), key=map50_95.__getitem__)
    best = map50_95[best_index]
    final = map50_95[-1]
    passed = best >= gate
    return {
        "contract": {
            "single_model": True,
            "fixed_split": "fixed_2400_train_600_val",
            "expected_dfl": expected_dfl,
            "gate": gate,
        },
        "run_name": config.get("RUN_NAME"),
        "best_epoch": epochs[best_index],
        "best_map50_95": best,
        "best_map50": map50[best_index],
        "final_map50_95": final,
        "delta_vs_baseline_0_70143": best - 0.70143,
        "margin_vs_gate": best - gate,
        "passes_full_data_gate": passed,
        "decision": "eligible_for_single_checkpoint_full_data" if passed else "reject_no_full_data",
        "training_attempt": successful_attempts[0],
        "hashes": {
            "status_sha256": _sha256(status_path),
            "results_sha256": _sha256(results_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--expected-dfl", type=float, required=True)
    parser.add_argument("--gate", type=float, default=0.70443)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = audit_fixed_split(
        status_path=args.status,
        results_path=args.results,
        expected_dfl=args.expected_dfl,
        gate=args.gate,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
