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
    expected_architecture: str = "yolo",
    expected_model: str | None = None,
    expected_dfl: float | None = None,
    expected_degrees: float = 0.0,
    expected_extra_channel_init: str = "random",
    expected_rtdetr_num_denoising: int = 100,
    gate: float,
) -> dict[str, object]:
    if expected_architecture not in {"yolo", "rtdetr"}:
        raise ValueError(f"Unsupported architecture: {expected_architecture!r}")
    if expected_model is None:
        expected_model = (
            "rtdetr-l.pt" if expected_architecture == "rtdetr" else "yolo26m.pt"
        )
    if expected_architecture == "yolo" and expected_dfl is None:
        raise ValueError("YOLO fixed-split audit requires expected_dfl")
    if expected_architecture == "rtdetr" and expected_dfl is not None:
        raise ValueError("RT-DETR does not use the YOLO expected_dfl contract")
    if not math.isfinite(expected_degrees) or not 0.0 <= expected_degrees <= 180.0:
        raise ValueError("expected_degrees must be finite and within [0, 180]")
    if expected_rtdetr_num_denoising <= 0:
        raise ValueError("expected_rtdetr_num_denoising must be positive")
    if expected_architecture == "rtdetr" and expected_degrees != 0.0:
        raise ValueError("RT-DETR does not use the YOLO expected_degrees contract")
    if expected_architecture == "yolo" and expected_rtdetr_num_denoising != 100:
        raise ValueError("YOLO does not use the RT-DETR denoising-query contract")
    if expected_extra_channel_init not in {"random", "zero"}:
        raise ValueError(
            "expected_extra_channel_init must be either 'random' or 'zero'"
        )

    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("result") != "success":
        raise ValueError(f"Run is not successful: result={status.get('result')!r}")

    config = status.get("config")
    if not isinstance(config, dict):
        raise ValueError("status.json is missing a config object")
    # Runs created before these isolated controls were added omitted both keys;
    # their effective Ultralytics defaults were degrees=0 and nd=100.
    config.setdefault("DEGREES", 0.0)
    config.setdefault("RTDETR_NUM_DENOISING", 100)
    observed_architecture = config.get("ARCHITECTURE", "yolo")
    if observed_architecture != expected_architecture:
        raise ValueError(
            "Fixed-split config mismatch: "
            f"{{'ARCHITECTURE': {{'observed': {observed_architecture!r}, "
            f"'expected': {expected_architecture!r}}}}}"
        )
    _expect_equal(
        config,
        {
            "MODE": "ablation",
            "MODEL": expected_model,
            "EPOCHS": 30,
            "MULTISCALE": False,
            "DATA": "hsi16",
            "SEED": 2026,
            "EXTRA_CHANNEL_INIT": expected_extra_channel_init,
            "SPECTRAL_STEM": False,
            "LOWER_PERCENTILE": 0.5,
            "UPPER_PERCENTILE": 99.5,
            "CLS_PW": 0.0,
            "SCALE": 0.5,
            "DEGREES": expected_degrees,
            "DFL": 1.5 if expected_architecture == "rtdetr" else expected_dfl,
            "RTDETR_NUM_DENOISING": expected_rtdetr_num_denoising,
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
    contract: dict[str, object] = {
        "single_model": True,
        "architecture": expected_architecture,
        "model": expected_model,
        "fixed_split": "fixed_2400_train_600_val",
        "expected_extra_channel_init": expected_extra_channel_init,
        "gate": gate,
    }
    if expected_dfl is not None:
        contract["expected_dfl"] = expected_dfl
    if expected_architecture == "yolo":
        contract["expected_degrees"] = expected_degrees
    else:
        contract["expected_rtdetr_num_denoising"] = expected_rtdetr_num_denoising
    return {
        "contract": contract,
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
    parser.add_argument(
        "--architecture", choices=("yolo", "rtdetr"), default="yolo"
    )
    parser.add_argument("--model")
    parser.add_argument("--expected-dfl", type=float)
    parser.add_argument("--expected-degrees", type=float, default=0.0)
    parser.add_argument(
        "--expected-extra-channel-init",
        choices=("random", "zero"),
        default="random",
    )
    parser.add_argument("--expected-rtdetr-num-denoising", type=int, default=100)
    parser.add_argument("--gate", type=float, default=0.70443)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.architecture == "yolo" and args.expected_dfl is None:
        parser.error("--expected-dfl is required for --architecture yolo")
    if args.architecture == "rtdetr" and args.expected_dfl is not None:
        parser.error("--expected-dfl is not valid for --architecture rtdetr")

    report = audit_fixed_split(
        status_path=args.status,
        results_path=args.results,
        expected_architecture=args.architecture,
        expected_model=args.model,
        expected_dfl=args.expected_dfl,
        expected_degrees=args.expected_degrees,
        expected_extra_channel_init=args.expected_extra_channel_init,
        expected_rtdetr_num_denoising=args.expected_rtdetr_num_denoising,
        gate=args.gate,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
