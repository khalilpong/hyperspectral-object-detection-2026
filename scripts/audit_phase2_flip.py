"""Read-only prediction risk audit for the two fixed Phase 2 flip candidates.

No labels, thresholds used for prediction, model calls, or parameter search are
involved. IoU and confidence cutoffs below are diagnostic reports only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from produce_phase2_flip import (
    BASELINE, BASELINE_SHA, CANDIDATES, CHECKPOINT_SHA, IMAGES, SCALES, WORK,
    _image_sizes, _sha256, _validated_submission, require_hash, verify_fixed_evidence, write_json,
)


def distribution(frame: pd.DataFrame) -> dict:
    widths = frame.x2.to_numpy() - frame.x1.to_numpy()
    heights = frame.y2.to_numpy() - frame.y1.to_numpy()
    areas = widths * heights
    quantiles = [0, 0.1, 0.5, 0.9, 0.99, 1]
    return {
        "rows": len(frame), "images": int(frame.image_id.nunique()),
        "class_counts": {str(k): int(v) for k, v in frame.class_id.value_counts().sort_index().items()},
        "area_bins_pixels": {"small_lt_32_squared": int((areas < 32**2).sum()),
                             "medium_32_to_96_squared": int(((areas >= 32**2) & (areas < 96**2)).sum()),
                             "large_ge_96_squared": int((areas >= 96**2).sum())},
        "quantile_probabilities": quantiles,
        "width_quantiles": np.quantile(widths, quantiles).tolist(),
        "height_quantiles": np.quantile(heights, quantiles).tolist(),
        "area_quantiles": np.quantile(areas, quantiles).tolist(),
        "confidence_quantiles": np.quantile(frame.confidence, quantiles).tolist(),
        "confidence_ge_025": int((frame.confidence >= 0.25).sum()),
        "confidence_ge_050": int((frame.confidence >= 0.5).sum()),
    }


def matching_audit(baseline: pd.DataFrame, candidate: pd.DataFrame,
                   *, min_conf: float, min_iou: float) -> dict:
    """Same-image/class greedy one-to-one matching, candidate confidence first."""
    base = baseline[baseline.confidence >= min_conf]
    cand = candidate[candidate.confidence >= min_conf]
    base_groups = {key: group for key, group in base.groupby(["image_id", "class_id"], sort=True)}
    matched, ious, confidence_delta = 0, [], []
    columns = ["x1", "y1", "x2", "y2"]
    for key, group in cand.groupby(["image_id", "class_id"], sort=True):
        if key not in base_groups:
            continue
        reference = base_groups[key]
        b = reference[columns].to_numpy()
        a = group[columns].to_numpy()
        bc = reference.confidence.to_numpy()
        ac = group.confidence.to_numpy()
        intersection = np.prod(np.maximum(
            np.minimum(a[:, None, 2:], b[None, :, 2:]) -
            np.maximum(a[:, None, :2], b[None, :, :2]), 0), axis=2)
        union = np.prod(a[:, 2:] - a[:, :2], axis=1)[:, None] + np.prod(
            b[:, 2:] - b[:, :2], axis=1)[None, :] - intersection
        overlap = intersection / np.maximum(union, 1e-12)
        available = np.ones(len(b), dtype=bool)
        for index in np.argsort(-ac, kind="stable"):
            scores = np.where(available, overlap[index], -1)
            best = int(np.argmax(scores))
            if scores[best] >= min_iou:
                matched += 1
                available[best] = False
                ious.append(float(scores[best]))
                confidence_delta.append(float(ac[index] - bc[best]))
    return {"confidence_cutoff_diagnostic_only": min_conf, "match_iou": min_iou,
            "baseline_boxes": len(base), "candidate_boxes": len(cand), "matched": matched,
            "baseline_unmatched": len(base) - matched, "candidate_unmatched": len(cand) - matched,
            "baseline_replacement_rate": (len(base) - matched) / max(len(base), 1),
            "candidate_new_or_replaced_rate": (len(cand) - matched) / max(len(cand), 1),
            "mean_matched_iou": float(np.mean(ious)) if ious else None,
            "mean_matched_confidence_delta": float(np.mean(confidence_delta)) if confidence_delta else None}


def audit_frames(base: pd.DataFrame, cand: pd.DataFrame) -> dict:
    before, after = distribution(base), distribution(cand)
    return {"baseline": before, "candidate": after,
            "row_delta": len(cand) - len(base), "row_ratio": len(cand) / len(base),
            "class_count_delta": {str(k): after["class_counts"].get(str(k), 0) - before["class_counts"].get(str(k), 0)
                                  for k in range(18)},
            "matching": [matching_audit(base, cand, min_conf=conf, min_iou=iou)
                         for conf, iou in ((0, 0.95), (0.25, 0.95), (0.25, 0.5))]}


def validate_lineage(manifest: dict, candidate: str, evidence: dict) -> None:
    expected = {"candidate": candidate, "checkpoint_sha256": CHECKPOINT_SHA,
                "single_checkpoint": True, "inference_only": True,
                "training_or_adaptation": False, "scales": list(SCALES),
                "flip_imgsz": 1024, "fusion_iou": CANDIDATES[candidate],
                "support_gain": 0.125, "max_det": 300, "test_ranking_overlap": 0,
                "calibration": {"width_scale": 1.01, "height_scale": 1.01,
                                "center_x_shift": 0.0, "center_y_shift": 0.0},
                "fixed_validation": evidence[candidate]}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Candidate lineage mismatch: {key}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    args = parser.parse_args()
    require_hash(BASELINE, BASELINE_SHA)
    source_manifest = WORK / f"{args.candidate}_manifest.json"
    manifest = json.loads(source_manifest.read_text())
    validate_lineage(manifest, args.candidate, verify_fixed_evidence())
    candidate_path = Path(manifest["combined"]["path"])
    require_hash(candidate_path, manifest["combined"]["output_sha256"])
    sizes = {split: _image_sizes(path, expected_count=1000, label=split) for split, path in IMAGES.items()}
    combined_sizes = {**sizes["test"], **sizes["ranking"]}
    baseline, _ = _validated_submission(BASELINE, combined_sizes, class_count=18, label="baseline")
    candidate, _ = _validated_submission(candidate_path, combined_sizes, class_count=18, label="candidate")
    parts = []
    for split in ("test", "ranking"):
        item = manifest["splits"][split]
        if item["checkpoint_sha256"] != CHECKPOINT_SHA:
            raise ValueError("Split checkpoint mismatch")
        for stage in ("raw", "calibrated"):
            require_hash(Path(item[stage]["path"]), item[stage]["output_sha256"])
        part, _ = _validated_submission(Path(item["calibrated"]["path"]), sizes[split],
                                        class_count=18, label=split)
        parts.append(part)
    reconstructed = pd.concat(parts, ignore_index=True)
    reconstructed["id"] = np.arange(len(reconstructed), dtype=np.int64)
    if reconstructed.shape != candidate.shape or not np.allclose(
            reconstructed.to_numpy(), candidate.to_numpy(), rtol=0, atol=1e-10):
        raise ValueError("Combined CSV does not match the two calibrated inputs")
    result = {"candidate": args.candidate, "manifest_sha256": _sha256(source_manifest),
              "csv_sha256": _sha256(candidate_path), "baseline_sha256": BASELINE_SHA,
              "fixed_validation": manifest["fixed_validation"],
              "risk": "Fixed-validation gain is tiny and measured before boxscale101; no guarantee of Public or private improvement. Public scores concern test1000 only, and cannot reveal ranking labels or justify ranking adaptation.",
              "splits": {}}
    for split, image_sizes in sizes.items():
        result["splits"][split] = audit_frames(
            baseline[baseline.image_id.isin(image_sizes)], candidate[candidate.image_id.isin(image_sizes)])
    write_json(WORK / f"{args.candidate}_risk_audit.json", result)
    print(json.dumps({split: {"rows": value["candidate"]["rows"], "row_delta": value["row_delta"],
                              "replacement_all_iou095": value["matching"][0]["baseline_replacement_rate"],
                              "replacement_conf025_iou095": value["matching"][1]["baseline_replacement_rate"],
                              "replacement_conf025_iou050": value["matching"][2]["baseline_replacement_rate"]}
                      for split, value in result["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
