import pandas as pd
import pytest

from scripts.audit_phase2_flip import distribution, matching_audit, validate_lineage
from hsi_detection.submission import SUBMISSION_COLUMNS


def test_matching_is_one_to_one_and_requires_same_image_and_class():
    base = pd.DataFrame([[0, 1, 2, 0.8, 0, 0, 10, 10]], columns=SUBMISSION_COLUMNS)
    candidate = pd.DataFrame([
        [0, 1, 2, 0.9, 0, 0, 10, 10],
        [1, 1, 2, 0.7, 0, 0, 10, 10],
        [2, 1, 3, 0.8, 0, 0, 10, 10],
        [3, 2, 2, 0.8, 0, 0, 10, 10],
    ], columns=SUBMISSION_COLUMNS)
    result = matching_audit(base, candidate, min_conf=0.25, min_iou=0.95)
    assert result["matched"] == 1
    assert result["candidate_unmatched"] == 3
    assert result["baseline_replacement_rate"] == 0
    assert distribution(candidate)["area_bins_pixels"]["small_lt_32_squared"] == 4


def test_lineage_rejects_wrong_candidate_before_reporting_metrics():
    with pytest.raises(ValueError, match="candidate"):
        validate_lineage({"candidate": "B"}, "A", {"A": {}})
