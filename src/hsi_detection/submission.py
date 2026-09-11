from __future__ import annotations

from collections.abc import Mapping
import math

import pandas as pd


SUBMISSION_COLUMNS = [
    "id",
    "image_id",
    "class_id",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
]


def clip_xyxy(
    box: tuple[float, float, float, float] | list[float],
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    """Clip a predicted box to its image, returning None when no area remains."""
    x1, y1, x2, y2 = (float(value) for value in box)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    if x1 >= x2 or y1 >= y2:
        return None
    return x1, y1, x2, y2


def validate_submission_frame(
    frame: pd.DataFrame,
    image_sizes: Mapping[int, tuple[int, int]],
    class_count: int,
) -> list[str]:
    """Return human-readable issues found in a competition submission."""
    issues: list[str] = []
    if list(frame.columns) != SUBMISSION_COLUMNS:
        issues.append(f"columns must be exactly {SUBMISSION_COLUMNS}")
        return issues
    if frame.empty:
        issues.append("submission contains no detections")
        return issues

    expected_ids = list(range(len(frame)))
    actual_ids = frame["id"].tolist()
    if actual_ids != expected_ids:
        issues.append("id must be consecutive integers in row order starting at 0")

    for row_number, row in enumerate(frame.itertuples(index=False)):
        try:
            image_id = int(row.image_id)
            class_id = int(row.class_id)
            confidence, x1, y1, x2, y2 = (
                float(row.confidence),
                float(row.x1),
                float(row.y1),
                float(row.x2),
                float(row.y2),
            )
        except (TypeError, ValueError, OverflowError):
            issues.append(f"row {row_number}: non-numeric value")
            continue
        if not all(math.isfinite(value) for value in (confidence, x1, y1, x2, y2)):
            issues.append(f"row {row_number}: NaN or infinite value")
            continue
        if image_id not in image_sizes:
            issues.append(f"row {row_number}: unknown image_id {image_id}")
            continue
        if not 0 <= class_id < class_count:
            issues.append(f"row {row_number}: class_id {class_id} outside 0..{class_count - 1}")
        if not 0 <= confidence <= 1:
            issues.append(f"row {row_number}: confidence outside 0..1")
        width, height = image_sizes[image_id]
        if not (0 <= x1 < x2 <= width):
            issues.append(f"row {row_number}: invalid x coordinates for width {width}")
        if not (0 <= y1 < y2 <= height):
            issues.append(f"row {row_number}: invalid y coordinates for height {height}")
    return issues
