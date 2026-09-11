import pandas as pd

from hsi_detection.submission import SUBMISSION_COLUMNS, validate_submission_frame


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [[0, 1009, 2, 0.9, 1.0, 2.0, 20.0, 30.0]],
        columns=SUBMISSION_COLUMNS,
    )


def test_valid_submission_frame() -> None:
    assert validate_submission_frame(valid_frame(), {1009: (501, 253)}, 18) == []


def test_submission_frame_rejects_bad_box_and_class() -> None:
    frame = valid_frame()
    frame.loc[0, "class_id"] = 18
    frame.loc[0, "x2"] = 999
    issues = validate_submission_frame(frame, {1009: (501, 253)}, 18)
    assert any("class_id" in issue for issue in issues)
    assert any("x coordinates" in issue for issue in issues)

