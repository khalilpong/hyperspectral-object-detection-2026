from pathlib import Path

import pytest

from scripts.eval_ensemble import _validate_submission_model_count


def test_submission_generation_requires_exactly_one_checkpoint() -> None:
    output = Path("submission.csv")

    _validate_submission_model_count(["m=last.pt"], output)
    _validate_submission_model_count(["m=last.pt", "s=other.pt"], None)

    with pytest.raises(ValueError, match="exactly one trained checkpoint"):
        _validate_submission_model_count([], output)
    with pytest.raises(ValueError, match="exactly one trained checkpoint"):
        _validate_submission_model_count(
            ["m=last.pt", "s=other.pt"],
            output,
        )
