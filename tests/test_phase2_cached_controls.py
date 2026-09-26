import pytest

from scripts.eval_phase2_cached_controls import PINNED, group_indices, require_control


def test_hash_pins_have_full_sha256_width():
    assert all(len(value) == 64 and int(value, 16) >= 0 for value in PINNED.values())


def test_cache_control_rejects_numeric_drift_and_nan():
    require_control(0.7 + 1e-12, 0.7)
    with pytest.raises(RuntimeError, match="drift"):
        require_control(0.7001, 0.7)
    with pytest.raises(RuntimeError, match="drift"):
        require_control(float("nan"), 0.7)


def test_diagnostic_groups_partition_all_images_without_overlap():
    groups = group_indices(600)
    assert [len(group) for group in groups] == [200, 200, 200]
    assert sorted(index for group in groups for index in group) == list(range(600))
