import numpy as np
import pytest

from hsi_detection.spectral import (
    encode_multispectral_uint8,
    make_multispectral_uint8,
    make_pseudo_rgb,
    multispectral_percentile_bounds,
    phase_aware_reconstruct,
    phase_aware_target_shape,
    x2cube,
)


def test_x2cube_matches_row_major_mosaic_cells() -> None:
    mosaic = np.arange(8 * 12, dtype=np.uint16).reshape(8, 12)
    cube = x2cube(mosaic, cell_size=4)
    assert cube.shape == (2, 3, 16)
    np.testing.assert_array_equal(cube[0, 0], mosaic[:4, :4].reshape(-1))
    np.testing.assert_array_equal(cube[1, 2], mosaic[4:8, 8:12].reshape(-1))


def test_x2cube_rejects_non_divisible_shape() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        x2cube(np.zeros((7, 8), dtype=np.uint16))


def test_phase_aware_target_shape_preserves_aspect_ratio() -> None:
    assert phase_aware_target_shape((964, 1972), target_long_edge=1024) == (501, 1024)
    assert phase_aware_target_shape((2000, 1000), target_long_edge=1024) == (1024, 512)


def test_phase_aware_reconstruct_keeps_a_constant_field_constant() -> None:
    mosaic = np.full((8, 12), 37, dtype=np.uint16)

    reconstructed = phase_aware_reconstruct(mosaic, target_shape=(5, 9))

    assert reconstructed.shape == (5, 9, 16)
    assert reconstructed.dtype == np.float32
    np.testing.assert_allclose(
        reconstructed,
        np.full((5, 9, 16), 37, dtype=np.float32),
        rtol=0,
        atol=1e-5,
    )


def test_phase_aware_reconstruct_preserves_each_physical_sample_location() -> None:
    mosaic = np.arange(16 * 20, dtype=np.float32).reshape(16, 20)

    reconstructed = phase_aware_reconstruct(mosaic, target_shape=mosaic.shape)

    for physical_band in range(16):
        phase_row, phase_column = divmod(physical_band, 4)
        np.testing.assert_array_equal(
            reconstructed[phase_row::4, phase_column::4, physical_band],
            mosaic[phase_row::4, phase_column::4],
        )


def test_phase_aware_reconstruct_uses_band_specific_phase_offsets() -> None:
    rows, columns = np.indices((16, 20), dtype=np.float32)
    linear_scene = rows * 100.0 + columns

    reconstructed = phase_aware_reconstruct(linear_scene, target_shape=linear_scene.shape)

    # This interior is covered by every physical phase without border clamping.
    expected = np.broadcast_to(linear_scene[3:13, 3:17, None], (10, 14, 16))
    np.testing.assert_allclose(reconstructed[3:13, 3:17], expected, rtol=0, atol=1e-4)


def test_phase_aware_reconstruct_validates_geometry_and_values() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        phase_aware_reconstruct(np.zeros((7, 8), dtype=np.uint16), target_shape=(4, 4))
    with pytest.raises(ValueError, match="positive"):
        phase_aware_reconstruct(np.zeros((8, 8), dtype=np.uint16), target_shape=(0, 4))
    with pytest.raises(ValueError, match="duplicates"):
        phase_aware_reconstruct(
            np.zeros((8, 8), dtype=np.uint16),
            target_shape=(4, 4),
            band_order=(0, 1, 1),
        )
    non_finite = np.zeros((8, 8), dtype=np.float32)
    non_finite[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        phase_aware_reconstruct(non_finite, target_shape=(4, 4))


def test_pseudo_rgb_has_expected_shape_and_type() -> None:
    cube = np.arange(5 * 7 * 16, dtype=np.uint16).reshape(5, 7, 16)
    image = make_pseudo_rgb(cube, bands=(5, 8, 13), lower_percentile=0, upper_percentile=100)
    assert image.shape == (5, 7, 3)
    assert image.dtype == np.uint8
    assert image.min() == 0
    assert image.max() == 255


def test_make_multispectral_uint8_preserves_requested_order_and_shared_scale() -> None:
    cube = np.array(
        [
            [[0, 10, 20, 30], [40, 50, 60, 70]],
            [[80, 90, 100, 110], [120, 130, 140, 150]],
        ],
        dtype=np.uint16,
    )

    result = make_multispectral_uint8(
        cube,
        band_order=(2, 0, 3, 1),
        lower_percentile=0,
        upper_percentile=100,
    )

    expected = np.rint(cube[:, :, (2, 0, 3, 1)] / 150.0 * 255.0).astype(np.uint8)
    np.testing.assert_array_equal(result, expected)


def test_make_multispectral_uint8_rejects_duplicate_bands() -> None:
    cube = np.zeros((2, 2, 4), dtype=np.uint16)
    with pytest.raises(ValueError, match="duplicates"):
        make_multispectral_uint8(cube, band_order=(0, 1, 1, 2))


def test_multispectral_bounds_can_be_reused_after_reconstruction() -> None:
    source = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    low, high = multispectral_percentile_bounds(
        source,
        band_order=(2, 0),
        lower_percentile=0,
        upper_percentile=100,
    )
    reconstructed = np.repeat(np.repeat(source[:, :, (2, 0)], 2, axis=0), 2, axis=1)

    encoded = encode_multispectral_uint8(reconstructed, low=low, high=high)

    expected = np.rint(np.clip((reconstructed - low) / (high - low), 0, 1) * 255).astype(
        np.uint8
    )
    np.testing.assert_array_equal(encoded, expected)
