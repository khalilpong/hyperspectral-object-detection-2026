import numpy as np
import pytest

from hsi_detection.spectral import make_multispectral_uint8, make_pseudo_rgb, x2cube


def test_x2cube_matches_row_major_mosaic_cells() -> None:
    mosaic = np.arange(8 * 12, dtype=np.uint16).reshape(8, 12)
    cube = x2cube(mosaic, cell_size=4)
    assert cube.shape == (2, 3, 16)
    np.testing.assert_array_equal(cube[0, 0], mosaic[:4, :4].reshape(-1))
    np.testing.assert_array_equal(cube[1, 2], mosaic[4:8, 8:12].reshape(-1))


def test_x2cube_rejects_non_divisible_shape() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        x2cube(np.zeros((7, 8), dtype=np.uint16))


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
