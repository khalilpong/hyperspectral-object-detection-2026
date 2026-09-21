from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _band_indices(channel_count: int, band_order: Sequence[int] | None) -> tuple[int, ...]:
    if band_order is None:
        bands = tuple(range(channel_count))
    else:
        bands = tuple(int(band) for band in band_order)
    if not bands:
        raise ValueError("At least one band is required")
    if len(bands) != len(set(bands)):
        raise ValueError("band_order must not contain duplicates")
    if min(bands) < 0 or max(bands) >= channel_count:
        raise ValueError(f"Band indices {bands} are invalid for {channel_count} bands")
    return bands


def x2cube(image: np.ndarray, cell_size: int = 4) -> np.ndarray:
    """Unpack a snapshot mosaic into an H x W x (cell_size**2) cube.

    The band order is row-major within each mosaic cell, matching the official
    ``pseudo_rgb_demo.py`` implementation supplied with the competition.
    """
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D mosaic, got shape {image.shape}")
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")
    height, width = image.shape
    if height % cell_size or width % cell_size:
        raise ValueError(
            f"Mosaic shape {image.shape} is not divisible by cell_size={cell_size}"
        )
    return (
        image.reshape(height // cell_size, cell_size, width // cell_size, cell_size)
        .transpose(0, 2, 1, 3)
        .reshape(height // cell_size, width // cell_size, cell_size * cell_size)
    )


def phase_aware_target_shape(
    source_shape: Sequence[int],
    target_long_edge: int = 1024,
) -> tuple[int, int]:
    """Return an aspect-preserving target shape with one edge fixed.

    The phase-aware dataset is reconstructed close to the network's actual
    pre-letterbox size instead of materializing a full-resolution 16-channel
    cube.  Keeping the source aspect ratio also keeps normalized YOLO labels
    valid without per-image label rewrites.
    """
    if len(source_shape) != 2:
        raise ValueError(f"Expected a 2-D source shape, got {tuple(source_shape)}")
    height, width = (int(value) for value in source_shape)
    if height <= 0 or width <= 0:
        raise ValueError(f"Source dimensions must be positive, got {(height, width)}")
    if target_long_edge <= 0:
        raise ValueError("target_long_edge must be positive")

    scale = float(target_long_edge) / max(height, width)
    target_height = max(1, int(round(height * scale)))
    target_width = max(1, int(round(width * scale)))
    return target_height, target_width


def phase_aware_reconstruct(
    image: np.ndarray,
    target_shape: Sequence[int],
    band_order: Sequence[int] | None = None,
    cell_size: int = 4,
) -> np.ndarray:
    """Reconstruct mosaic bands on a shared canvas using their true phases.

    ``x2cube`` puts every physical band on the same compact H/4 x W/4 grid.
    A plain resize therefore incorrectly treats the 16 band samples as
    co-located.  Here physical band ``b`` uses phase
    ``(b // cell_size, b % cell_size)``.  Target pixel centres are mapped back
    to raw-mosaic coordinates with the ``align_corners=False`` convention and
    then into that band's native lattice.  Bilinear interpolation uses
    replicate borders.

    This is geometry-aware interpolation, not super-resolution: it preserves
    measured sample locations but cannot recover unmeasured high frequencies.
    """
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D mosaic, got shape {image.shape}")
    if np.issubdtype(image.dtype, np.floating) and not np.isfinite(image).all():
        raise ValueError("Mosaic must contain only finite values")
    if len(target_shape) != 2:
        raise ValueError(f"Expected a 2-D target shape, got {tuple(target_shape)}")
    target_height, target_width = (int(value) for value in target_shape)
    if target_height <= 0 or target_width <= 0:
        raise ValueError(
            f"Target dimensions must be positive, got {(target_height, target_width)}"
        )

    cube = x2cube(image, cell_size=cell_size)
    bands = _band_indices(cube.shape[2], band_order)
    source_height, source_width = image.shape
    band_height, band_width = cube.shape[:2]

    # Pixel-centre mapping, equivalent to align_corners=False.
    y_raw = (
        (np.arange(target_height, dtype=np.float32) + np.float32(0.5))
        * np.float32(source_height / target_height)
        - np.float32(0.5)
    )
    x_raw = (
        (np.arange(target_width, dtype=np.float32) + np.float32(0.5))
        * np.float32(source_width / target_width)
        - np.float32(0.5)
    )
    output = np.empty((target_height, target_width, len(bands)), dtype=np.float32)

    for output_channel, physical_band in enumerate(bands):
        phase_row, phase_column = divmod(physical_band, cell_size)
        y_band = np.clip(
            (y_raw - np.float32(phase_row)) / np.float32(cell_size),
            0.0,
            float(band_height - 1),
        )
        x_band = np.clip(
            (x_raw - np.float32(phase_column)) / np.float32(cell_size),
            0.0,
            float(band_width - 1),
        )

        y0 = np.floor(y_band).astype(np.intp)
        x0 = np.floor(x_band).astype(np.intp)
        y1 = np.minimum(y0 + 1, band_height - 1)
        x1 = np.minimum(x0 + 1, band_width - 1)
        weight_y = (y_band - y0).astype(np.float32)[:, None]
        weight_x = (x_band - x0).astype(np.float32)[None, :]

        source = cube[:, :, physical_band].astype(np.float32, copy=False)
        top_left = source[y0[:, None], x0[None, :]]
        top_right = source[y0[:, None], x1[None, :]]
        bottom_left = source[y1[:, None], x0[None, :]]
        bottom_right = source[y1[:, None], x1[None, :]]
        output[:, :, output_channel] = (
            (1.0 - weight_y) * (1.0 - weight_x) * top_left
            + (1.0 - weight_y) * weight_x * top_right
            + weight_y * (1.0 - weight_x) * bottom_left
            + weight_y * weight_x * bottom_right
        )

    return output


def make_pseudo_rgb(
    cube: np.ndarray,
    bands: Sequence[int] = (5, 8, 13),
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> np.ndarray:
    """Select and robustly stretch three bands to an 8-bit pseudo-RGB image."""
    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError(f"Expected an H x W x C cube, got shape {cube.shape}")
    bands = tuple(int(band) for band in bands)
    if len(bands) != 3:
        raise ValueError("Exactly three band indices are required")
    if min(bands) < 0 or max(bands) >= cube.shape[2]:
        raise ValueError(f"Band indices {bands} are invalid for {cube.shape[2]} bands")
    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100")

    selected = cube[:, :, bands].astype(np.float32, copy=False)
    output = np.zeros_like(selected, dtype=np.uint8)
    for channel in range(3):
        values = selected[:, :, channel]
        low, high = np.percentile(values, [lower_percentile, upper_percentile])
        if high > low:
            stretched = np.clip((values - low) / (high - low), 0.0, 1.0)
            output[:, :, channel] = np.rint(stretched * 255.0).astype(np.uint8)
    return output


def multispectral_percentile_bounds(
    cube: np.ndarray,
    band_order: Sequence[int] | None = None,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> tuple[float, float]:
    """Return one robust low/high pair shared by the requested bands."""
    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError(f"Expected an H x W x C cube, got shape {cube.shape}")
    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100")
    bands = _band_indices(cube.shape[2], band_order)
    selected = cube[:, :, bands].astype(np.float32, copy=False)
    low, high = np.percentile(selected, [lower_percentile, upper_percentile])
    return float(low), float(high)


def encode_multispectral_uint8(
    cube: np.ndarray,
    low: float,
    high: float,
    band_order: Sequence[int] | None = None,
) -> np.ndarray:
    """Apply a supplied shared affine scale and return an H x W x C uint8 cube."""
    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError(f"Expected an H x W x C cube, got shape {cube.shape}")
    bands = _band_indices(cube.shape[2], band_order)
    selected = cube[:, :, bands].astype(np.float32, copy=False)
    if not np.isfinite(selected).all():
        raise ValueError("Cube must contain only finite values")
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("Encoding bounds must be finite")
    if high <= low:
        return np.zeros_like(selected, dtype=np.uint8)
    stretched = np.clip((selected - low) / (high - low), 0.0, 1.0)
    return np.rint(stretched * 255.0).astype(np.uint8)


def make_multispectral_uint8(
    cube: np.ndarray,
    band_order: Sequence[int] | None = None,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> np.ndarray:
    """Scale a cube with one shared affine transform to preserve band relationships.

    Unlike :func:`make_pseudo_rgb`, this function deliberately uses one low/high
    pair for all selected channels. This keeps relative cross-band intensity
    information available to a multispectral model while robustly normalizing
    each capture's overall exposure.
    """
    low, high = multispectral_percentile_bounds(
        cube,
        band_order=band_order,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )
    return encode_multispectral_uint8(cube, low=low, high=high, band_order=band_order)
