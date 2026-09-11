from __future__ import annotations

from collections.abc import Sequence

import numpy as np


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

