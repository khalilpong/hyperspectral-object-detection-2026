from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoxCalibration:
    """Global box scale and center shift relative to each box's own size."""

    width_scale: float = 1.0
    height_scale: float = 1.0
    center_x_shift: float = 0.0
    center_y_shift: float = 0.0

    @property
    def distance_from_identity(self) -> float:
        return (
            abs(self.width_scale - 1.0)
            + abs(self.height_scale - 1.0)
            + abs(self.center_x_shift)
            + abs(self.center_y_shift)
        )


IDENTITY_BOX_CALIBRATION = BoxCalibration()


def box_calibration_name(calibration: BoxCalibration) -> str:
    return (
        f"ws{calibration.width_scale:.3f}_hs{calibration.height_scale:.3f}_"
        f"dx{calibration.center_x_shift:+.3f}_dy{calibration.center_y_shift:+.3f}"
    )


def calibrate_xyxy(
    boxes: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    calibration: BoxCalibration,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one calibration and return clipped boxes plus their valid mask."""

    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1:] != (4,):
        raise ValueError(f"Expected boxes with shape N x 4, got {boxes.shape}")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")
    parameters = (
        calibration.width_scale,
        calibration.height_scale,
        calibration.center_x_shift,
        calibration.center_y_shift,
    )
    if not all(np.isfinite(value) for value in parameters):
        raise ValueError("Every box calibration parameter must be finite")
    if calibration.width_scale <= 0.0 or calibration.height_scale <= 0.0:
        raise ValueError("Box calibration scales must be positive")

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    center_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
    center_y = (boxes[:, 1] + boxes[:, 3]) / 2.0
    center_x += widths * calibration.center_x_shift
    center_y += heights * calibration.center_y_shift
    half_width = widths * calibration.width_scale / 2.0
    half_height = heights * calibration.height_scale / 2.0
    calibrated = np.stack(
        (
            center_x - half_width,
            center_y - half_height,
            center_x + half_width,
            center_y + half_height,
        ),
        axis=1,
    ).astype(np.float32, copy=False)
    calibrated[:, (0, 2)] = np.clip(
        calibrated[:, (0, 2)], 0.0, float(image_width)
    )
    calibrated[:, (1, 3)] = np.clip(
        calibrated[:, (1, 3)], 0.0, float(image_height)
    )
    valid = (
        np.isfinite(calibrated).all(axis=1)
        & (calibrated[:, 2] > calibrated[:, 0])
        & (calibrated[:, 3] > calibrated[:, 1])
    )
    return calibrated, valid
