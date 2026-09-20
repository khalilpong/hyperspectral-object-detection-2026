from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class LabeledBox:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(self.x2 - self.x1, 0.0) * max(self.y2 - self.y1, 0.0)


@dataclass(frozen=True)
class TileWindow:
    x0: int
    y0: int
    x1: int
    y1: int
    core_x0: float
    core_y0: float
    core_x1: float
    core_y1: float
    image_width: int
    image_height: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def axis_starts(length: int, tile_size: int, stride: int) -> tuple[int, ...]:
    """Return deterministic starts that cover an axis and include its far edge."""
    if length <= 0 or tile_size <= 0 or stride <= 0:
        raise ValueError("length, tile_size, and stride must be positive")
    if stride > tile_size:
        raise ValueError("stride must not exceed tile_size or coverage would have gaps")
    if length <= tile_size:
        return (0,)
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return tuple(starts)


def _core_ranges(
    starts: tuple[int, ...], tile_size: int, length: int
) -> tuple[tuple[float, float], ...]:
    centers = [start + min(tile_size, length - start) / 2.0 for start in starts]
    boundaries = [(left + right) / 2.0 for left, right in zip(centers, centers[1:])]
    return tuple(
        (
            0.0 if index == 0 else boundaries[index - 1],
            float(length) if index == len(starts) - 1 else boundaries[index],
        )
        for index in range(len(starts))
    )


def tile_windows(
    image_height: int,
    image_width: int,
    *,
    tile_height: int,
    tile_width: int,
    stride_height: int,
    stride_width: int,
) -> tuple[TileWindow, ...]:
    """Create overlapping tiles with non-overlapping center-ownership cores."""
    y_starts = axis_starts(image_height, tile_height, stride_height)
    x_starts = axis_starts(image_width, tile_width, stride_width)
    y_cores = _core_ranges(y_starts, tile_height, image_height)
    x_cores = _core_ranges(x_starts, tile_width, image_width)
    return tuple(
        TileWindow(
            x0=x0,
            y0=y0,
            x1=min(x0 + tile_width, image_width),
            y1=min(y0 + tile_height, image_height),
            core_x0=x_cores[x_index][0],
            core_y0=y_cores[y_index][0],
            core_x1=x_cores[x_index][1],
            core_y1=y_cores[y_index][1],
            image_width=image_width,
            image_height=image_height,
        )
        for y_index, y0 in enumerate(y_starts)
        for x_index, x0 in enumerate(x_starts)
    )


def map_tile_boxes_to_image(
    boxes: np.ndarray,
    window: TileWindow,
) -> tuple[np.ndarray, np.ndarray]:
    """Map tile-local boxes to the image and keep centers owned by this tile's core."""
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1:] != (4,):
        raise ValueError(f"Expected boxes with shape N x 4, got {boxes.shape}")
    if not len(boxes):
        return boxes.copy(), np.empty(0, dtype=bool)

    mapped = boxes.copy()
    mapped[:, (0, 2)] += window.x0
    mapped[:, (1, 3)] += window.y0
    mapped[:, (0, 2)] = np.clip(mapped[:, (0, 2)], 0.0, float(window.image_width))
    mapped[:, (1, 3)] = np.clip(mapped[:, (1, 3)], 0.0, float(window.image_height))
    centers_x = (mapped[:, 0] + mapped[:, 2]) / 2.0
    centers_y = (mapped[:, 1] + mapped[:, 3]) / 2.0
    x_upper = (
        centers_x <= window.core_x1
        if window.core_x1 == window.image_width
        else centers_x < window.core_x1
    )
    y_upper = (
        centers_y <= window.core_y1
        if window.core_y1 == window.image_height
        else centers_y < window.core_y1
    )
    keep = (
        np.isfinite(mapped).all(axis=1)
        & (mapped[:, 2] > mapped[:, 0])
        & (mapped[:, 3] > mapped[:, 1])
        & (centers_x >= window.core_x0)
        & x_upper
        & (centers_y >= window.core_y0)
        & y_upper
    )
    return mapped, keep


def parse_yolo_rows(text: str, *, width: int, height: int) -> list[LabeledBox]:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    boxes: list[LabeledBox] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"Invalid YOLO row {line_number}: {line!r}")
        class_text, center_x_text, center_y_text, box_width_text, box_height_text = values
        class_value = float(class_text)
        if not class_value.is_integer() or class_value < 0:
            raise ValueError(f"Invalid class ID on YOLO row {line_number}: {class_text!r}")
        center_x, center_y, box_width, box_height = map(
            float, (center_x_text, center_y_text, box_width_text, box_height_text)
        )
        if not all(
            math.isfinite(value) for value in (center_x, center_y, box_width, box_height)
        ):
            raise ValueError(f"Non-finite value on YOLO row {line_number}")
        x1 = (center_x - box_width / 2.0) * width
        y1 = (center_y - box_height / 2.0) * height
        x2 = (center_x + box_width / 2.0) * width
        y2 = (center_y + box_height / 2.0) * height
        tolerance = 1e-3
        if not (
            -tolerance <= x1 < x2 <= width + tolerance
            and -tolerance <= y1 < y2 <= height + tolerance
        ):
            raise ValueError(f"Out-of-bounds box on YOLO row {line_number}: {line!r}")
        x1 = float(np.clip(x1, 0.0, float(width)))
        y1 = float(np.clip(y1, 0.0, float(height)))
        x2 = float(np.clip(x2, 0.0, float(width)))
        y2 = float(np.clip(y2, 0.0, float(height)))
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"Zero-area box on YOLO row {line_number}: {line!r}")
        boxes.append(LabeledBox(int(class_value), x1, y1, x2, y2))
    return boxes


def crop_labeled_boxes(
    boxes: Iterable[LabeledBox],
    *,
    x0: int,
    y0: int,
    crop_width: int,
    crop_height: int,
    minimum_visible: float,
) -> list[LabeledBox]:
    """Clip labels to a crop, retaining only boxes that remain sufficiently visible."""
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError("crop_width and crop_height must be positive")
    if not 0.0 < minimum_visible <= 1.0:
        raise ValueError("minimum_visible must be in (0, 1]")
    cropped: list[LabeledBox] = []
    x1_limit = x0 + crop_width
    y1_limit = y0 + crop_height
    for box in boxes:
        intersection_x1 = max(box.x1, float(x0))
        intersection_y1 = max(box.y1, float(y0))
        intersection_x2 = min(box.x2, float(x1_limit))
        intersection_y2 = min(box.y2, float(y1_limit))
        intersection_area = max(intersection_x2 - intersection_x1, 0.0) * max(
            intersection_y2 - intersection_y1, 0.0
        )
        if box.area <= 0.0 or intersection_area / box.area < minimum_visible:
            continue
        cropped.append(
            LabeledBox(
                class_id=box.class_id,
                x1=intersection_x1 - x0,
                y1=intersection_y1 - y0,
                x2=intersection_x2 - x0,
                y2=intersection_y2 - y0,
            )
        )
    return cropped


def to_yolo_rows(boxes: Iterable[LabeledBox], *, width: int, height: int) -> list[str]:
    rows: list[str] = []
    for box in boxes:
        if not (0 <= box.x1 < box.x2 <= width and 0 <= box.y1 < box.y2 <= height):
            raise ValueError(f"Box lies outside {width}x{height}: {box}")
        center_x = (box.x1 + box.x2) / (2.0 * width)
        center_y = (box.y1 + box.y2) / (2.0 * height)
        box_width = (box.x2 - box.x1) / width
        box_height = (box.y2 - box.y1) / height
        rows.append(
            f"{box.class_id} {center_x:.8f} {center_y:.8f} {box_width:.8f} {box_height:.8f}"
        )
    return rows
