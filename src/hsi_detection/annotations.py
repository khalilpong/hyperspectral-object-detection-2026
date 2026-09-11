from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class VocObject:
    name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass(frozen=True)
class VocAnnotation:
    filename: str
    width: int
    height: int
    depth: int
    objects: tuple[VocObject, ...]


@dataclass(frozen=True)
class AnnotationSanitization:
    annotation: VocAnnotation
    clipped_objects: int
    dropped_objects: int


def _required_text(parent: ET.Element, path: str) -> str:
    value = parent.findtext(path)
    if value is None:
        raise ValueError(f"Missing XML field: {path}")
    return value.strip()


def read_voc_annotation(path: str | Path) -> VocAnnotation:
    root = ET.parse(path).getroot()
    objects: list[VocObject] = []
    for item in root.findall("object"):
        box = item.find("bndbox")
        if box is None:
            raise ValueError(f"Missing bndbox in {path}")
        objects.append(
            VocObject(
                name=_required_text(item, "name"),
                xmin=float(_required_text(box, "xmin")),
                ymin=float(_required_text(box, "ymin")),
                xmax=float(_required_text(box, "xmax")),
                ymax=float(_required_text(box, "ymax")),
            )
        )
    return VocAnnotation(
        filename=_required_text(root, "filename"),
        width=int(_required_text(root, "size/width")),
        height=int(_required_text(root, "size/height")),
        depth=int(_required_text(root, "size/depth")),
        objects=tuple(objects),
    )


def sanitize_annotation(annotation: VocAnnotation) -> AnnotationSanitization:
    """Clip boxes to the image and drop boxes with no area, without changing the raw XML."""
    objects: list[VocObject] = []
    clipped_objects = 0
    dropped_objects = 0
    for obj in annotation.objects:
        clipped = VocObject(
            name=obj.name,
            xmin=min(max(obj.xmin, 0.0), float(annotation.width)),
            ymin=min(max(obj.ymin, 0.0), float(annotation.height)),
            xmax=min(max(obj.xmax, 0.0), float(annotation.width)),
            ymax=min(max(obj.ymax, 0.0), float(annotation.height)),
        )
        if clipped.xmin >= clipped.xmax or clipped.ymin >= clipped.ymax:
            dropped_objects += 1
            continue
        if clipped != obj:
            clipped_objects += 1
        objects.append(clipped)
    return AnnotationSanitization(
        annotation=VocAnnotation(
            filename=annotation.filename,
            width=annotation.width,
            height=annotation.height,
            depth=annotation.depth,
            objects=tuple(objects),
        ),
        clipped_objects=clipped_objects,
        dropped_objects=dropped_objects,
    )


def to_yolo_rows(annotation: VocAnnotation, class_to_id: dict[str, int]) -> list[str]:
    rows: list[str] = []
    for obj in annotation.objects:
        if obj.name not in class_to_id:
            raise ValueError(f"Unknown class {obj.name!r} in {annotation.filename}")
        if not (0 <= obj.xmin < obj.xmax <= annotation.width):
            raise ValueError(f"Invalid x coordinates for {annotation.filename}: {obj}")
        if not (0 <= obj.ymin < obj.ymax <= annotation.height):
            raise ValueError(f"Invalid y coordinates for {annotation.filename}: {obj}")
        center_x = (obj.xmin + obj.xmax) / (2.0 * annotation.width)
        center_y = (obj.ymin + obj.ymax) / (2.0 * annotation.height)
        width = (obj.xmax - obj.xmin) / annotation.width
        height = (obj.ymax - obj.ymin) / annotation.height
        rows.append(
            f"{class_to_id[obj.name]} {center_x:.8f} {center_y:.8f} "
            f"{width:.8f} {height:.8f}"
        )
    return rows
