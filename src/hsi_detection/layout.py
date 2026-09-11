from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetLayout:
    root: Path
    class_file: Path
    train_images: Path
    train_annotations: Path
    test_images: Path


def _one(candidates: list[Path], description: str) -> Path:
    if len(candidates) != 1:
        rendered = ", ".join(str(path) for path in candidates) or "none"
        raise FileNotFoundError(f"Expected one {description}, found: {rendered}")
    return candidates[0]


def discover_layout(root: str | Path) -> DatasetLayout:
    root = Path(root).resolve()
    class_file = _one(list(root.rglob("class.txt")), "class.txt")
    vis_dirs = [path for path in root.rglob("VIS") if path.is_dir()]
    train_annotations = _one(
        [path for path in vis_dirs if "Annotations" in path.parts],
        "training annotation VIS directory",
    )
    train_images = _one(
        [
            path
            for path in vis_dirs
            if "data_train" in path.parts and "Annotations" not in path.parts
        ],
        "training image VIS directory",
    )
    test_images = _one(
        [path for path in vis_dirs if "data_test" in path.parts],
        "test image VIS directory",
    )
    return DatasetLayout(root, class_file, train_images, train_annotations, test_images)


def read_classes(path: str | Path) -> list[str]:
    classes = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    classes = [name for name in classes if name]
    if len(classes) != len(set(classes)):
        raise ValueError("class.txt contains duplicate names")
    return classes

