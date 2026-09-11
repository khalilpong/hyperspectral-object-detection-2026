from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from hsi_detection.annotations import read_voc_annotation
from hsi_detection.spectral import make_pseudo_rgb, x2cube


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a VOC annotation over pseudo-RGB")
    parser.add_argument("image", type=Path)
    parser.add_argument("annotation", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/annotation_preview.png"))
    parser.add_argument("--bands", type=int, nargs=3, default=(5, 8, 13))
    args = parser.parse_args()

    with Image.open(args.image) as image:
        mosaic = np.asarray(image)
    rendered = Image.fromarray(make_pseudo_rgb(x2cube(mosaic), bands=args.bands), mode="RGB")
    annotation = read_voc_annotation(args.annotation)
    if rendered.size != (annotation.width, annotation.height):
        raise ValueError(
            f"Image cube size {rendered.size} does not match XML size "
            f"{(annotation.width, annotation.height)}"
        )

    draw = ImageDraw.Draw(rendered)
    for obj in annotation.objects:
        box = (obj.xmin, obj.ymin, obj.xmax, obj.ymax)
        draw.rectangle(box, outline=(255, 40, 40), width=3)
        text_box = draw.textbbox((obj.xmin, obj.ymin), obj.name)
        draw.rectangle(text_box, fill=(255, 40, 40))
        draw.text((obj.xmin, obj.ymin), obj.name, fill=(255, 255, 255))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()

