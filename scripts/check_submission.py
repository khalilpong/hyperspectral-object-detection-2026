from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

from hsi_detection.submission import validate_submission_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a submission before uploading")
    parser.add_argument("submission", type=Path)
    parser.add_argument("--images", type=Path, default=Path("data/processed/pseudo_rgb/images/test"))
    parser.add_argument("--class-count", type=int, default=18)
    args = parser.parse_args()

    image_sizes: dict[int, tuple[int, int]] = {}
    for image_path in args.images.glob("*.png"):
        with Image.open(image_path) as image:
            image_sizes[int(image_path.stem)] = image.size
    frame = pd.read_csv(args.submission)
    issues = validate_submission_frame(frame, image_sizes, args.class_count)
    if issues:
        preview = "\n".join(f"- {issue}" for issue in issues[:50])
        remainder = len(issues) - 50
        if remainder > 0:
            preview += f"\n- ... and {remainder} more"
        raise SystemExit(f"Submission validation failed with {len(issues)} issue(s):\n{preview}")
    print(
        f"valid submission: {len(frame)} detections across "
        f"{frame['image_id'].nunique()} images"
    )


if __name__ == "__main__":
    main()

