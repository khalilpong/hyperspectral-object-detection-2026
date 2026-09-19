"""把 4x4 快照式马赛克做成全分辨率全色(pan)图像。

=============================================================================
为什么要这么做
=============================================================================
现有的 x2cube 把 4x4 马赛克块直接拆成 16 个通道：964x1972 -> 241x493x16。
每个波段只保留了 1/16 的空间采样，**94% 的空间信息被丢弃**。

但传感器的 964x1972 个感光点，每一个都是对场景的一次独立空间采样，
只是各自透过了不同的滤光片 —— 这和相机的 Bayer 阵列是同一回事。
处理 Bayer 的标准做法是去马赛克，而不是抽掉 3/4 的像素。

本脚本用**相位增益校正**恢复全分辨率：
  1. 4x4 网格里的 16 个相位各自有固定的滤光片透过率（实测均值差 3.39 倍）
  2. 统计每个相位在整幅图上的均值，把它们归一化到同一水平
  3. 校正后棋盘纹消失，剩下的就是全分辨率的场景结构

目标中位框在解包图里只有约 25x25 像素，在全分辨率下是 100x100 —— 对
mAP@[0.5:0.95] 这种重罚定位误差的指标，差别很大。

标注不用改：YOLO 格式是归一化坐标(0~1)，与分辨率无关，直接复用。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import yaml
from tqdm import tqdm

CELL = 4


def phase_corrected_pan(raw: np.ndarray, cell: int = CELL) -> np.ndarray:
    """按相位增益校正，返回与输入同尺寸的全分辨率 float32 图。"""
    height, width = raw.shape
    if height % cell or width % cell:
        raise ValueError(f"马赛克尺寸 {raw.shape} 不能被 cell={cell} 整除")
    blocks = raw.reshape(height // cell, cell, width // cell, cell)
    # 每个相位在整幅图上的均值
    phase_mean = blocks.transpose(1, 3, 0, 2).reshape(cell * cell, -1).mean(axis=1)
    phase_mean = np.maximum(phase_mean, 1e-6)
    gain = (phase_mean / phase_mean.mean()).reshape(cell, cell, 1, 1)
    corrected = blocks.transpose(1, 3, 0, 2) / gain
    return corrected.transpose(2, 0, 3, 1).reshape(height, width)


def encode_uint8(pan: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    low, high = np.percentile(pan, [low_pct, high_pct])
    if high <= low:
        high = low + 1.0
    scaled = np.clip((pan - low) / (high - low), 0.0, 1.0)
    return (scaled * 255.0 + 0.5).astype(np.uint8)


def _process(task: tuple[str, str, float, float]) -> bool:
    source, destination, low_pct, high_pct = task
    destination_path = Path(destination)
    if destination_path.exists():
        return True
    with Image.open(source) as image:
        raw = np.asarray(image).astype(np.float32)
    pan = phase_corrected_pan(raw)
    encoded = encode_uint8(pan, low_pct, high_pct)
    rgb = np.repeat(encoded[:, :, None], 3, axis=2)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_suffix(".png.tmp")
    Image.fromarray(rgb, mode="RGB").save(temporary, format="PNG", compress_level=3)
    temporary.replace(destination_path)
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path,
                        default=Path("data/processed/hsi16_shared_p005_995"),
                        help="复用这个数据集的划分与标注，保证 A/B 干净")
    parser.add_argument("--raw-train", type=Path,
                        default=Path("data/raw/extracted/data_train"))
    parser.add_argument("--raw-test", type=Path,
                        default=Path("data/raw/extracted/data_test"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/pan_fullres"))
    parser.add_argument("--lower-percentile", type=float, default=0.5)
    parser.add_argument("--upper-percentile", type=float, default=99.5)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    raw_lookup: dict[str, Path] = {}
    for root in (args.raw_train, args.raw_test):
        for path in root.rglob("*.png"):
            raw_lookup[path.stem] = path
    if not raw_lookup:
        raise FileNotFoundError("没有找到任何原始 PNG")
    print(f"原始图像索引: {len(raw_lookup)} 张")

    tasks: list[tuple[str, str, float, float]] = []
    for split in ("train", "val", "test"):
        reference_images = args.reference / "images" / split
        if not reference_images.is_dir():
            continue
        stems = sorted({p.stem for p in reference_images.glob("*.npy")})
        if not stems:
            stems = sorted({p.stem for p in reference_images.glob("*.png")})
        missing = [s for s in stems if s not in raw_lookup]
        if missing:
            raise KeyError(f"{split} 缺少原图: {missing[:5]}")
        for stem in stems:
            tasks.append((
                str(raw_lookup[stem]),
                str(args.output / "images" / split / f"{stem}.png"),
                args.lower_percentile,
                args.upper_percentile,
            ))
        # 标注直接复用：YOLO 是归一化坐标，与分辨率无关
        source_labels = args.reference / "labels" / split
        if source_labels.is_dir():
            destination_labels = args.output / "labels" / split
            destination_labels.mkdir(parents=True, exist_ok=True)
            for label in source_labels.glob("*.txt"):
                shutil.copy2(label, destination_labels / label.name)
        print(f"{split}: {len(stems)} 张")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        skipped = sum(tqdm(pool.map(_process, tasks, chunksize=8),
                           total=len(tasks), desc="生成 pan 图"))
    print(f"完成，其中已存在跳过 {skipped} 张")

    reference_yaml = yaml.safe_load((args.reference / "dataset.yaml").read_text(encoding="utf-8"))
    dataset = {
        "path": str(args.output.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": reference_yaml["names"],
        "encoding": (
            f"full-resolution phase-gain-corrected pan from 4x4 snapshot mosaic; "
            f"P{args.lower_percentile}-P{args.upper_percentile} uint8; replicated to 3 channels"
        ),
        "split_seed": reference_yaml.get("split_seed"),
        "training_scope": reference_yaml.get("training_scope"),
    }
    (args.output / "dataset.yaml").write_text(
        yaml.safe_dump(dataset, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"数据集配置已写入 {args.output / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
