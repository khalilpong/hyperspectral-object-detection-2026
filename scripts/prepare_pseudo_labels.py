"""用现有最佳模型给测试集打伪标签，扩充训练集。

=============================================================================
为什么伪标签可能有效
=============================================================================
1. 训练数据从 2400 张扩到 ~3100 张（+30%）
2. 新增数据**就是测试集本身**，分布与最终评测完全一致
3. 比赛规则允许：没有引入任何外部数据

=============================================================================
最大的风险，以及怎么规避
=============================================================================
伪标签里**没标出来的区域会被模型当成"背景"**。如果一张图里某个真实目标
只被检测到 0.3 分、没进伪标签，就等于教模型"这里没有东西"——这是负监督，
会直接损害召回。

规避方法是**按整张图过滤，而不是按单个框过滤**：

  高置信框   conf >= 0.5          -> 写入伪标签
  模糊框     0.1 <= conf < 0.5    -> 若与某个高置信框 IoU>=0.5，视为同一目标的重复框，忽略；
                                      否则是"可能漏掉的目标"，**整张图丢弃**
  噪声框     conf < 0.1           -> 视为背景（实测每图约 144 个，几乎全是误检）

只保留"至少一个高置信框、且没有任何独立模糊框"的图。

实测模型置信度呈清晰双峰：高置信框每图 3.67 个，与训练集每图 3.37 个真实目标
几乎吻合，说明 0.5 阈值基本卡准了真实目标数。

=============================================================================
关于验证集是否被污染
=============================================================================
伪标签来自全量模型（训练时见过验证集）。但验证集的标注**从未进入**新模型的训练，
伪标签只打在测试图上。教师模型更好只是让学生整体更好，不构成针对验证集的泄漏，
所以验证集分数仍可作为泛化能力的公平估计。

图像文件用**硬链接**，不占用额外磁盘空间。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import yaml


def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-9)


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True, help="提交格式的预测 CSV")
    parser.add_argument("--reference", type=Path, default=Path("data/processed/hsi16_shared_p005_995"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/hsi16_pseudo"))
    parser.add_argument("--high", type=float, default=0.5, help="高置信阈值：写入伪标签")
    parser.add_argument("--ambiguous", type=float, default=0.1, help="模糊下限：低于此视为背景")
    parser.add_argument("--dup-iou", type=float, default=0.5, help="模糊框与高置信框重叠达此值视为重复")
    args = parser.parse_args()

    ref = args.reference
    frame = pd.read_csv(args.predictions)
    test_dir = ref / "images" / "test"

    kept, dropped_ambiguous, dropped_empty, boxes_written = [], 0, 0, 0
    label_dir = args.output / "labels" / "pseudo"
    label_dir.mkdir(parents=True, exist_ok=True)

    for image_id, group in frame.groupby("image_id"):
        high = group[group.confidence >= args.high]
        if high.empty:
            dropped_empty += 1
            continue
        high_boxes = high[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
        ambiguous = group[(group.confidence >= args.ambiguous) & (group.confidence < args.high)]
        independent = 0
        for row in ambiguous[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float):
            if iou_one_to_many(row, high_boxes).max() < args.dup_iou:
                independent += 1
        if independent > 0:
            dropped_ambiguous += 1
            continue

        npy = test_dir / f"{image_id}.npy"
        height, width = np.load(npy, mmap_mode="r").shape[:2]
        lines = []
        for _, r in high.iterrows():
            x1, y1, x2, y2 = (float(r.x1), float(r.y1), float(r.x2), float(r.y2))
            if x2 <= x1 or y2 <= y1:
                continue
            cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
            w, h = (x2 - x1) / width, (y2 - y1) / height
            lines.append(f"{int(r.class_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        if not lines:
            dropped_empty += 1
            continue
        (label_dir / f"{image_id}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        boxes_written += len(lines)
        kept.append(image_id)

    for image_id in kept:
        for suffix in (".npy", ".png"):
            source = test_dir / f"{image_id}{suffix}"
            if source.exists():
                link_or_copy(source, args.output / "images" / "pseudo" / source.name)

    for split in ("train", "val"):
        for source in (ref / "images" / split).iterdir():
            link_or_copy(source, args.output / "images" / split / source.name)
        destination_labels = args.output / "labels" / split
        destination_labels.mkdir(parents=True, exist_ok=True)
        for source in (ref / "labels" / split).glob("*.txt"):
            shutil.copy2(source, destination_labels / source.name)

    reference_yaml = yaml.safe_load((ref / "dataset.yaml").read_text(encoding="utf-8"))
    dataset = dict(reference_yaml)
    dataset["path"] = str(args.output.resolve()).replace("\\", "/")
    dataset["train"] = ["images/train", "images/pseudo"]
    dataset["val"] = "images/val"
    dataset["training_scope"] = f"2400_train_plus_{len(kept)}_pseudo_test_600_val"
    dataset["pseudo_label_source"] = str(args.predictions).replace("\\", "/")
    dataset["pseudo_label_rule"] = (
        f"conf>={args.high} as labels; drop whole image if any {args.ambiguous}<=conf<{args.high} "
        f"box has IoU<{args.dup_iou} with every high box"
    )
    (args.output / "dataset.yaml").write_text(
        yaml.safe_dump(dataset, allow_unicode=True, sort_keys=False), encoding="utf-8")

    total = frame.image_id.nunique()
    print(f"测试图总数          {total}")
    print(f"保留为伪标签        {len(kept)}  ({boxes_written} 个框，每图 {boxes_written/max(1,len(kept)):.2f} 个)")
    print(f"丢弃：含独立模糊框  {dropped_ambiguous}")
    print(f"丢弃：无高置信框    {dropped_empty}")
    print(f"训练集规模          2400 真实 + {len(kept)} 伪标签 = {2400 + len(kept)}")
    print(f"已写入 {args.output / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
