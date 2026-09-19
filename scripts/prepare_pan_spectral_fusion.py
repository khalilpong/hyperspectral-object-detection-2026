"""pan + 光谱融合数据集：同时保留全分辨率空间细节和 16 波段光谱信息。

=============================================================================
为什么做融合
=============================================================================
两条已有路线各丢一半信息：

  HSI16（当前最佳）  16 个光谱波段，但每个只有 241x493（丢了 94% 空间采样）
  pan               全分辨率 964x1972，但把 16 个波段混成一个亮度值（丢了光谱）

融合把两者叠在一起喂给模型：一个全分辨率 pan 通道 + 16 个光谱波段。

=============================================================================
通道顺序为什么重要
=============================================================================
Ultralytics 只把预训练权重迁移到**前 3 个输入通道**，其余通道随机初始化。
预训练卷积核是在 RGB 自然图像上学的边缘/纹理检测器，放进前 3 槽位的信号
能直接享受预训练红利。因此前 3 槽位应该放**实测更强**的那种表示：

  --order pan_first    前3 = pan x3，之后 16 个波段      共 19 通道
  --order bands_first  前3 = 波段 5/8/13（与 HSI16 一致），之后 13 个波段 + pan   共 17 通道

用哪个取决于 pan 单独实验和 HSI16 谁更强。

=============================================================================
分辨率取舍
=============================================================================
训练时 imgsz=1024 会把长边缩放到 1024，实际输入约 502x1024。
所以存成 482x986（光谱立方体 241x493 的 2 倍）就能覆盖模型真正看到的细节，
而不必存 964x1972 全尺寸 —— 后者 4000 张会超过 120GB。

  pan    从 964x1972 下采样到 482x986（区域平均，保留真实细节）
  波段   从 241x493 双线性上采样到 482x986（与 pan 对齐）

标注直接复用：YOLO 归一化坐标与分辨率无关。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from PIL import Image
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_pan_fullres import phase_corrected_pan  # noqa: E402

CELL = 4
HSI16_BAND_ORDER = (5, 8, 13, 0, 1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15)


def unpack_cube(raw: np.ndarray, cell: int = CELL) -> np.ndarray:
    """与 hsi_detection.spectral.x2cube 完全一致的行优先解包。"""
    height, width = raw.shape
    return (
        raw.reshape(height // cell, cell, width // cell, cell)
        .transpose(0, 2, 1, 3)
        .reshape(height // cell, width // cell, cell * cell)
    )


def to_uint8(array: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    """与 make_multispectral_uint8 一致：所有通道共用一组百分位缩放。"""
    low, high = np.percentile(array, [low_pct, high_pct])
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)
    stretched = np.clip((array - low) / (high - low), 0.0, 1.0)
    return np.rint(stretched * 255.0).astype(np.uint8)


def build_fusion(raw: np.ndarray, order: str, low_pct: float, high_pct: float) -> np.ndarray:
    height, width = raw.shape
    target_h, target_w = height // 2, width // 2

    pan_full = phase_corrected_pan(raw)
    # INTER_AREA 做区域平均下采样，保留真实细节且抗混叠
    pan = cv2.resize(pan_full, (target_w, target_h), interpolation=cv2.INTER_AREA)
    pan_u8 = to_uint8(pan, low_pct, high_pct)

    cube = unpack_cube(raw)[:, :, list(HSI16_BAND_ORDER)].astype(np.float32)
    cube_up = cv2.resize(cube, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    if cube_up.ndim == 2:
        cube_up = cube_up[:, :, None]
    bands_u8 = to_uint8(cube_up, low_pct, high_pct)

    pan_u8 = pan_u8[:, :, None]
    if order == "pan_first":
        return np.concatenate([pan_u8, pan_u8, pan_u8, bands_u8], axis=2)
    if order == "bands_first":
        return np.concatenate([bands_u8, pan_u8], axis=2)
    raise ValueError(f"未知的 order: {order}")


def _process(task: tuple[str, str, str, float, float]) -> bool:
    source, destination, order, low_pct, high_pct = task
    destination_npy = Path(destination)
    destination_png = destination_npy.with_suffix(".png")
    if destination_npy.exists() and destination_png.exists():
        return True
    with Image.open(source) as image:
        raw = np.asarray(image).astype(np.float32)
    fused = build_fusion(raw, order, low_pct, high_pct)
    destination_npy.parent.mkdir(parents=True, exist_ok=True)
    temporary_npy = destination_npy.with_suffix(".npy.tmp")
    with temporary_npy.open("wb") as handle:
        np.save(handle, fused, allow_pickle=False)
    temporary_png = destination_png.with_suffix(".png.tmp")
    Image.fromarray(fused[:, :, :3], mode="RGB").save(temporary_png, format="PNG", compress_level=3)
    temporary_npy.replace(destination_npy)
    temporary_png.replace(destination_png)
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("pan_first", "bands_first"), required=True)
    parser.add_argument("--reference", type=Path, default=Path("data/processed/hsi16_shared_p005_995"))
    parser.add_argument("--raw-train", type=Path, default=Path("data/raw/extracted/data_train"))
    parser.add_argument("--raw-test", type=Path, default=Path("data/raw/extracted/data_test"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--lower-percentile", type=float, default=0.5)
    parser.add_argument("--upper-percentile", type=float, default=99.5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 张，用于冒烟测试")
    args = parser.parse_args()
    output = args.output or Path(f"data/processed/fusion_{args.order}")
    channels = 19 if args.order == "pan_first" else 17

    raw_lookup: dict[str, Path] = {}
    for root in (args.raw_train, args.raw_test):
        for path in root.rglob("*.png"):
            raw_lookup[path.stem] = path

    tasks: list[tuple[str, str, str, float, float]] = []
    for split in ("train", "val", "test"):
        reference_images = args.reference / "images" / split
        if not reference_images.is_dir():
            continue
        stems = sorted({p.stem for p in reference_images.glob("*.npy")})
        if args.limit:
            stems = stems[: args.limit]
        missing = [s for s in stems if s not in raw_lookup]
        if missing:
            raise KeyError(f"{split} 缺少原图: {missing[:5]}")
        for stem in stems:
            tasks.append((str(raw_lookup[stem]), str(output / "images" / split / f"{stem}.npy"),
                          args.order, args.lower_percentile, args.upper_percentile))
        source_labels = args.reference / "labels" / split
        if source_labels.is_dir():
            destination_labels = output / "labels" / split
            destination_labels.mkdir(parents=True, exist_ok=True)
            for stem in stems:
                label = source_labels / f"{stem}.txt"
                if label.exists():
                    shutil.copy2(label, destination_labels / label.name)
        print(f"{split}: {len(stems)} 张")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        skipped = sum(tqdm(pool.map(_process, tasks, chunksize=4), total=len(tasks),
                           desc=f"融合 {args.order}", mininterval=30))
    print(f"完成 {len(tasks)} 张，其中已存在跳过 {skipped} 张")

    reference_yaml = yaml.safe_load((args.reference / "dataset.yaml").read_text(encoding="utf-8"))
    base = {
        "path": str(output.resolve()).replace("\\", "/"),
        "val": "images/val",
        "test": "images/test",
        "channels": channels,
        "names": reference_yaml["names"],
        "fusion_order": args.order,
        "encoding": (f"pan(phase-corrected, area-downsampled) + 16 HSI bands(bilinear-upsampled) "
                     f"at half mosaic resolution; shared P{args.lower_percentile}-P{args.upper_percentile} uint8"),
        "split_seed": reference_yaml.get("split_seed"),
    }
    heldout = dict(base, train="images/train", training_scope="fixed_2400_train_600_val")
    full = dict(base, train=["images/train", "images/val"], training_scope="all_3000_labeled")
    (output / "dataset.yaml").write_text(yaml.safe_dump(heldout, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (output / "dataset_all.yaml").write_text(yaml.safe_dump(full, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"已写入 {output / 'dataset.yaml'}（留出）和 dataset_all.yaml（全量），channels={channels}")


if __name__ == "__main__":
    main()
