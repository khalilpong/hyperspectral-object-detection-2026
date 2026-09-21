"""从主脚本 run_hsi_yolo26.py 生成一个 Kaggle Notebook 目录。

为什么要生成而不是手写：Kaggle 的 script 类型 Notebook 只上传一个代码文件，
没法传命令行参数或环境变量，配置只能写死在文件里。如果为 smoke / ablation / full
各手工维护一份脚本，很容易改了一份忘了另一份。所以只维护一份主脚本，
用本工具把配置写进 CONFIG 块，再生成 kernel-metadata.json。

用法（在 kaggle_remote 目录下）：
    python make_kernel.py --mode smoke    --model yolo26m.pt --epochs 1
    python make_kernel.py --mode smoke    --architecture rtdetr --epochs 1
    python make_kernel.py --mode ablation --model yolo26m.pt --epochs 30 --attempts 8:2,6:2,4:2,4:0
    python make_kernel.py --mode full     --model yolo26m.pt --epochs 30 --multiscale
    python make_kernel.py --mode full     --model yolo26m.pt --epochs 30 --multiscale \
        --lower-percentile 1 --upper-percentile 99
    python make_kernel.py --mode ablation --model yolo26m.pt --epochs 30 \
        --extra-channel-init zero
    python make_kernel.py --mode ablation --model yolo26m.pt --epochs 30 \
        --spectral-stem
    python make_kernel.py --mode ablation --model yolo26m.pt --epochs 30 \
        --data hsi16_phase --phase-target-long-edge 1024
    python make_kernel.py --mode ablation --model yolo26m.pt --epochs 30 \
        --cls-pw 0.25
    python make_kernel.py --mode ablation --model yolo26m.pt --epochs 30 \
        --dfl 2.0
    python make_kernel.py --mode ablation --model yolo26m.pt --epochs 30 \
        --box-iou-loss eiou

生成后推送：
    cd kernel_<mode> && kaggle kernels push -p .
"""

import argparse
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
USERNAME = "zephyrpong"
CODE_DATASET = f"{USERNAME}/hsi-detection-code"
CROP_CODE_DATASET = f"{USERNAME}/hsi-object-crop-code"
# 比赛数据无法通过 competition_sources 挂载（Kaggle 会静默丢弃该字段），
# 所以把原始比赛 zip 上传成了私有数据集，改用 dataset_sources 挂载
RAW_DATASET = f"{USERNAME}/hsi-competition-raw"


def _percentile_tag(value: float) -> str:
    scaled = round(value * 10)
    if abs(value * 10 - scaled) < 1e-9:
        return f"{scaled:03d}"
    return f"{value:g}".replace(".", "p")


def _validate_percentiles(lower: float, upper: float) -> None:
    if not 0 <= lower < upper <= 100:
        raise ValueError(
            "百分位必须满足 0 <= lower < upper <= 100，"
            f"收到 {lower:g}/{upper:g}"
        )


def _fraction_tag(value: float) -> str:
    scaled = round(value * 100)
    if abs(value * 100 - scaled) < 1e-9:
        return f"{scaled:03d}"
    return f"{value:g}".replace(".", "p")


def _validate_cls_pw(value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"cls_pw 必须满足 0.0 <= value <= 1.0，收到 {value:g}")


def _validate_scale(value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"random-affine scale 必须满足 0.0 <= value <= 1.0，收到 {value:g}")


def _validate_degrees(value: float) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 180.0:
        raise ValueError(
            "random-affine degrees 必须满足 0.0 <= value <= 180.0，"
            f"收到 {value:g}"
        )


def _validate_dfl(value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"dfl loss gain 必须是有限非负数，收到 {value:g}")


def _validate_inference_fusion(fusion_iou: float, support_gain: float) -> None:
    if not math.isfinite(fusion_iou) or not 0.0 < fusion_iou <= 1.0:
        raise ValueError(f"fusion IoU 必须是 (0, 1] 内的有限数，收到 {fusion_iou:g}")
    if not math.isfinite(support_gain) or support_gain < 0.0:
        raise ValueError(f"support gain 必须是有限非负数，收到 {support_gain:g}")


def _compact_decimal_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "ablation", "full"), required=True)
    parser.add_argument("--architecture", choices=("yolo", "rtdetr"), default="yolo")
    parser.add_argument(
        "--model",
        help="预训练权重；默认随架构选择 yolo26m.pt 或 rtdetr-l.pt",
    )
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument(
        "--attempts",
        help=(
            "batch:workers[:devices] 降级序列；devices 用 + 连接表示多卡 DDP，如 8:2:0+1。"
            "默认 YOLO=8:2,6:2,4:2,4:0，RT-DETR=2:2,1:2,1:0"
        ),
    )
    parser.add_argument("--multiscale", action="store_true", help="推理用 7 尺度融合（出正式提交时用）")
    parser.add_argument("--data", default="hsi16",
                        help="训练数据：hsi16（普通 16 波段 NPY，默认）、hsi16_phase（相位感知 16 波段 NPY）、"
                             "pseudo_rgb（波段 5/8/13 伪RGB PNG）"
                             "或 pseudo_rgb:3,6,8（自定义三个波段）")
    parser.add_argument("--seed", type=int, default=2026, help="训练随机种子")
    parser.add_argument("--extra-channel-init", choices=("random", "zero"), default="random",
                        help="16 通道输入首层新增通道的初始化；zero 会保留预训练 RGB 初始函数")
    parser.add_argument("--spectral-stem", action="store_true",
                        help="使用 identity 初始化的可学习 16→3 光谱投影，再进入完整预训练 YOLO")
    parser.add_argument("--lower-percentile", type=float, default=0.5,
                        help="HSI16 共享缩放下百分位")
    parser.add_argument("--upper-percentile", type=float, default=99.5,
                        help="HSI16 共享缩放上百分位")
    parser.add_argument("--cls-pw", type=float, default=0.0,
                        help="分类 BCE 的类别频次权重幂；0.0 关闭，1.0 为完整逆频率")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="训练 random-affine scale；0.5 为 Ultralytics 基线，区别于推理 multiscale")
    parser.add_argument("--degrees", type=float, default=0.0,
                        help="YOLO 训练 random-affine 最大绝对旋转角；0.0 为基线")
    parser.add_argument("--dfl", type=float, default=1.5,
                        help="Distribution Focal Loss gain；1.5 为 Ultralytics 基线")
    parser.add_argument("--box-iou-loss", choices=("ciou", "eiou"), default="ciou",
                        help="YOLO 边框重叠损失；ciou 为 Ultralytics 基线")
    parser.add_argument("--rtdetr-num-denoising", type=int, default=100,
                        help="RT-DETR 训练 denoising query 数；100 为 Ultralytics 基线")
    parser.add_argument("--fusion-iou", type=float, default=0.70,
                        help="同一 checkpoint 多尺度 box voting IoU；仅 --multiscale 生效")
    parser.add_argument("--support-gain", type=float, default=0.0,
                        help="同一 checkpoint 多尺度支持票加成；仅 --multiscale 生效")
    parser.add_argument("--phase-target-long-edge", type=int, default=1024,
                        help="hsi16_phase 保持宽高比重建后的目标长边")
    parser.add_argument("--object-crops", action="store_true",
                        help="训练时给每张源图增加一个 128x256 对象感知 crop")
    parser.add_argument("--tile-inference", action="store_true",
                        help="推理时合并同一 checkpoint 的全图与 128x256 tiles")
    parser.add_argument("--run-name", help="默认按 模式_模型_数据_轮数 自动生成")
    parser.add_argument("--machine-shape", default=None,
                        help="GPU 型号；注意 NvidiaL4 对本账号不开放（推送会 400），默认留空用 Kaggle 分配的 GPU")
    args = parser.parse_args()
    if args.model is None:
        args.model = "rtdetr-l.pt" if args.architecture == "rtdetr" else "yolo26m.pt"
    if args.attempts is None:
        args.attempts = (
            "2:2,1:2,1:0"
            if args.architecture == "rtdetr"
            else "8:2,6:2,4:2,4:0"
        )

    try:
        _validate_percentiles(args.lower_percentile, args.upper_percentile)
        _validate_cls_pw(args.cls_pw)
        _validate_scale(args.scale)
        _validate_degrees(args.degrees)
        _validate_dfl(args.dfl)
        _validate_inference_fusion(args.fusion_iou, args.support_gain)
    except ValueError as error:
        parser.error(str(error))
    if (args.object_crops or args.tile_inference) and args.data != "hsi16":
        parser.error("--object-crops/--tile-inference 目前只支持 --data hsi16")
    if args.spectral_stem and args.data != "hsi16":
        parser.error("--spectral-stem 目前只支持 --data hsi16")
    if args.spectral_stem and args.extra_channel_init != "random":
        parser.error("--spectral-stem 与 --extra-channel-init zero 互斥")
    if args.phase_target_long_edge <= 0:
        parser.error("--phase-target-long-edge 必须是正整数")
    if args.rtdetr_num_denoising <= 0:
        parser.error("--rtdetr-num-denoising 必须是正整数")
    if args.data != "hsi16_phase" and args.phase_target_long_edge != 1024:
        parser.error("--phase-target-long-edge 只适用于 --data hsi16_phase")
    if args.data == "hsi16_phase" and args.extra_channel_init != "random":
        parser.error("hsi16_phase 单变量实验必须使用 --extra-channel-init random")
    if not args.multiscale and (args.fusion_iou != 0.70 or args.support_gain != 0.0):
        parser.error("--fusion-iou/--support-gain 的非默认值要求同时启用 --multiscale")
    if args.architecture == "rtdetr":
        if args.data != "hsi16":
            parser.error("RT-DETR 远程路径目前只支持 --data hsi16")
        if args.spectral_stem:
            parser.error("RT-DETR 不支持 YOLO 专属的 --spectral-stem")
        if args.cls_pw != 0.0:
            parser.error("RT-DETR 不支持 YOLO 专属的 --cls-pw")
        if args.scale != 0.5:
            parser.error("RT-DETR 远程路径尚未暴露 random-affine --scale")
        if args.degrees != 0.0:
            parser.error("RT-DETR 远程路径不使用 YOLO --degrees")
        if args.dfl != 1.5:
            parser.error("RT-DETR 不使用 YOLO DFL gain；--dfl 必须保持默认 1.5")
        if args.box_iou_loss != "ciou":
            parser.error("RT-DETR 不使用 YOLO --box-iou-loss；必须保持 ciou")
        if args.object_crops or args.tile_inference:
            parser.error("RT-DETR smoke/fixed 路径暂不支持 object crops 或 tile inference")
    elif args.rtdetr_num_denoising != 100:
        parser.error("YOLO 路径不使用 --rtdetr-num-denoising；必须保持默认 100")

    model_tag = args.model.removesuffix(".pt")
    architecture_tag = "_rtdetr" if args.architecture == "rtdetr" else ""
    if args.data in ("hsi16", "hsi16_phase"):
        percentile_tag = "" if (args.lower_percentile, args.upper_percentile) == (0.5, 99.5) else (
            f"_p{_percentile_tag(args.lower_percentile)}_"
            f"{_percentile_tag(args.upper_percentile)}"
        )
        data_tag = ("_phase" if args.data == "hsi16_phase" else "") + percentile_tag
    elif args.data == "pseudo_rgb":
        if (args.lower_percentile, args.upper_percentile) != (0.5, 99.5):
            parser.error("--lower-percentile/--upper-percentile 只适用于 hsi16")
        data_tag = "_prgb"
    elif args.data.startswith("pseudo_rgb:"):
        if (args.lower_percentile, args.upper_percentile) != (0.5, 99.5):
            parser.error("--lower-percentile/--upper-percentile 只适用于 hsi16")
        data_tag = "_prgb" + args.data.split(":", 1)[1].replace(",", "")
    else:
        raise SystemExit(f"未知 --data {args.data}")
    init_tag = "" if args.extra_channel_init == "random" else f"_xc{args.extra_channel_init}"
    stem_tag = "_stem" if args.spectral_stem else ""
    cls_pw_tag = "" if args.cls_pw == 0.0 else f"_clspw{_fraction_tag(args.cls_pw)}"
    scale_tag = "" if args.scale == 0.5 else f"_scale{_fraction_tag(args.scale)}"
    degrees_tag = "" if args.degrees == 0.0 else f"_deg{_compact_decimal_tag(args.degrees)}"
    dfl_tag = "" if args.dfl == 1.5 else f"_dfl{_fraction_tag(args.dfl)}"
    box_iou_tag = "" if args.box_iou_loss == "ciou" else f"_{args.box_iou_loss}"
    denoising_tag = (
        ""
        if args.rtdetr_num_denoising == 100
        else f"_nd{args.rtdetr_num_denoising}"
    )
    crop_tag = "_crop" if args.object_crops else ""
    tile_tag = "_tile" if args.tile_inference else ""
    variant_tag = (
        data_tag
        + init_tag
        + stem_tag
        + cls_pw_tag
        + scale_tag
        + degrees_tag
        + dfl_tag
        + box_iou_tag
        + denoising_tag
        + crop_tag
        + tile_tag
    )
    inference_tag = ""
    if args.multiscale:
        if args.fusion_iou != 0.70:
            inference_tag += f"_f{_compact_decimal_tag(args.fusion_iou)}"
        if args.support_gain != 0.0:
            inference_tag += f"_sg{_compact_decimal_tag(args.support_gain)}"
    run_name = args.run_name or f"kaggle_{args.mode}_{model_tag}{variant_tag}_e{args.epochs}"
    slug = f"hsi-{model_tag}{variant_tag}{inference_tag}-{args.mode}".replace("_", "-")

    source = (HERE / "run_hsi_yolo26.py").read_text(encoding="utf-8")
    config = (
        "CONFIG = {\n"
        f'    "MODE": "{args.mode}",\n'
        f'    "ARCHITECTURE": "{args.architecture}",\n'
        f'    "MODEL": "{args.model}",\n'
        f'    "EPOCHS": {args.epochs},\n'
        f'    "RUN_NAME": "{run_name}",\n'
        f'    "ATTEMPTS": "{args.attempts}",\n'
        f'    "MULTISCALE": {1 if args.multiscale else 0},\n'
        f'    "DATA": "{args.data}",\n'
        f'    "SEED": {args.seed},\n'
        f'    "EXTRA_CHANNEL_INIT": "{args.extra_channel_init}",\n'
        f'    "SPECTRAL_STEM": {1 if args.spectral_stem else 0},\n'
        f'    "LOWER_PERCENTILE": {args.lower_percentile!r},\n'
        f'    "UPPER_PERCENTILE": {args.upper_percentile!r},\n'
        f'    "CLS_PW": {args.cls_pw!r},\n'
        f'    "SCALE": {args.scale!r},\n'
        f'    "DEGREES": {args.degrees!r},\n'
        f'    "DFL": {args.dfl!r},\n'
        f'    "BOX_IOU_LOSS": "{args.box_iou_loss}",\n'
        f'    "RTDETR_NUM_DENOISING": {args.rtdetr_num_denoising},\n'
        f'    "FUSION_IOU": {args.fusion_iou!r},\n'
        f'    "SUPPORT_GAIN": {args.support_gain!r},\n'
        f'    "PHASE_TARGET_LONG_EDGE": {args.phase_target_long_edge},\n'
        f'    "OBJECT_CROPS": {1 if args.object_crops else 0},\n'
        f'    "TILE_INFERENCE": {1 if args.tile_inference else 0},\n'
        "}"
    )
    rendered, count = re.subn(r"CONFIG = \{.*?\n\}", config, source, count=1, flags=re.S)
    if count != 1:
        raise SystemExit("主脚本里没有找到 CONFIG 块，无法生成")

    folder = HERE / f"kernel_{args.mode}{architecture_tag}{variant_tag}{inference_tag}"
    folder.mkdir(exist_ok=True)
    (folder / "run_hsi_yolo26.py").write_text(rendered, encoding="utf-8")
    metadata = {
        "id": f"{USERNAME}/{slug}",
        "title": slug,
        "code_file": "run_hsi_yolo26.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [
            CODE_DATASET,
            RAW_DATASET,
            *([CROP_CODE_DATASET] if args.object_crops or args.tile_inference else []),
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    if args.machine_shape:
        metadata["machine_shape"] = args.machine_shape
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                                                 encoding="utf-8")
    print(f"已生成 {folder}")
    print(f"  Notebook : {metadata['id']}")
    print(f"  运行名   : {run_name}")
    print(f"  配置     : mode={args.mode} architecture={args.architecture} "
          f"model={args.model} epochs={args.epochs} "
          f"attempts={args.attempts} multiscale={args.multiscale} data={args.data} "
          f"seed={args.seed} extra_channel_init={args.extra_channel_init} "
          f"spectral_stem={args.spectral_stem} "
          f"cls_pw={args.cls_pw:g} "
          f"scale={args.scale:g} "
          f"degrees={args.degrees:g} "
          f"dfl={args.dfl:g} "
          f"box_iou_loss={args.box_iou_loss} "
          f"rtdetr_num_denoising={args.rtdetr_num_denoising} "
          f"fusion_iou={args.fusion_iou:g} support_gain={args.support_gain:g} "
          f"percentiles={args.lower_percentile:g}/{args.upper_percentile:g} "
          f"phase_target_long_edge={args.phase_target_long_edge} "
          f"object_crops={args.object_crops} tile_inference={args.tile_inference}")
    print(f"推送：cd {folder.name} && kaggle kernels push -p .")


if __name__ == "__main__":
    main()
