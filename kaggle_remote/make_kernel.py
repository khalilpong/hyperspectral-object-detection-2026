"""从主脚本 run_hsi_yolo26.py 生成一个 Kaggle Notebook 目录。

为什么要生成而不是手写：Kaggle 的 script 类型 Notebook 只上传一个代码文件，
没法传命令行参数或环境变量，配置只能写死在文件里。如果为 smoke / ablation / full
各手工维护一份脚本，很容易改了一份忘了另一份。所以只维护一份主脚本，
用本工具把配置写进 CONFIG 块，再生成 kernel-metadata.json。

用法（在 kaggle_remote 目录下）：
    python make_kernel.py --mode smoke    --model yolo26m.pt --epochs 1
    python make_kernel.py --mode ablation --model yolo26m.pt --epochs 30 --attempts 8:2,6:2,4:2,4:0
    python make_kernel.py --mode full     --model yolo26m.pt --epochs 30 --multiscale
    python make_kernel.py --mode full     --model yolo26m.pt --epochs 30 --multiscale \
        --lower-percentile 1 --upper-percentile 99

生成后推送：
    cd kernel_<mode> && kaggle kernels push -p .
"""

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
USERNAME = "zephyrpong"
CODE_DATASET = f"{USERNAME}/hsi-detection-code"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "ablation", "full"), required=True)
    parser.add_argument("--model", default="yolo26m.pt")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--attempts", default="8:2,6:2,4:2,4:0", help="batch:workers[:devices] 降级序列；devices 用 + 连接表示多卡 DDP，如 8:2:0+1")
    parser.add_argument("--multiscale", action="store_true", help="推理用 7 尺度融合（出正式提交时用）")
    parser.add_argument("--data", default="hsi16",
                        help="训练数据：hsi16（16 波段 NPY，默认）、pseudo_rgb（波段 5/8/13 伪RGB PNG）"
                             "或 pseudo_rgb:3,6,8（自定义三个波段）")
    parser.add_argument("--seed", type=int, default=2026, help="训练随机种子")
    parser.add_argument("--lower-percentile", type=float, default=0.5,
                        help="HSI16 共享缩放下百分位")
    parser.add_argument("--upper-percentile", type=float, default=99.5,
                        help="HSI16 共享缩放上百分位")
    parser.add_argument("--run-name", help="默认按 模式_模型_数据_轮数 自动生成")
    parser.add_argument("--machine-shape", default=None,
                        help="GPU 型号；注意 NvidiaL4 对本账号不开放（推送会 400），默认留空用 Kaggle 分配的 GPU")
    args = parser.parse_args()

    try:
        _validate_percentiles(args.lower_percentile, args.upper_percentile)
    except ValueError as error:
        parser.error(str(error))

    model_tag = args.model.removesuffix(".pt")
    if args.data == "hsi16":
        data_tag = "" if (args.lower_percentile, args.upper_percentile) == (0.5, 99.5) else (
            f"_p{_percentile_tag(args.lower_percentile)}_"
            f"{_percentile_tag(args.upper_percentile)}"
        )
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
    run_name = args.run_name or f"kaggle_{args.mode}_{model_tag}{data_tag}_e{args.epochs}"
    slug = f"hsi-{model_tag}{data_tag}-{args.mode}".replace("_", "-")

    source = (HERE / "run_hsi_yolo26.py").read_text(encoding="utf-8")
    config = (
        "CONFIG = {\n"
        f'    "MODE": "{args.mode}",\n'
        f'    "MODEL": "{args.model}",\n'
        f'    "EPOCHS": {args.epochs},\n'
        f'    "RUN_NAME": "{run_name}",\n'
        f'    "ATTEMPTS": "{args.attempts}",\n'
        f'    "MULTISCALE": {1 if args.multiscale else 0},\n'
        f'    "DATA": "{args.data}",\n'
        f'    "SEED": {args.seed},\n'
        f'    "LOWER_PERCENTILE": {args.lower_percentile!r},\n'
        f'    "UPPER_PERCENTILE": {args.upper_percentile!r},\n'
        "}"
    )
    rendered, count = re.subn(r"CONFIG = \{.*?\n\}", config, source, count=1, flags=re.S)
    if count != 1:
        raise SystemExit("主脚本里没有找到 CONFIG 块，无法生成")

    folder = HERE / f"kernel_{args.mode}{data_tag}"
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
        "dataset_sources": [CODE_DATASET, RAW_DATASET],
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
    print(f"  配置     : mode={args.mode} model={args.model} epochs={args.epochs} "
          f"attempts={args.attempts} multiscale={args.multiscale} data={args.data} "
          f"seed={args.seed} percentiles={args.lower_percentile:g}/{args.upper_percentile:g}")
    print(f"推送：cd {folder.name} && kaggle kernels push -p .")


if __name__ == "__main__":
    main()
