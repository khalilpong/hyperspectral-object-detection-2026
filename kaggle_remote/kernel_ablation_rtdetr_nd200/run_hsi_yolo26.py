"""远程训练流水线：单模型 YOLO / RT-DETR + HSI16。

同一个脚本既能在 Kaggle Notebook 上跑，也能在任意 Linux GPU 云服务器上跑。

为什么需要远程训练：本机 RTX 4060 只有 8GB 显存，yolo26m 被迫 batch=2，
全量成绩 0.60553 反而输给 s 模型的 0.61766。需要 16GB 以上显存把 batch 提到 4~8。

────────────────────────────────────────────────────────────────────────────
配置方式（优先级从高到低）
────────────────────────────────────────────────────────────────────────────
1. 环境变量 HSI_*（云服务器上用这个）
2. 下方 CONFIG 字典的默认值（Kaggle 上无法设环境变量，由 make_kernel.py 写入）

  HSI_MODE         smoke | ablation | full
  HSI_ARCHITECTURE yolo | rtdetr
  HSI_MODEL        预训练权重文件名，如 yolo26m.pt（需在代码目录里）
  HSI_EPOCHS       训练轮数
  HSI_RUN_NAME     本次运行名（决定输出子目录和提交文件名）
  HSI_ATTEMPTS     batch/workers 降级序列，如 "8:2,6:2,4:2,4:0"
  HSI_MULTISCALE   1 = 7 尺度融合推理（出正式提交用），0 = 单尺度（快速验证用）
  HSI_SEED         训练随机种子
  HSI_EXTRA_CHANNEL_INIT  random | zero；16 通道输入首层新增 13 通道的初始化方式
  HSI_SPECTRAL_STEM  1 = 可学习的 16→3 光谱投影后接完整预训练 YOLO，0 = 原始首层扩展
  HSI_LOWER_PERCENTILE  HSI16 共享缩放下百分位，默认 0.5
  HSI_UPPER_PERCENTILE  HSI16 共享缩放上百分位，默认 99.5
  HSI_CLS_PW        分类 BCE 的类别频次权重幂，0.0 关闭，范围 0.0~1.0
  HSI_DFL           Distribution Focal Loss gain，默认 1.5
  HSI_DEGREES       YOLO random-affine 最大旋转角，默认 0.0
  HSI_RTDETR_NUM_DENOISING  RT-DETR 训练 denoising query 数，默认 100
  HSI_FUSION_IOU    同一 checkpoint 多尺度 box voting IoU，默认 0.70
  HSI_SUPPORT_GAIN  同一 checkpoint 多尺度支持票加成，默认 0.0
  HSI_PHASE_TARGET_LONG_EDGE  相位感知 HSI16 重建长边，默认 1024
  HSI_OBJECT_CROPS  1 = 训练集增加一份 128x256 对象感知 crop，0 = 不增加
  HSI_TILE_INFERENCE  1 = 同一 checkpoint 的全图多尺度 + 切片推理，0 = 仅原推理
  HSI_CODE_ROOT    代码目录（含 scripts.*.py 平铺文件），不设则在输入目录里自动查找
  HSI_VARIANT_CODE_ROOT  crop/tile 增量代码目录；不设则按 prepare_object_crops 文件自动查找
  HSI_COMP_ROOT    比赛原始数据目录（含 class.txt），不设则自动查找
  HSI_INPUT_ROOT   自动查找的起点，默认 /kaggle/input
  HSI_WORK_ROOT    临时工作目录（放 ~9GB 的 16 波段数据），默认取 /kaggle/temp 与 /tmp 中空间大的
  HSI_OUT_DIR      产物目录，默认 /kaggle/working；非 Kaggle 环境默认 ./hsi_outputs/<RUN_NAME>

────────────────────────────────────────────────────────────────────────────
无人值守的自动降级
────────────────────────────────────────────────────────────────────────────
  CUDA 显存不足          -> 按 ATTEMPTS 依次降低 batch
  DataLoader 共享内存报错 -> 退回 workers=0
  其他错误               -> 不重试，直接失败，日志尾部写入 status.json

────────────────────────────────────────────────────────────────────────────
产物
────────────────────────────────────────────────────────────────────────────
  status.json           每一步的耗时与结果、完整配置、失败原因（**出问题先看它**）
  resource_log.csv      每 30 秒一次的显存 / GPU 利用率 / 内存
  *.log                 各步骤完整日志
  <RUN_NAME>/           results.csv、args.yaml、last.pt / best.pt（smoke 不存权重）
  submission_<RUN_NAME>.csv  已通过校验的提交文件
"""

import csv
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

# ============================== 默认配置 ==============================
# 由 make_kernel.py 生成 Kaggle Notebook 时替换这一块；云服务器上用环境变量覆盖。
CONFIG = {
    "MODE": "ablation",
    "ARCHITECTURE": "rtdetr",
    "MODEL": "rtdetr-l.pt",
    "EPOCHS": 30,
    "RUN_NAME": "kaggle_ablation_rtdetr-l_nd200_e30",
    "ATTEMPTS": "2:2,1:2,1:0",
    "MULTISCALE": 0,
    "DATA": "hsi16",
    "SEED": 2026,
    "EXTRA_CHANNEL_INIT": "random",
    "SPECTRAL_STEM": 0,
    "LOWER_PERCENTILE": 0.5,
    "UPPER_PERCENTILE": 99.5,
    "CLS_PW": 0.0,
    "SCALE": 0.5,
    "DEGREES": 0.0,
    "DFL": 1.5,
    "RTDETR_NUM_DENOISING": 200,
    "FUSION_IOU": 0.7,
    "SUPPORT_GAIN": 0.0,
    "PHASE_TARGET_LONG_EDGE": 1024,
    "OBJECT_CROPS": 0,
    "TILE_INFERENCE": 0,
}
# =====================================================================

IMGSZ = 1024
ULTRALYTICS_VERSION = "8.4.147"
MULTISCALE_SIZES = ["832", "896", "960", "1024", "1088", "1152", "1216"]


def _cfg(key: str):
    value = os.environ.get(f"HSI_{key}", CONFIG[key])
    if key in (
        "EPOCHS",
        "MULTISCALE",
        "SEED",
        "SPECTRAL_STEM",
        "PHASE_TARGET_LONG_EDGE",
        "OBJECT_CROPS",
        "TILE_INFERENCE",
        "RTDETR_NUM_DENOISING",
    ):
        return int(value)
    if key in (
        "LOWER_PERCENTILE",
        "UPPER_PERCENTILE",
        "CLS_PW",
        "SCALE",
        "DEGREES",
        "DFL",
        "FUSION_IOU",
        "SUPPORT_GAIN",
    ):
        return float(value)
    return value


MODE = _cfg("MODE")
ARCHITECTURE = _cfg("ARCHITECTURE")
MODEL = _cfg("MODEL")
EPOCHS = _cfg("EPOCHS")
RUN_NAME = _cfg("RUN_NAME")
SEED = _cfg("SEED")
EXTRA_CHANNEL_INIT = _cfg("EXTRA_CHANNEL_INIT")
SPECTRAL_STEM = bool(_cfg("SPECTRAL_STEM"))
LOWER_PERCENTILE = _cfg("LOWER_PERCENTILE")
UPPER_PERCENTILE = _cfg("UPPER_PERCENTILE")
CLS_PW = _cfg("CLS_PW")
SCALE = _cfg("SCALE")
DEGREES = _cfg("DEGREES")
DFL = _cfg("DFL")
RTDETR_NUM_DENOISING = _cfg("RTDETR_NUM_DENOISING")
FUSION_IOU = _cfg("FUSION_IOU")
SUPPORT_GAIN = _cfg("SUPPORT_GAIN")
PHASE_TARGET_LONG_EDGE = _cfg("PHASE_TARGET_LONG_EDGE")
OBJECT_CROPS = bool(_cfg("OBJECT_CROPS"))
TILE_INFERENCE = bool(_cfg("TILE_INFERENCE"))
if ARCHITECTURE not in ("yolo", "rtdetr"):
    raise SystemExit(
        "HSI_ARCHITECTURE 必须是 yolo 或 rtdetr，"
        f"收到 {ARCHITECTURE!r}"
    )
if not 0 <= LOWER_PERCENTILE < UPPER_PERCENTILE <= 100:
    raise SystemExit(
        "HSI 百分位必须满足 0 <= LOWER_PERCENTILE < UPPER_PERCENTILE <= 100，"
        f"收到 {LOWER_PERCENTILE:g}/{UPPER_PERCENTILE:g}"
    )
if EXTRA_CHANNEL_INIT not in ("random", "zero"):
    raise SystemExit(
        "HSI_EXTRA_CHANNEL_INIT 必须是 random 或 zero，"
        f"收到 {EXTRA_CHANNEL_INIT!r}"
    )
if not 0.0 <= CLS_PW <= 1.0:
    raise SystemExit(f"HSI_CLS_PW 必须满足 0.0 <= value <= 1.0，收到 {CLS_PW:g}")
if not 0.0 <= SCALE <= 1.0:
    raise SystemExit(f"HSI_SCALE 必须满足 0.0 <= value <= 1.0，收到 {SCALE:g}")
if not math.isfinite(DEGREES) or not 0.0 <= DEGREES <= 180.0:
    raise SystemExit(f"HSI_DEGREES 必须满足 0.0 <= value <= 180.0，收到 {DEGREES:g}")
if not math.isfinite(DFL) or DFL < 0.0:
    raise SystemExit(f"HSI_DFL 必须是有限非负数，收到 {DFL:g}")
if RTDETR_NUM_DENOISING <= 0:
    raise SystemExit(
        "HSI_RTDETR_NUM_DENOISING 必须是正整数，"
        f"收到 {RTDETR_NUM_DENOISING}"
    )
if not math.isfinite(FUSION_IOU) or not 0.0 < FUSION_IOU <= 1.0:
    raise SystemExit(f"HSI_FUSION_IOU 必须是 (0, 1] 内的有限数，收到 {FUSION_IOU:g}")
if not math.isfinite(SUPPORT_GAIN) or SUPPORT_GAIN < 0.0:
    raise SystemExit(f"HSI_SUPPORT_GAIN 必须是有限非负数，收到 {SUPPORT_GAIN:g}")
if SPECTRAL_STEM and EXTRA_CHANNEL_INIT != "random":
    raise SystemExit(
        "HSI_SPECTRAL_STEM 与 HSI_EXTRA_CHANNEL_INIT=zero 互斥；"
        "SpectralStem 自己负责 16→3 identity 初始化"
    )
if PHASE_TARGET_LONG_EDGE <= 0:
    raise SystemExit(
        "HSI_PHASE_TARGET_LONG_EDGE 必须是正整数，"
        f"收到 {PHASE_TARGET_LONG_EDGE}"
    )


def _percentile_tag(value: float) -> str:
    scaled = round(value * 10)
    if abs(value * 10 - scaled) < 1e-9:
        return f"{scaled:03d}"
    return f"{value:g}".replace(".", "p")


def _parse_attempts(text: str):
    """"batch:workers[:devices]"，devices 用 + 连接，例如 "8:2:0+1" = 两张卡 DDP，总 batch 8（每卡 4）。"""
    attempts = []
    for item in text.split(","):
        parts = item.split(":")
        attempts.append((int(parts[0]), int(parts[1]), parts[2].replace("+", ",") if len(parts) > 2 else "0"))
    return attempts


ATTEMPTS = _parse_attempts(_cfg("ATTEMPTS"))
MULTISCALE = bool(_cfg("MULTISCALE"))
if not MULTISCALE and (FUSION_IOU != 0.70 or SUPPORT_GAIN != 0.0):
    raise SystemExit("非默认 HSI_FUSION_IOU/HSI_SUPPORT_GAIN 要求 HSI_MULTISCALE=1")
DATA = _cfg("DATA")   # hsi16 / hsi16_phase = 16 波段 NPY；pseudo_rgb[:bands] = 三通道 PNG
BANDS = ["5", "8", "13"]
if DATA.startswith("pseudo_rgb"):
    if ":" in DATA:
        BANDS = DATA.split(":", 1)[1].split(",")
        if len(BANDS) != 3:
            raise SystemExit(f"伪RGB 需要正好 3 个波段，收到 {BANDS}")
    DATA = "pseudo_rgb"
elif DATA not in ("hsi16", "hsi16_phase"):
    raise SystemExit(
        f"未知 DATA={DATA}，只支持 hsi16 / hsi16_phase / pseudo_rgb[:波段,波段,波段]"
    )
if DATA not in ("hsi16", "hsi16_phase") and EXTRA_CHANNEL_INIT != "random":
    raise SystemExit("--extra-channel-init zero 只适用于 16 通道 HSI 输入")
if DATA not in ("hsi16", "hsi16_phase") and SPECTRAL_STEM:
    raise SystemExit("SpectralStem 目前只支持 16 通道 HSI 输入")
if DATA != "hsi16" and (OBJECT_CROPS or TILE_INFERENCE):
    raise SystemExit("对象 crop 与切片推理目前只支持 16 通道 HSI 输入")
if DATA == "hsi16_phase" and (
    EXTRA_CHANNEL_INIT != "random" or SPECTRAL_STEM or OBJECT_CROPS or TILE_INFERENCE
):
    raise SystemExit(
        "hsi16_phase 是只改变输入表示的单变量实验，必须保持 random 首层初始化，"
        "并关闭 SpectralStem、object crops 与 tile inference"
    )
if ARCHITECTURE == "rtdetr":
    if DATA != "hsi16":
        raise SystemExit("RT-DETR 远程路径目前只支持 HSI_DATA=hsi16")
    if SPECTRAL_STEM:
        raise SystemExit("RT-DETR 不支持 YOLO 专属的 HSI_SPECTRAL_STEM")
    if CLS_PW != 0.0:
        raise SystemExit("RT-DETR 不支持 YOLO 专属的 HSI_CLS_PW")
    if SCALE != 0.5:
        raise SystemExit("RT-DETR 远程路径尚未暴露 HSI_SCALE")
    if DEGREES != 0.0:
        raise SystemExit("RT-DETR 远程路径不使用 YOLO HSI_DEGREES")
    if DFL != 1.5:
        raise SystemExit("RT-DETR 不使用 YOLO DFL gain；HSI_DFL 必须保持 1.5")
    if OBJECT_CROPS or TILE_INFERENCE:
        raise SystemExit("RT-DETR smoke/fixed 路径暂不支持 object crops 或 tile inference")
elif RTDETR_NUM_DENOISING != 100:
    raise SystemExit("YOLO 路径不使用 HSI_RTDETR_NUM_DENOISING；必须保持默认 100")
IMAGE_EXT = "npy" if DATA in ("hsi16", "hsi16_phase") else "png"
ON_KAGGLE = Path("/kaggle/working").exists()

OUT = Path(os.environ.get("HSI_OUT_DIR") or ("/kaggle/working" if ON_KAGGLE else f"hsi_outputs/{RUN_NAME}")).resolve()
INPUT_ROOT = Path(os.environ.get("HSI_INPUT_ROOT", "/kaggle/input"))

STATUS = {
    "config": {"MODE": MODE, "ARCHITECTURE": ARCHITECTURE,
               "MODEL": MODEL, "EPOCHS": EPOCHS, "RUN_NAME": RUN_NAME,
               "ATTEMPTS": ATTEMPTS, "MULTISCALE": MULTISCALE, "DATA": DATA, "BANDS": BANDS,
               "EXTRA_CHANNEL_INIT": EXTRA_CHANNEL_INIT,
               "SPECTRAL_STEM": SPECTRAL_STEM,
               "LOWER_PERCENTILE": LOWER_PERCENTILE, "UPPER_PERCENTILE": UPPER_PERCENTILE,
               "CLS_PW": CLS_PW,
               "SCALE": SCALE,
               "DEGREES": DEGREES,
               "DFL": DFL,
               "RTDETR_NUM_DENOISING": RTDETR_NUM_DENOISING,
               "FUSION_IOU": FUSION_IOU, "SUPPORT_GAIN": SUPPORT_GAIN,
               "PHASE_TARGET_LONG_EDGE": PHASE_TARGET_LONG_EDGE,
               "OBJECT_CROPS": OBJECT_CROPS, "TILE_INFERENCE": TILE_INFERENCE,
               "IMGSZ": IMGSZ, "SEED": SEED},
    "host": {"on_kaggle": ON_KAGGLE, "node": platform.node(), "python": sys.version.split()[0]},
    "steps": {},
    "started": time.time(),
    "started_iso": time.strftime("%Y-%m-%d %H:%M:%S %z"),
}


def save_status() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "status.json").write_text(json.dumps(STATUS, indent=2, ensure_ascii=False), encoding="utf-8")


def step(name: str, **info) -> None:
    STATUS["steps"][name] = {"t": round(time.time() - STATUS["started"], 1), **info}
    save_status()
    print(f"\n===== [{STATUS['steps'][name]['t']:>7.1f}s] {name} {info}", flush=True)


def list_tree(root: Path, depth: int = 3, limit: int = 60) -> list[str]:
    """列出目录树（跟随符号链接），用于找不到文件时自我诊断。"""
    lines: list[str] = []
    base = len(root.parts)
    for current, dirs, files in os.walk(root, followlinks=True):
        level = len(Path(current).parts) - base
        if level >= depth:
            dirs[:] = []
        marker = " -> " + os.path.realpath(current) if os.path.islink(current) else ""
        lines.append(f"{'  ' * level}{Path(current).name}/  ({len(files)} 文件){marker}")
        if len(lines) >= limit:
            lines.append("  ...")
            break
    return lines


def find_one(root: Path, filename: str) -> Path:
    # 用 os.walk(followlinks=True) 而不是 Path.rglob：Kaggle 的比赛数据以符号链接挂载，
    # Python 3.12 的 rglob 不会进入符号链接指向的目录，第一次冒烟测试就因此找不到 class.txt
    for current, _dirs, files in os.walk(root, followlinks=True):
        if filename in files:
            return Path(current) / filename
    tree = "\n".join(list_tree(root))
    raise FileNotFoundError(f"在 {root} 下找不到 {filename}。当前目录树：\n{tree}")


def locate_competition_root(input_root: Path, work_root: Path) -> Path:
    """找到比赛原始数据目录（含 class.txt）。

    Kaggle 不会把本比赛的数据挂载进 Notebook（metadata 里的 competition_sources
    会被静默丢弃），所以原始数据以私有数据集 hsi-competition-raw 的形式上传了一个 zip。
    Kaggle 可能自动解压它，也可能原样保留，两种情况都要处理。
    """
    for current, _dirs, files in os.walk(input_root, followlinks=True):
        if "class.txt" in files:
            return Path(current)
    import zipfile
    extracted = work_root / "competition_raw"
    if (extracted / "class.txt").exists():
        return extracted
    for current, _dirs, files in os.walk(input_root, followlinks=True):
        for name in files:
            if not name.endswith(".zip"):
                continue
            path = Path(current) / name
            with zipfile.ZipFile(path) as archive:
                if "class.txt" not in archive.namelist():
                    continue
                print(f"解压比赛数据 {path} -> {extracted}", flush=True)
                archive.extractall(extracted)
            return extracted
    tree = "\n".join(list_tree(input_root))
    raise FileNotFoundError(f"在 {input_root} 下既没有 class.txt，也没有包含它的 zip。当前目录树：\n{tree}")


def run(cmd: list[str], log_path: Path, cwd: Path, env: dict) -> tuple[int, str]:
    """执行命令，完整输出写入日志文件，返回 (返回码, 日志尾部)。"""
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
    return proc.returncode, tail


def build_training_command(
    weights: Path,
    data_yaml: Path,
    *,
    batch: int,
    device: str,
    workers: int,
) -> list[str]:
    """Build one architecture-specific, single-checkpoint training command."""
    common = [
        "--model",
        str(weights),
        "--data",
        str(data_yaml),
        "--epochs",
        str(EPOCHS),
        "--imgsz",
        str(IMGSZ),
        "--batch",
        str(batch),
        "--device",
        device,
        "--workers",
        str(workers),
        "--seed",
        str(SEED),
        "--name",
        RUN_NAME,
    ]
    if ARCHITECTURE == "rtdetr":
        command = [
            sys.executable,
            "scripts/train_rtdetr_hsi.py",
            *common,
            "--extra-channel-init",
            EXTRA_CHANNEL_INIT,
            "--num-denoising",
            str(RTDETR_NUM_DENOISING),
        ]
    else:
        command = [
            sys.executable,
            "scripts/train_baseline.py",
            *common,
            "--cls-pw",
            f"{CLS_PW:g}",
            "--scale",
            f"{SCALE:g}",
            "--degrees",
            f"{DEGREES:g}",
            "--dfl",
            f"{DFL:g}",
        ]
        if SPECTRAL_STEM:
            command.append("--spectral-stem")
        else:
            command += ["--extra-channel-init", EXTRA_CHANNEL_INIT]
    if MODE == "full":
        command.append("--no-val")
    return command


def monitor(stop: threading.Event) -> None:
    """每 30 秒记录一次 GPU 显存 / 利用率与系统内存。监控失败不影响训练。"""
    with (OUT / "resource_log.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t_sec", "gpu_mem_mib", "gpu_util_pct", "ram_used_gb", "ram_avail_gb"])
        while not stop.is_set():
            try:
                q = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=20,
                ).stdout.strip().splitlines()[0].split(",")
                import psutil
                vm = psutil.virtual_memory()
                writer.writerow([round(time.time() - STATUS["started"]), q[0].strip(), q[1].strip(),
                                 round(vm.used / 2**30, 1), round(vm.available / 2**30, 1)])
            except Exception as error:
                writer.writerow([round(time.time() - STATUS["started"]), "err", str(error)[:60], "", ""])
            handle.flush()
            stop.wait(30)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    threading.Thread(target=monitor, args=(stop,), daemon=True).start()
    try:
        _main()
        STATUS["result"] = "success"
    except Exception as error:
        STATUS["result"] = "failed"
        STATUS["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        STATUS["total_sec"] = round(time.time() - STATUS["started"], 1)
        save_status()
        stop.set()


def _main() -> None:
    # ---------- 1. 环境 ----------
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    if os.environ.get("HSI_WORK_ROOT"):
        work_root = Path(os.environ["HSI_WORK_ROOT"]).resolve()
        work_root.mkdir(parents=True, exist_ok=True)
    else:
        candidates = [p for p in (Path("/kaggle/temp"), Path("/tmp")) if p.exists()]
        work_root = max(candidates, key=lambda p: shutil.disk_usage(p).free)
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        ram_gb = None
    step("environment", gpu=gpu, cpus=os.cpu_count(), ram_gb=ram_gb, out_dir=str(OUT),
         work_root=str(work_root), work_free_gb=round(shutil.disk_usage(work_root).free / 2**30, 1),
         input_tree=list_tree(INPUT_ROOT, depth=3) if INPUT_ROOT.exists() else "（不存在）")
    if not gpu:
        raise RuntimeError("没有检测到 NVIDIA GPU（nvidia-smi 无输出）")

    # ---------- 2. 依赖 ----------
    pip = subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"ultralytics=={ULTRALYTICS_VERSION}"],
                         capture_output=True, text=True)
    if pip.returncode != 0:
        raise RuntimeError("pip 安装 ultralytics 失败（Kaggle 上需开启 Internet）\n" + pip.stderr[-2000:])
    import torch
    import ultralytics
    step("dependencies", ultralytics=ultralytics.__version__, torch=torch.__version__,
         cuda_available=torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch 看不到 CUDA。云服务器上请先按文档安装带 CUDA 的 torch，再运行本脚本")

    # ---------- 3. 还原代码目录结构（代码包是平铺上传的） ----------
    code_root = (Path(os.environ["HSI_CODE_ROOT"]) if os.environ.get("HSI_CODE_ROOT")
                 else find_one(INPUT_ROOT, "scripts.train_baseline.py").parent).resolve()
    comp_root = (Path(os.environ["HSI_COMP_ROOT"]) if os.environ.get("HSI_COMP_ROOT")
                 else locate_competition_root(INPUT_ROOT, work_root)).resolve()
    project = work_root / "project"
    (project / "src" / "hsi_detection").mkdir(parents=True, exist_ok=True)
    (project / "scripts").mkdir(parents=True, exist_ok=True)
    code_roots = [code_root]
    variant_code_root = None
    if OBJECT_CROPS or TILE_INFERENCE:
        variant_code_root = (
            Path(os.environ["HSI_VARIANT_CODE_ROOT"])
            if os.environ.get("HSI_VARIANT_CODE_ROOT")
            else find_one(INPUT_ROOT, "scripts.prepare_object_crops.py").parent
        ).resolve()
        code_roots.append(variant_code_root)
    for source_root in code_roots:
        for f in source_root.glob("hsi_detection.*.py"):
            shutil.copy2(f, project / "src" / "hsi_detection" / f.name.removeprefix("hsi_detection."))
        for f in source_root.glob("scripts.*.py"):
            shutil.copy2(f, project / "scripts" / f.name.removeprefix("scripts."))
    manifest_dir = project / "data" / "processed" / "pseudo_rgb"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(code_root / "split_manifest.csv", manifest_dir / "split_manifest.csv")
    env = dict(os.environ, PYTHONPATH=str(project / "src"))
    step(
        "code_restored",
        code_root=str(code_root),
        variant_code_root=str(variant_code_root) if variant_code_root else None,
        comp_root=str(comp_root),
    )

    # ---------- 4. 生成训练数据（16 波段 NPY 或 伪RGB PNG） ----------
    if DATA in ("hsi16", "hsi16_phase"):
        percentile_dir = (
            f"hsi16_shared_p{_percentile_tag(LOWER_PERCENTILE)}_"
            f"{_percentile_tag(UPPER_PERCENTILE)}"
        )
        if DATA == "hsi16_phase":
            percentile_dir = (
                f"hsi16_phase_p{_percentile_tag(LOWER_PERCENTILE)}_"
                f"{_percentile_tag(UPPER_PERCENTILE)}_l{PHASE_TARGET_LONG_EDGE}"
            )
            free_gib = shutil.disk_usage(work_root).free / 2**30
            if free_gib < 45:
                raise RuntimeError(
                    "相位感知 HSI16 预计需要约 32 GiB 数据缓存；"
                    f"工作盘只剩 {free_gib:.1f} GiB，低于 45 GiB 安全门槛"
                )
        data_dir = project / "data" / "processed" / percentile_dir
        prepare_script = (
            "scripts/prepare_phase_aware_multispectral.py"
            if DATA == "hsi16_phase"
            else "scripts/prepare_multispectral.py"
        )
        prepare_cmd = [sys.executable, prepare_script, "--raw-root", str(comp_root),
                       "--output", str(data_dir), "--workers", str(os.cpu_count() or 4),
                       "--lower-percentile", f"{LOWER_PERCENTILE:g}",
                       "--upper-percentile", f"{UPPER_PERCENTILE:g}"]
        if DATA == "hsi16_phase":
            prepare_cmd += ["--target-long-edge", str(PHASE_TARGET_LONG_EDGE)]
    else:
        # 输出目录不能和上面放 split_manifest.csv 的 pseudo_rgb 目录重名
        data_dir = project / "data" / "processed" / ("pseudo_rgb_b" + "-".join(BANDS))
        prepare_cmd = [sys.executable, "scripts/prepare_pseudo_rgb.py", "--raw-root", str(comp_root),
                       "--output", str(data_dir), "--workers", str(os.cpu_count() or 4), "--bands", *BANDS]
    code, tail = run(prepare_cmd, OUT / "prepare.log", project, env)
    if code != 0:
        raise RuntimeError("数据准备失败\n" + tail)
    counts = {s: len(list((data_dir / "images" / s).glob(f"*.{IMAGE_EXT}"))) for s in ("train", "val", "test")}
    step("data_prepared", counts=counts, work_free_gb=round(shutil.disk_usage(work_root).free / 2**30, 1))
    if counts != {"train": 2400, "val": 600, "test": 1000}:
        raise RuntimeError(f"数据数量不符合预期 2400/600/1000：{counts}")
    if DATA == "pseudo_rgb":
        # 伪RGB 的划分是现场按种子重新生成的，必须和 16 波段用的 split_manifest.csv 完全一致，否则留出集分数没法比
        def _split(path):
            with open(path, newline="", encoding="utf-8") as handle:
                return {row["image_id"]: row["split"] for row in csv.DictReader(handle)}
        if _split(data_dir / "split_manifest.csv") != _split(code_root / "split_manifest.csv"):
            raise RuntimeError("伪RGB 数据的训练/验证划分和 split_manifest.csv 不一致")
        # prepare_pseudo_rgb.py 只写留出用的 dataset.yaml，全量训练的 dataset_all.yaml 在这里补上（train = train + val）
        import yaml
        cfg = yaml.safe_load((data_dir / "dataset.yaml").read_text(encoding="utf-8"))
        cfg["train"] = ["images/train", "images/val"]
        cfg["training_scope"] = "all_3000_labeled_images"
        (data_dir / "dataset_all.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                                                   encoding="utf-8")

    if OBJECT_CROPS:
        crop_splits = ["train", "val"] if MODE == "full" else ["train"]
        crop_cmd = [
            sys.executable,
            "scripts/prepare_object_crops.py",
            "--dataset-root",
            str(data_dir),
            "--splits",
            *crop_splits,
            "--tile-height",
            "128",
            "--tile-width",
            "256",
            "--crops-per-image",
            "1",
            "--jitter-fraction",
            "0.15",
            "--minimum-visible",
            "0.9",
            "--seed",
            str(SEED),
        ]
        code, tail = run(crop_cmd, OUT / "prepare_object_crops.log", project, env)
        if code != 0:
            raise RuntimeError("对象 crop 数据准备失败\n" + tail)
        crop_report = json.loads((data_dir / "object_crop_report.json").read_text(encoding="utf-8"))
        expected_sources = 3000 if MODE == "full" else 2400
        if crop_report["generated_crops"] < int(expected_sources * 0.95):
            raise RuntimeError(
                f"对象 crop 数量过少：{crop_report['generated_crops']} / {expected_sources}"
            )
        step(
            "object_crops_prepared",
            generated=crop_report["generated_crops"],
            skipped=len(crop_report["skipped_images"]),
            splits=crop_splits,
        )

    # ---------- 5. 训练（自动降级） ----------
    if OBJECT_CROPS:
        data_yaml = data_dir / (
            "dataset_all_object_crops.yaml" if MODE == "full" else "dataset_object_crops.yaml"
        )
    else:
        data_yaml = data_dir / ("dataset_all.yaml" if MODE == "full" else "dataset.yaml")
    weights = code_root / MODEL
    if not weights.exists():
        weights = Path(MODEL)   # 代码包里没有的官方权重（如 yolo26l.pt）交给 Ultralytics 联网自动下载
    run_dir = project / "runs" / RUN_NAME
    used = None
    for batch, workers, device in ATTEMPTS:
        shutil.rmtree(run_dir, ignore_errors=True)
        cmd = build_training_command(
            weights,
            data_yaml,
            batch=batch,
            device=device,
            workers=workers,
        )
        t0 = time.time()
        code, tail = run(cmd, OUT / f"train_b{batch}_w{workers}_d{device.replace(',', '+')}.log", project, env)
        low = tail.lower()
        oom = "out of memory" in low
        shm = "shared memory" in low or "bus error" in low or "dataloader worker" in low
        step(f"train_attempt_b{batch}_w{workers}_d{device.replace(',', '+')}", returncode=code, sec=round(time.time() - t0),
             cuda_oom=oom, shm_error=shm)
        if code == 0:
            used = (batch, workers, device)
            break
        if not (oom or shm or "," in device):   # 多卡 DDP 出任何错都降级到单卡再试
            raise RuntimeError(f"训练失败（非显存/共享内存问题，不再降级）\n{tail}")
    if used is None:
        raise RuntimeError("所有 batch/workers 组合都失败")

    keep = OUT / RUN_NAME
    keep.mkdir(parents=True, exist_ok=True)
    for name in ("results.csv", "args.yaml"):
        if (run_dir / name).exists():
            shutil.copy2(run_dir / name, keep / name)
    last = run_dir / "weights" / "last.pt"
    if MODE != "smoke":
        shutil.copy2(last, keep / "last.pt")
        if (run_dir / "weights" / "best.pt").exists():
            shutil.copy2(run_dir / "weights" / "best.pt", keep / "best.pt")
    step(
        "trained",
        batch=used[0],
        workers=used[1],
        device=used[2],
        architecture=ARCHITECTURE,
        spectral_stem=SPECTRAL_STEM,
        cls_pw=CLS_PW,
        scale=SCALE,
        dfl=DFL,
    )

    # ---------- 6. 推理 + 校验 ----------
    submission = OUT / f"submission_{RUN_NAME}.csv"
    cmd = [sys.executable, "scripts/predict_submission.py", "--weights", str(last),
           "--architecture", ARCHITECTURE,
           "--images", str(data_dir / "images" / "test"), "--output", str(submission),
           "--input-format", IMAGE_EXT, "--batch", "1", "--device", "0", "--half",
           "--conf", "0.0001", "--iou", "0.70", "--max-det", "300", "--imgsz", str(IMGSZ)]
    if MULTISCALE:
        cmd += [
            "--multi-scale", *MULTISCALE_SIZES,
            "--fusion-iou", f"{FUSION_IOU:g}",
            "--support-gain", f"{SUPPORT_GAIN:g}",
        ]
    if TILE_INFERENCE:
        cmd += [
            "--tile-size", "128", "256",
            "--tile-stride", "96", "192",
            "--tile-imgsz", str(IMGSZ),
        ]
    t0 = time.time()
    code, tail = run(cmd, OUT / "predict.log", project, env)
    if code != 0:
        raise RuntimeError("推理失败\n" + tail)
    # 校验脚本的 --images 默认指向本地才有的 pseudo_rgb 目录，必须显式指定，
    # 否则读不到图片尺寸，会把每一行都判为"未知图片 ID"
    code, tail = run([sys.executable, "scripts/check_submission.py", str(submission),
                      "--images", str(data_dir / "images" / "test")],
                     OUT / "check.log", project, env)
    step(
        "predicted",
        sec=round(time.time() - t0),
        multiscale=MULTISCALE,
        fusion_iou=FUSION_IOU if MULTISCALE else None,
        support_gain=SUPPORT_GAIN if MULTISCALE else None,
        tiled=TILE_INFERENCE,
        check=tail.strip()[-200:],
    )
    if code != 0:
        raise RuntimeError("提交文件校验失败\n" + tail)

    if ON_KAGGLE:
        shutil.rmtree(project, ignore_errors=True)   # Kaggle 上 16 波段数据不进输出，避免产物过大


if __name__ == "__main__":
    main()
