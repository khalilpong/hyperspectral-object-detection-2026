"""远程训练流水线：YOLO26 + HSI16（16 波段，共享 P0.5-P99.5 缩放）。

同一个脚本既能在 Kaggle Notebook 上跑，也能在任意 Linux GPU 云服务器上跑。

为什么需要远程训练：本机 RTX 4060 只有 8GB 显存，yolo26m 被迫 batch=2，
全量成绩 0.60553 反而输给 s 模型的 0.61766。需要 16GB 以上显存把 batch 提到 4~8。

────────────────────────────────────────────────────────────────────────────
配置方式（优先级从高到低）
────────────────────────────────────────────────────────────────────────────
1. 环境变量 HSI_*（云服务器上用这个）
2. 下方 CONFIG 字典的默认值（Kaggle 上无法设环境变量，由 make_kernel.py 写入）

  HSI_MODE         smoke | ablation | full
  HSI_MODEL        预训练权重文件名，如 yolo26m.pt（需在代码目录里）
  HSI_EPOCHS       训练轮数
  HSI_RUN_NAME     本次运行名（决定输出子目录和提交文件名）
  HSI_ATTEMPTS     batch/workers 降级序列，如 "8:2,6:2,4:2,4:0"
  HSI_MULTISCALE   1 = 7 尺度融合推理（出正式提交用），0 = 单尺度（快速验证用）
  HSI_CODE_ROOT    代码目录（含 scripts.*.py 平铺文件），不设则在输入目录里自动查找
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
    "MODE": "full",
    "MODEL": "yolo26m.pt",
    "EPOCHS": 30,
    "RUN_NAME": "kaggle_full_yolo26m_prgb368_e30",
    "ATTEMPTS": "8:2,6:2,4:2,4:0",
    "MULTISCALE": 1,
    "DATA": "pseudo_rgb:3,6,8",
}
# =====================================================================

IMGSZ = 1024
SEED = 2026
ULTRALYTICS_VERSION = "8.4.147"
MULTISCALE_SIZES = ["832", "896", "960", "1024", "1088", "1152", "1216"]


def _cfg(key: str):
    value = os.environ.get(f"HSI_{key}", CONFIG[key])
    if key in ("EPOCHS", "MULTISCALE"):
        return int(value)
    return value


MODE = _cfg("MODE")
MODEL = _cfg("MODEL")
EPOCHS = _cfg("EPOCHS")
RUN_NAME = _cfg("RUN_NAME")
def _parse_attempts(text: str):
    """"batch:workers[:devices]"，devices 用 + 连接，例如 "8:2:0+1" = 两张卡 DDP，总 batch 8（每卡 4）。"""
    attempts = []
    for item in text.split(","):
        parts = item.split(":")
        attempts.append((int(parts[0]), int(parts[1]), parts[2].replace("+", ",") if len(parts) > 2 else "0"))
    return attempts


ATTEMPTS = _parse_attempts(_cfg("ATTEMPTS"))
MULTISCALE = bool(_cfg("MULTISCALE"))
DATA = _cfg("DATA")   # "hsi16" = 16 波段 NPY（默认）；"pseudo_rgb" 或 "pseudo_rgb:3,6,8" = 三通道伪RGB PNG
BANDS = ["5", "8", "13"]
if DATA.startswith("pseudo_rgb"):
    if ":" in DATA:
        BANDS = DATA.split(":", 1)[1].split(",")
        if len(BANDS) != 3:
            raise SystemExit(f"伪RGB 需要正好 3 个波段，收到 {BANDS}")
    DATA = "pseudo_rgb"
elif DATA != "hsi16":
    raise SystemExit(f"未知 DATA={DATA}，只支持 hsi16 / pseudo_rgb[:波段,波段,波段]")
IMAGE_EXT = "npy" if DATA == "hsi16" else "png"
ON_KAGGLE = Path("/kaggle/working").exists()

OUT = Path(os.environ.get("HSI_OUT_DIR") or ("/kaggle/working" if ON_KAGGLE else f"hsi_outputs/{RUN_NAME}")).resolve()
INPUT_ROOT = Path(os.environ.get("HSI_INPUT_ROOT", "/kaggle/input"))

STATUS = {
    "config": {"MODE": MODE, "MODEL": MODEL, "EPOCHS": EPOCHS, "RUN_NAME": RUN_NAME,
               "ATTEMPTS": ATTEMPTS, "MULTISCALE": MULTISCALE, "DATA": DATA, "BANDS": BANDS, "IMGSZ": IMGSZ, "SEED": SEED},
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
    for f in code_root.glob("hsi_detection.*.py"):
        shutil.copy2(f, project / "src" / "hsi_detection" / f.name.removeprefix("hsi_detection."))
    for f in code_root.glob("scripts.*.py"):
        shutil.copy2(f, project / "scripts" / f.name.removeprefix("scripts."))
    manifest_dir = project / "data" / "processed" / "pseudo_rgb"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(code_root / "split_manifest.csv", manifest_dir / "split_manifest.csv")
    env = dict(os.environ, PYTHONPATH=str(project / "src"))
    step("code_restored", code_root=str(code_root), comp_root=str(comp_root))

    # ---------- 4. 生成训练数据（16 波段 NPY 或 伪RGB PNG） ----------
    if DATA == "hsi16":
        data_dir = project / "data" / "processed" / "hsi16_shared_p005_995"
        prepare_cmd = [sys.executable, "scripts/prepare_multispectral.py", "--raw-root", str(comp_root),
                       "--output", str(data_dir), "--workers", str(os.cpu_count() or 4),
                       "--lower-percentile", "0.5", "--upper-percentile", "99.5"]
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

    # ---------- 5. 训练（自动降级） ----------
    data_yaml = data_dir / ("dataset_all.yaml" if MODE == "full" else "dataset.yaml")
    weights = code_root / MODEL
    if not weights.exists():
        weights = Path(MODEL)   # 代码包里没有的官方权重（如 yolo26l.pt）交给 Ultralytics 联网自动下载
    run_dir = project / "runs" / RUN_NAME
    used = None
    for batch, workers, device in ATTEMPTS:
        shutil.rmtree(run_dir, ignore_errors=True)
        cmd = [sys.executable, "scripts/train_baseline.py", "--model", str(weights), "--data", str(data_yaml),
               "--epochs", str(EPOCHS), "--imgsz", str(IMGSZ), "--batch", str(batch), "--device", device,
               "--workers", str(workers), "--seed", str(SEED), "--name", RUN_NAME]
        if MODE == "full":
            cmd.append("--no-val")   # 全量训练时验证集已在训练集里，逐轮验证无意义
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
    step("trained", batch=used[0], workers=used[1], device=used[2])

    # ---------- 6. 推理 + 校验 ----------
    submission = OUT / f"submission_{RUN_NAME}.csv"
    cmd = [sys.executable, "scripts/predict_submission.py", "--weights", str(last),
           "--images", str(data_dir / "images" / "test"), "--output", str(submission),
           "--input-format", IMAGE_EXT, "--batch", "1", "--device", "0", "--half",
           "--conf", "0.0001", "--iou", "0.70", "--max-det", "300", "--imgsz", str(IMGSZ)]
    if MULTISCALE:
        cmd += ["--multi-scale", *MULTISCALE_SIZES, "--fusion-iou", "0.70"]
    t0 = time.time()
    code, tail = run(cmd, OUT / "predict.log", project, env)
    if code != 0:
        raise RuntimeError("推理失败\n" + tail)
    # 校验脚本的 --images 默认指向本地才有的 pseudo_rgb 目录，必须显式指定，
    # 否则读不到图片尺寸，会把每一行都判为"未知图片 ID"
    code, tail = run([sys.executable, "scripts/check_submission.py", str(submission),
                      "--images", str(data_dir / "images" / "test")],
                     OUT / "check.log", project, env)
    step("predicted", sec=round(time.time() - t0), multiscale=MULTISCALE, check=tail.strip()[-200:])
    if code != 0:
        raise RuntimeError("提交文件校验失败\n" + tail)

    if ON_KAGGLE:
        shutil.rmtree(project, ignore_errors=True)   # Kaggle 上 16 波段数据不进输出，避免产物过大


if __name__ == "__main__":
    main()
