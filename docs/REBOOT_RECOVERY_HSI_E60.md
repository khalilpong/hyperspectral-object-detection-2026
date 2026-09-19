# HSI16 e60 重启恢复说明

> ✅ **已结束（2026-09-16）**：该训练已跑满 60 epoch，无需再恢复。
> 结论是**否定结果**（e60 相对 e30 仅 +0.00093，低于轮间标准差 0.00158），
> 详见 [`experiments/ablation-s1024-hsi16-e60-r2.md`](../experiments/ablation-s1024-hsi16-e60-r2.md)。
> 下方内容保留作为「长训练中断恢复」的通用操作参考。

## 当前唯一活动训练

- 队列：`artifacts/background_queue/hsi_e60_validation_2026-09-15/`
- 运行名：`ablation_s1024_b4_hsi16_shared_p005_995_e60_r2`
- 目标：60 epoch
- 结果：`runs/ablation_s1024_b4_hsi16_shared_p005_995_e60_r2/results.csv`
- 断点：`runs/ablation_s1024_b4_hsi16_shared_p005_995_e60_r2/weights/last.pt`
- 数据：`data/processed/hsi16_shared_p005_995/dataset.yaml`
- 固定安全设置：`batch=4`、`imgsz=1024`、`workers=0`、`seed=2026`

最新状态（2026-09-15 11:56，UTC+08:00）：已按用户要求在 epoch 13 完整写盘后暂停；最佳 mAP50-95 为 `0.65381`。`last.pt` 大小 80,539,205 字节，SHA-256 为 `1A04377A1F9EE519679901C8092DFE06396B6513DBF490F9E5B68B58E080745E`。runner 与训练 Python 均已退出，队列状态和 `queue.paused` 都记录 epoch 13。

Ultralytics 在每个完整 epoch 后保存 `last.pt`。电脑突然重启最多丢失正在进行的那一个 epoch；已经写入 `results.csv` 的 epoch 可以从 `last.pt` 继续。

## 重启前查看精确进度

```powershell
Set-Location D:\kaggle\hyperspectral-object-detection-2026
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\check_hsi_e60_progress.ps1
```

输出中的 `completed_epoch` 必须与 `results.csv` 最后一行一致。需要备份哈希时可加 `-HashCheckpoint`；如果恰逢 epoch 写盘，脚本会拒绝给出不稳定哈希。

## 重启后安全续跑

```powershell
Set-Location D:\kaggle\hyperspectral-object-detection-2026
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\resume_hsi_e60_after_reboot.ps1
```

脚本是幂等的：

- 已完成 60 轮时不启动任何进程。
- 已检测到同一个 runner 仍在运行时拒绝重复启动。
- 断点小于 10 MB 或仍在变化时拒绝恢复。
- 如果中断留下失败 sentinel，会先保留为带时间戳的 `.bak`，再复用原队列和原 run 目录。
- 后台窗口用隐藏模式启动；日志写回原队列目录。

恢复后再次运行进度检查脚本，确认 `runner_identity_matches=true`。不要另开第二个训练命令，也不要把 `workers` 改回大于 0。

如果训练是通过 `pause_hsi_e60_at_epoch.ps1` 主动暂停，队列中会有 `queue.paused`。续跑脚本会先把该标记保留为带时间戳的 `.bak`，然后从稳定的 `last.pt` 继续。

## 本地恢复备份

恢复 ZIP 由当前工作区生成，文件名以 `private_recovery_hsi_e60_` 开头。它仅供本人防重启/磁盘误操作使用，不得在比赛期间私下发给其他参赛队伍。ZIP 不包含原始图像数据或 `.venv`；实际续跑仍依赖本机现有的数据目录和 Python 环境。
