#!/usr/bin/env bash
# 生成八成员融合提交文件（Kaggle 0.64831 的配方）。在项目根目录用 Git Bash 运行：
#
#   bash scripts/build_ensemble_submission.sh submissions/my_try.csv
#
# 调权重：用环境变量覆盖，例如把 G 成员降到 0.5：
#
#   W_G=0.5 bash scripts/build_ensemble_submission.sh submissions/try_g05.csv
#
# 轻量支持票加成（留出集推荐 0.125；0 保持旧算法逐字节不变）：
#
#   SUPPORT_GAIN=0.125 bash scripts/build_ensemble_submission.sh submissions/try_sg0125.csv
#
# 可选加入 P1-P99 HSI16 新成员（训练完成并下载后）：
#
#   P1_WEIGHTS=kaggle_remote/outputs/hsi16-p010-990-full/kaggle_full_yolo26m_hsi16_p010_990_e30/last.pt \
#   W_I=0.4 SUPPORT_GAIN=0.125 bash scripts/build_ensemble_submission.sh submissions/try_p1_w04_sg0125.csv
#
# 为什么这么快：artifacts/ensemble_cache/<标签>.pkl 里已经缓存了每个成员在测试集上 7 个尺度的预测，
# 只要标签没变就不会再跑 GPU，只做融合（约 10 分钟）。**换了权重文件或图片目录，必须换标签或删对应 pkl**。
# 想加/删成员，直接改下面的 --model 行；每个成员格式：标签=权重文件@置信度系数#该成员的图片数据集根目录
#（16 波段成员不写 #...；第一个成员是基准，系数固定 1.0，必须是 16 波段成员）。
set -euo pipefail

OUT="${1:?用法: bash scripts/build_ensemble_submission.sh 输出.csv}"

# 各成员置信度系数（A 固定 1.0）
W_B="${W_B:-0.6}"   # s   16 波段
W_C="${W_C:-0.8}"   # m   伪RGB 5/8/13
W_D="${W_D:-0.6}"   # s   伪RGB 5/8/13
W_E="${W_E:-0.5}"   # m   16 波段 batch2（弱）
W_F="${W_F:-0.5}"   # s   16 波段 伪标签（弱）
W_G="${W_G:-0.8}"   # m   伪RGB 3/6/8
W_H="${W_H:-0.8}"   # m   伪RGB 0/7/15
W_I="${W_I:-0.4}"   # m   16 波段 P1-P99（可选；必须同时设置 P1_WEIGHTS）
FUSION_IOU="${FUSION_IOU:-0.70}"
SUPPORT_GAIN="${SUPPORT_GAIN:-0}"

P1_MODEL=()
if [[ -n "${P1_WEIGHTS:-}" ]]; then
  [[ -f "${P1_WEIGHTS}" ]] || { echo "P1_WEIGHTS 不存在：${P1_WEIGHTS}" >&2; exit 2; }
  [[ -d data/processed/hsi16_shared_p010_990/images/test ]] || {
    echo "缺少 P1-P99 测试数据：data/processed/hsi16_shared_p010_990/images/test" >&2
    exit 2
  }
  P1_MODEL=(--model "m_p010_990_full=${P1_WEIGHTS}@${W_I}#data/processed/hsi16_shared_p010_990")
fi

PYTHONPATH=src .venv/Scripts/python.exe -u scripts/eval_ensemble.py \
  --data data/processed/hsi16_shared_p005_995/dataset.yaml \
  --model "m_full=kaggle_remote/outputs/full/kaggle_full_yolo26m_e30/last.pt" \
  --model "s_full=runs/final_s1024_all3000_hsi16_shared_p005_995_r1/weights/last.pt@${W_B}" \
  --model "prgb_full=kaggle_remote/outputs/prgb-full/kaggle_full_yolo26m_prgb_e30/last.pt@${W_C}#data/processed/pseudo_rgb" \
  --model "prgbs_full=runs/final_s1024_all3000_b5-8-13_r1/weights/last.pt@${W_D}#data/processed/pseudo_rgb" \
  --model "mb2_full=runs/final_m1024_all3000_hsi16_shared_p005_995_r1/weights/last.pt@${W_E}" \
  --model "spseudo_full=runs/final_s1024_all3000_hsi16_pseudo722_r1/weights/last.pt@${W_F}" \
  --model "prgb368=kaggle_remote/outputs/prgb368-full/kaggle_full_yolo26m_prgb368_e30/last.pt@${W_G}#data/processed/pseudo_rgb_b3-6-8" \
  --model "prgb0715=kaggle_remote/outputs/prgb0715-full/kaggle_full_yolo26m_prgb0715_e30/last.pt@${W_H}#data/processed/pseudo_rgb_b0-7-15" \
  "${P1_MODEL[@]}" \
  --images data/processed/hsi16_shared_p005_995/images/test \
  --fusion-ious "${FUSION_IOU}" \
  --support-gain "${SUPPORT_GAIN}" \
  --submission "${OUT}" \
  --output artifacts/ensemble/unused.json 2>&1 | grep -v WARNING

.venv/Scripts/python.exe scripts/check_submission.py "${OUT}" \
  --images data/processed/hsi16_shared_p005_995/images/test | tail -1
