#!/usr/bin/env bash
# 消融实验：逐项加入改动，量化每一项的独立贡献。
#
# 所有配置都在【严格 held-out 段 070-099】上评测——该段从未参与任何训练。
# 用法：bash scripts/run_ablation.sh [night_right|day_right]

set -euo pipefail

SPLIT="${1:-night_right}"
EVAL="python -m src.evaluate --top-k 10 --tolerance 3 --split-name ${SPLIT} --min-index 70 --max-index 99"

BASE_DB=outputs/gardens_database_features.pt
BASE_Q=outputs/gardens_query_features.pt
V1_DB=outputs/gardens_database_features_triplet_split.pt
V1_Q=outputs/gardens_query_features_triplet_split.pt
V2_DB=outputs/db_v2.pt
V2_Q=outputs/q_v2.pt

banner() { echo; echo "############ $1 ############"; }

banner "① Baseline：预训练 ResNet18，零训练"
$EVAL --database $BASE_DB --query $BASE_Q

banner "② v1 Triplet：原实现（5 epoch 固定 / 无验证 / 无增强 / Adam / 三次 forward）"
$EVAL --database $V1_DB --query $V1_Q

banner "③ v2 Triplet：BN修复 + ColorJitter + AdamW + 验证集早停 + database purged gap"
$EVAL --database $V2_DB --query $V2_Q

banner "④ ③ + 序列匹配（因果，仅用过去帧 —— 在线 SLAM 的真实条件）"
$EVAL --database $V2_DB --query $V2_Q --seq-window 15 --seq-causal

banner "⑤ ③ + 序列匹配（非因果，离线/批处理条件）"
$EVAL --database $V2_DB --query $V2_Q --seq-window 5

banner "⑥ ⑤ + 几何验证（门控模式：只做接受/拒绝，不改排序）"
$EVAL --database $V2_DB --query $V2_Q --seq-window 5 --geometric-verify --geometric-mode gate

banner "⑦ 对照：几何验证用作【重排】——实测有害，保留在表里作为负面结果"
$EVAL --database $V2_DB --query $V2_Q --seq-window 5 --geometric-verify --geometric-mode rerank

echo
echo "说明：几何验证在跨昼夜配对上判别力接近随机（正确候选 25.1 内点 vs 错误候选 24.4），"
echo "详见 src/geometric_verification.py 顶部与 docs/experiments.md。"
echo "换成 day_right 跑本脚本可以看到同域下它真正有信号的样子。"
