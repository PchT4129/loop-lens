#!/usr/bin/env bash
# 在一组特征上跑仓库原有的 VPR 评测协议——一行不改，只是把命令串起来。
#
# 为什么是 shell 而不是重写成 Python：精度评估必须复用 src.evaluate 的原有实现。
# 任何"我自己重新算一遍 F1"的做法都会引入第二份实现，两份一旦不一致，
# 就分不清差异来自量化还是来自我的代码。
#
# 用法：
#   bash deploy/eval/run_protocol.sh <tag> <db.pt> <query.pt> [frozen_gate.json]
#
#   不传 frozen_gate：在本组特征的 validation 段上拟合 gate 并保存（参考档用）
#   传 frozen_gate  ：直接应用给定的 gate（阶段 4：FP32 上拟合、低精度特征上应用）
#
# 四步：
#   1 闭集 held-out   night_right 070-099，30 query          → R@1 / R@5 / P@5 / R@100%P
#   2 历史开集 oracle  night_right 全 100，排除 db 30-49      → oracle best F1（即 0.874 那个数）
#   3 拟合 gate       validation 000-049，排除 db 30-39      → 冻结阈值（若未给定）
#   4 冻结 gate 测试  test 050-099，排除 db 70-79，bootstrap → deployed/* 主指标
set -euo pipefail

TAG="$1"; DB="$2"; Q="$3"; FROZEN="${4:-}"
PY="${PY:-python}"
OUT="deploy/results/protocol/$TAG"
mkdir -p "$OUT"

EVAL="$PY -m src.evaluate --database $DB --query $Q --top-k 10 --tolerance 3 --split-name night_right"
BOOT="${BOOT:-5000}"          # 与 README 的 CI 用同样的 bootstrap 次数，数字才可直接对照

echo "=== [$TAG] 1/4 闭集 held-out 070-099 ==="
$EVAL --min-index 70 --max-index 99 2>&1 | grep -v xFormers | tee "$OUT/1_closed_heldout.log" \
  | grep -E "recall@(1|5):|precision@5|recall@100" || true

echo "=== [$TAG] 2/4 历史开集 oracle（全 100 query，排除 db 30-49）==="
$EVAL --db-exclude-range 30 49 2>&1 | grep -v xFormers | tee "$OUT/2_oracle_openset.log" \
  | grep -E "best_f1|recall@100" || true

if [[ -z "$FROZEN" ]]; then
  FROZEN="$OUT/gate_similarity.json"
  echo "=== [$TAG] 3/4 在 validation 上拟合 gate → $FROZEN ==="
  $EVAL --min-index 0 --max-index 49 --db-exclude-range 30 39 \
    --gate-mode similarity --fit-thresholds --thresholds-out "$FROZEN" 2>&1 \
    | grep -v xFormers > "$OUT/3_fit_gate.log"
  grep -E "threshold|f1" "$OUT/3_fit_gate.log" | head -5 || true
else
  echo "=== [$TAG] 3/4 跳过拟合，使用冻结 gate：$FROZEN ==="
fi

echo "=== [$TAG] 4/4 冻结 gate 测试 050-099（bootstrap $BOOT）==="
$EVAL --min-index 50 --max-index 99 --db-exclude-range 70 79 \
  --thresholds-in "$FROZEN" --bootstrap-samples "$BOOT" ${EXTRA_EVAL_ARGS:-} 2>&1 \
  | grep -v xFormers | tee "$OUT/4_frozen_test.log" \
  | grep -E "recall@1:|deployed/open_set/(precision|recall|f1|FP)|auprc" || true

echo "日志：$OUT/"
