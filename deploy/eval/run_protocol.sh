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
# 每步先把 src.evaluate 的完整输出写进日志（它失败 = 管道失败 = 脚本停下），再从日志里挑几行显示。
# 第一版写成 `$EVAL ... | grep | tee | grep ... || true`，末尾的 `|| true` 连 evaluate 的失败也一起吞掉了。
filter() { { grep -v xFormers || true; }; }
show() { grep -E "$1" "$2" || true; }

TAG="$1"; DB="$2"; Q="$3"; FROZEN="${4:-}"
PY="${PY:-python}"
OUT="deploy/results/protocol/$TAG"
mkdir -p "$OUT"

EVAL="$PY -m src.evaluate --database $DB --query $Q --top-k 10 --tolerance 3 --split-name night_right"
BOOT="${BOOT:-5000}"
RUNTIME_FLAG=""          # 与 README 的 CI 用同样的 bootstrap 次数，数字才可直接对照

echo "=== [$TAG] 1/4 闭集 held-out 070-099 ==="
$EVAL --min-index 70 --max-index 99 2>&1 | filter > "$OUT/1_closed_heldout.log"
show "recall@(1|5):|precision@5|recall@100" "$OUT/1_closed_heldout.log"

echo "=== [$TAG] 2/4 历史开集 oracle（全 100 query，排除 db 30-49）==="
$EVAL --db-exclude-range 30 49 2>&1 | filter > "$OUT/2_oracle_openset.log"
show "best_f1|recall@100" "$OUT/2_oracle_openset.log"

if [[ -z "$FROZEN" ]]; then
  FROZEN="$OUT/gate_similarity.json"
  echo "=== [$TAG] 3/4 在 validation 上拟合 gate → $FROZEN ==="
  $EVAL --min-index 0 --max-index 49 --db-exclude-range 30 39 \
    --gate-mode similarity --fit-thresholds --thresholds-out "$FROZEN" 2>&1 \
    | filter > "$OUT/3_fit_gate.log"
  show "threshold|f1" "$OUT/3_fit_gate.log" | head -5
else
  echo "=== [$TAG] 3/4 跳过拟合，使用冻结 gate：$FROZEN ==="
  # 冻结 gate 通常拟合于 FP32 参考，而这组特征来自另一种执行方式：只放开 runtime 这一个键
  RUNTIME_FLAG="--allow-runtime-mismatch"
fi

echo "=== [$TAG] 4/4 冻结 gate 测试 050-099（bootstrap $BOOT）==="
$EVAL --min-index 50 --max-index 99 --db-exclude-range 70 79 \
  --thresholds-in "$FROZEN" --bootstrap-samples "$BOOT" $RUNTIME_FLAG ${EXTRA_EVAL_ARGS:-} 2>&1 \
  | filter > "$OUT/4_frozen_test.log"
show "recall@1:|deployed/open_set/(precision|recall|f1|FP)|auprc" "$OUT/4_frozen_test.log"

echo "日志：$OUT/"
