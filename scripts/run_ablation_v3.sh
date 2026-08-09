#!/usr/bin/env bash
# v3 消融：经典 → 现代，逐级替换并量化。
#
# 用法：
#   bash scripts/run_ablation_v3.sh            # 全部
#   bash scripts/run_ablation_v3.sh extract    # 只抽特征
#   bash scripts/run_ablation_v3.sh closed     # 只跑闭集表
#   bash scripts/run_ablation_v3.sh openset    # 只跑开集表
#
# 说明：
#   * 闭集表在【held-out 070-099】上评测（30 query），因为涉及微调过的模型。
#   * 零训练的配置（baseline / GeM / DINOv2）额外在全部 100 query 上报一次——
#     它们从未见过任何训练数据，所以整个数据集对它们都是 held-out，样本量大 3.3 倍。
#   * 开集表移除 database 帧号 30-49，使 14/100 个 query 没有正确答案。

set -euo pipefail

OUT=outputs/v3
CKPT=outputs/checkpoints
mkdir -p "$OUT" "$CKPT"

TRAIN_ARGS="--anchor-dir data/gardens_point/query/night_right \
  --database-dir data/gardens_point/database/day_left \
  --epochs 8 --min-index 0 --max-index 59 --db-max-index 56 \
  --val-min-index 60 --val-max-index 69 --seed 0"

banner() { echo; echo "############ $1 ############"; }

# ---------------------------------------------------------------- 抽特征
do_extract() {
  banner "抽特征"

  # ① baseline：预训练 ResNet18 + GAP
  for d in database query; do
    [ -f "$OUT/base_$d.pt" ] || python -m src.extract_features \
      --image-dir "data/gardens_point/$d" --output "$OUT/base_$d.pt"
  done

  # ② GeM 池化（零训练，只换池化）
  for d in database query; do
    [ -f "$OUT/gem_$d.pt" ] || python -m src.extract_features \
      --image-dir "data/gardens_point/$d" --output "$OUT/gem_$d.pt" --pooling gem
  done

  # ③ triplet 微调
  [ -f "$CKPT/v3_triplet.pt" ] || python -m src.train_triplet $TRAIN_ARGS \
    --loss triplet --early-stop-patience 4 --output "$CKPT/v3_triplet.pt"
  for d in database query; do
    [ -f "$OUT/triplet_$d.pt" ] || python -m src.extract_features \
      --image-dir "data/gardens_point/$d" --output "$OUT/triplet_$d.pt" \
      --checkpoint "$CKPT/v3_triplet.pt"
  done

  # ④ InfoNCE 微调（batch=60 全批量 + tau=0.2，见 experiments.md 的诊断）
  [ -f "$CKPT/v3_infonce.pt" ] || python -m src.train_triplet $TRAIN_ARGS \
    --loss infonce --batch-size 60 --tau 0.2 --epochs 6 --output "$CKPT/v3_infonce.pt"
  for d in database query; do
    [ -f "$OUT/infonce_$d.pt" ] || python -m src.extract_features \
      --image-dir "data/gardens_point/$d" --output "$OUT/infonce_$d.pt" \
      --checkpoint "$CKPT/v3_infonce.pt"
  done

  # ⑤ DINOv2 零训练（三种聚合方式）
  for agg in cls mean gem; do
    for d in database query; do
      [ -f "$OUT/dino_${agg}_$d.pt" ] || python -m src.extract_features \
        --image-dir "data/gardens_point/$d" --output "$OUT/dino_${agg}_$d.pt" \
        --backbone dinov2_vits14 --dinov2-aggregation "$agg"
    done
  done
}

# ---------------------------------------------------------------- 闭集表
EVAL="python -m src.evaluate --top-k 10 --tolerance 3 --split-name night_right"

row() {  # row <名称> <特征前缀> [额外参数...]
  local name="$1" prefix="$2"; shift 2
  echo "--- $name ---"
  $EVAL --database "$OUT/${prefix}_database.pt" --query "$OUT/${prefix}_query.pt" \
    --min-index 70 --max-index 99 "$@" 2>&1 \
    | grep -E "recall@[0-9]|precision@|recall@100"
}

do_closed() {
  banner "闭集消融（held-out night_right 070-099，30 query）"
  row "① ResNet18-GAP 预训练"        base
  row "② ResNet18-GeM 预训练(零训练)" gem
  row "③ ResNet18-GAP + triplet"     triplet
  row "④ ResNet18-GAP + InfoNCE"     infonce
  row "⑤ DINOv2 CLS 零训练"          dino_cls
  row "⑥ DINOv2 patch-mean 零训练"   dino_mean
  row "⑦ ⑥ + 因果序列匹配"           dino_mean --seq-window 15 --seq-causal
  row "⑧ ⑥ + 非因果序列匹配"         dino_mean --seq-window 5
}

# ---------------------------------------------------------------- 开集表
do_openset() {
  banner "开集评测（移除 database 帧号 30-49，14/100 query 应被拒绝）"
  for cfg in "① ResNet18 预训练:base" "② ResNet18+triplet:triplet" "③ DINOv2:dino_mean"; do
    name="${cfg%%:*}"; prefix="${cfg##*:}"
    echo "--- $name ---"
    $EVAL --database "$OUT/${prefix}_database.pt" --query "$OUT/${prefix}_query.pt" \
      --db-exclude-range 30 49 2>&1 \
      | grep -E "recall@1:|open_set/(best_f1:|recall@100)"
  done

  echo
  echo "--- DINOv2 上逐级叠加（含门控） ---"
  for cfg in "单帧:" \
             "+因果序列:--seq-window 15 --seq-causal" \
             "+序列+ORB门控:--seq-window 15 --seq-causal --geometric-verify --verifier orb" \
             "+序列+LightGlue门控:--seq-window 15 --seq-causal --geometric-verify --verifier lightglue"; do
    name="${cfg%%:*}"; extra="${cfg#*:}"
    echo "  [$name]"
    $EVAL --database "$OUT/dino_mean_database.pt" --query "$OUT/dino_mean_query.pt" \
      --db-exclude-range 30 49 $extra 2>&1 \
      | grep -E "recall@1:|open_set/(best_f1:|recall@100)" | sed 's/^/    /'
  done
}

case "${1:-all}" in
  extract) do_extract ;;
  closed)  do_closed ;;
  openset) do_openset ;;
  all)     do_extract; do_closed; do_openset ;;
  *) echo "用法: $0 [all|extract|closed|openset]"; exit 1 ;;
esac

echo
echo "完整分析见 docs/experiments.md"
