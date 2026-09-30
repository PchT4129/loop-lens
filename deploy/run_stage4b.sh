#!/usr/bin/env bash
# 阶段 4b 一键运行：ONNX 导出 → TRT 引擎构建 → 精度评估 → 延迟/稳定性/剖析 → 显存。
# 输出汇总到 deploy/results/stage4b.log。约 10–20 分钟，期间不要运行其他 GPU 任务。
#
#   bash deploy/run_stage4b.sh
#
# 遇错即停（set -e + pipefail）：第一版没有 -e，bench_trt 崩溃后脚本照常跑完、打印"完成"，
# 失败被吞掉，是后来发现 trt_r4.json 没生成才察觉的。现在任何一步的 Python 失败都会让脚本立刻停下。
# 过滤用的 grep 包在 `{ ... || true; }` 里：它什么都没过滤出来时返回 1，不应被当成失败；
# 而 pipefail 取管道里最右边的非零状态，所以前面 Python 的失败仍会传出来。
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-$HOME/miniforge3/envs/vpr-deploy/bin/python}"
export PY
LOG=deploy/results/stage4b.log
: > "$LOG"
step() { echo; echo "########## $* ##########" | tee -a "$LOG"; }
run() { "$@" 2>&1 | { grep -v -E "xFormers|Not enough SMs" || true; } | tee -a "$LOG"; }

step "1/7 导出 ONNX（batch 1 与 32）"
run "$PY" -m deploy.export.to_onnx --batch 1 32

step "2/7 构建 TensorRT 引擎（FP32 / FP16 × batch 1 / 32）"
run "$PY" -m deploy.export.to_trt --precision fp32 fp16 --batch 1 32

step "3/7 用 TRT 引擎抽特征（batch 1）"
run "$PY" -m deploy.eval.runtimes extract trt-fp32 trt-fp16

step "4/7 特征级对比：相对 FP32 参考；以及 TRT FP32 相对 PyTorch FP32（导出正确性核对）"
run "$PY" -m deploy.eval.compare_features deploy/results/runtime/trt_fp32 deploy/results/runtime/trt_fp16 --out deploy/results/compare_r4.json
run "$PY" -m deploy.eval.compare_features deploy/results/runtime/trt_fp32 --ref deploy/results/runtime/torch_fp32

step "5/7 VPR 协议（冻结 FP32 门控）"
G=deploy/results/protocol/ref_fp32/gate_similarity.json
for r in trt_fp32 trt_fp16; do
  bash deploy/eval/run_protocol.sh "r4_${r}" "deploy/results/runtime/${r}_database.pt" \
    "deploy/results/runtime/${r}_query.pt" "$G" > /dev/null 2>&1      # 失败会直接停（run_protocol.sh 也是遇错即停）
  { echo "== $r =="
    { grep -hE "^recall@(1|5):" "deploy/results/protocol/r4_${r}/1_closed_heldout.log" || true; } | tr '\n' ' '
    { grep -hE "best_f1:|oracle/open_set/recall@100" "deploy/results/protocol/r4_${r}/2_oracle_openset.log" || true; } | tr '\n' ' '; echo
    { grep -hE "deployed/open_set/(f1|FP):|f1/ci95" "deploy/results/protocol/r4_${r}/4_frozen_test.log" || true; } | tr '\n' ' '; echo
  } | tee -a "$LOG"
done

step "6/7 延迟（配对）、稳定性、剖析"
run "$PY" -u -m deploy.bench.bench_trt

step "7/7 显存（每个变体独立进程）"
run "$PY" -m deploy.bench.memory_isolated

echo; echo "完成。汇总日志：$LOG"
