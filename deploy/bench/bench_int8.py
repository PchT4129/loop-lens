"""阶段 4c · R5：TensorRT INT8 相对 TensorRT FP16 的延迟、剖析与显存。

对照对象是 4b 的最优档（TRT FP16 + CUDA Graph，b1 0.50–0.53 ms；b32 TRT 7.9 ms）。
方法与 bench_trt.py 完全一致（每一条都来自 4b 踩过的坑）：
  * 全部测量放在非默认 stream 上（默认 stream 会让 TRT 每次额外同步）
  * b1 与 b32 的对象分开命名、全程持有引用（闭包延迟绑定曾让 CUDA Graph 指向已释放的显存）
  * 配对交替多轮（b1 有间歇的 CPU 侧慢状态）
默认测 4c 的主方案（隐式 INT8、Entropy、day 校准）；阶段 5、6 用 `--int8 qdq_<标签>` 测显式 Q/DQ 引擎（INT8 或 FP8）。

用法：
    python -u -m deploy.bench.bench_int8                              # 4c：隐式 INT8
    python -u -m deploy.bench.bench_int8 --int8 qdq_lin_nofc2_sq      # 阶段 5：显式 INT8
    python -u -m deploy.bench.bench_int8 --int8 qdq_fp8_lin           # 阶段 6：FP8（变量名仍叫 int8，测的是 FP8 引擎）
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from deploy.bench.bench_torch import _paired_rounds                # noqa: E402
from deploy.bench.bench_trt import graph_wrap, kernels_per_forward  # noqa: E402
from deploy.bench.timing import ramp_clocks, time_cuda              # noqa: E402
from deploy.export.trt_runner import TRTRunner                      # noqa: E402

warnings.filterwarnings("ignore")
RESULTS = REPO / "deploy" / "results"
ENG = RESULTS / "engines"
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--int8", default="int8_entropy_day", help="引擎标签：vits14_mean_224_b{1,32}_<标签>.engine")
    INT8 = ap.parse_args().int8
    out: dict = {"int8_engine": INT8, "latency": [], "stability": [], "profile": []}
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        # ---------------- batch 1 ----------------
        x1 = torch.randn(1, 3, 224, 224, device="cuda")
        fp16_1 = TRTRunner(ENG / "vits14_mean_224_b1_fp16.engine")
        int8_1 = TRTRunner(ENG / f"vits14_mean_224_b1_{INT8}.engine")
        with torch.inference_mode():
            time_cuda(lambda: fp16_1(x1), warmup=20, iters=50, sample_conditions=False)   # 丢弃首测
        ramp_clocks()
        g16, e16 = graph_wrap(fp16_1, x1)
        g8, e8 = graph_wrap(int8_1, x1)
        if g16 is None or g8 is None:
            raise SystemExit(f"CUDA Graph 录制失败：fp16={e16} int8={e8}")
        run_fp16_1 = lambda: fp16_1(x1)                # noqa: E731
        run_int8_1 = lambda: int8_1(x1)                # noqa: E731
        variants = {"trt-fp16": run_fp16_1, "trt-int8": run_int8_1,
                    "trt-fp16+cudagraph": g16, "trt-int8+cudagraph": g8}

        for a, b in (("trt-fp16+cudagraph", "trt-int8+cudagraph"), ("trt-fp16", "trt-int8")):
            r = _paired_rounds(variants[a], variants[b], 10, 30, 300)
            sp = r["a_median_ms"] / r["b_median_ms"]
            out["latency"].append({"batch": 1, "a": a, "b": b, **r, "speedup_b_over_a": round(sp, 3)})
            print(f"b1   {a:<20} {r['a_median_ms']:7.3f} ms → {b:<20} {r['b_median_ms']:7.3f} ms  "
                  f"×{sp:.2f}（后者更快的轮数 {r['rounds_b_faster']}/10）")

        for name, fn in (("trt-fp16", run_fp16_1), ("trt-int8", run_int8_1)):
            p = kernels_per_forward(fn)
            out["profile"].append({"batch": 1, "variant": name, **p})
            print(f"b1   剖析 {name:<10} {p['kernels_per_forward']:>5.0f} 个 kernel / 前向 | "
                  f"GPU 工作量 {p['kernel_ms_per_forward']:.3f} ms")

        with torch.inference_mode():
            for name in ("trt-fp16+cudagraph", "trt-int8+cudagraph"):
                r = time_cuda(variants[name], warmup=50, iters=1000, sample_conditions=False)
                out["stability"].append({"variant": name, "p50_ms": round(r.median_ms, 4),
                                         "p99_ms": round(r.p99_ms, 4), "p99_over_p50": round(r.jitter_ratio, 3)})
                print(f"b1   稳定性 {name:<20} p50 {r.median_ms:.3f} | p99 {r.p99_ms:.3f} ms")

        # ---------------- batch 32 ----------------
        x32 = torch.randn(32, 3, 224, 224, device="cuda")
        fp16_32 = TRTRunner(ENG / "vits14_mean_224_b32_fp16.engine")
        int8_32 = TRTRunner(ENG / f"vits14_mean_224_b32_{INT8}.engine")
        r = _paired_rounds(lambda: fp16_32(x32), lambda: int8_32(x32), 6, 30, 100)
        sp = r["a_median_ms"] / r["b_median_ms"]
        out["latency"].append({"batch": 32, "a": "trt-fp16", "b": "trt-int8", **r, "speedup_b_over_a": round(sp, 3)})
        print(f"b32  trt-fp16 {r['a_median_ms']:8.3f} ms → trt-int8 {r['b_median_ms']:8.3f} ms  "
              f"×{sp:.2f}（后者更快的轮数 {r['rounds_b_faster']}/6）")
        for name, fn in (("trt-fp16", lambda: fp16_32(x32)), ("trt-int8", lambda: int8_32(x32))):
            p = kernels_per_forward(fn)
            out["profile"].append({"batch": 32, "variant": name, **p})
            print(f"b32  剖析 {name:<10} {p['kernels_per_forward']:>5.0f} 个 kernel / 前向 | "
                  f"GPU 工作量 {p['kernel_ms_per_forward']:.3f} ms")

    path = RESULTS / ("trt_int8_r5.json" if INT8 == "int8_entropy_day" else f"trt_{INT8}.json")
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已写入 {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
