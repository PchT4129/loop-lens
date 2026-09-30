"""阶段 4b · R4：TensorRT FP16 的延迟、稳定性与剖析。

对照对象是 4a 的最优档：FP16 torch.compile `reduce-overhead`（b1 ≈ 0.86 ms，已等于 PyTorch 的 GPU 工作量）。

三部分：
  1 配对延迟：compile-RO vs TRT FP16（b1、b32）；TRT FP16 vs TRT FP16 + CUDA Graph（b1）
  2 稳定性：每个变体一次长测（1000 次迭代），报 p50 / p99 / p99÷p50——4a 发现 CUDA Graph 让延迟更稳
  3 剖析：TRT 每次前向的 kernel 数、GPU 工作量、GPU 忙碌占比（墙钟取自第 1 部分，不开 profiler）

TRT 引擎能不能被 CUDA Graph 录制：TRT 的 execute_async_v3 只是把 kernel 发射到给定的 stream 上，
可以在 PyTorch 的图录制上下文里调用。先热身几次再录制（首次执行可能有一次性的准备工作）。
"""

from __future__ import annotations

import copy
import json
import statistics
import sys
import warnings
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from deploy.bench.bench_torch import _paired_rounds                # noqa: E402
from deploy.bench.timing import ramp_clocks, time_cuda              # noqa: E402
from deploy.eval.runtimes import _base_fp32                         # noqa: E402
from deploy.export.trt_runner import TRTRunner                      # noqa: E402
from deploy.profile.profile_torch import _is_kernel                 # noqa: E402

warnings.filterwarnings("ignore")
RESULTS = REPO / "deploy" / "results"
ENG = RESULTS / "engines"


def compiled_ro_fp16():
    torch._dynamo.reset()
    m = copy.deepcopy(_base_fp32()).half().eval()
    return torch.compile(m, mode="reduce-overhead", fullgraph=True, dynamic=False)


def graph_wrap(runner: TRTRunner, x: torch.Tensor):
    """把一次 TRT 执行录成 CUDA Graph；返回重放函数。失败时返回 (None, 错误信息)。"""
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(5):
                runner(x)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            runner(x)
        return g.replay, None
    except Exception as exc:                                        # noqa: BLE001
        return None, f"{type(exc).__name__}: {str(exc)[:300]}"


def kernels_per_forward(fn, n: int = 10) -> dict:
    with torch.inference_mode():
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(n):
                fn()
            torch.cuda.synchronize()
    ks = [e for e in prof.events() if _is_kernel(e)]
    return {"kernels_per_forward": round(len(ks) / n, 1),
            "kernel_ms_per_forward": round(sum(k.time_range.elapsed_us() for k in ks) / n / 1000, 4)}


def main() -> None:
    out: dict = {"latency": [], "stability": [], "profile": []}
    x0 = torch.randn(1, 3, 224, 224, device="cuda", dtype=torch.float16)
    comp = compiled_ro_fp16()
    with torch.inference_mode():
        for _ in range(5):
            comp(x0)
        time_cuda(lambda: comp(x0), warmup=20, iters=50, sample_conditions=False)   # 丢弃首测
    ramp_clocks()

    # ✏️ 第一版的两个错误（都已修正）：
    #  1. 用 PyTorch 默认 stream 调 execute_async_v3，TRT 警告"会额外调用 cudaStreamSynchronize"——
    #     每次调用都同步，CPU 与 GPU 失去并行，b1 延迟被人为抬高（测出 1.158 ms，而 GPU 工作量只有 0.53 ms）。
    #     现在全部测量都放在一个非默认 stream 上；计时的 CUDA Event 记录在同一条 stream 上。
    #  2. 在一个循环里先测 b1 再测 b32，lambda 引用的 x / trt16 被循环重新赋值（Python 闭包延迟绑定），
    #     "b1 稳定性"实际测的是 b32；更糟的是 b1 的 TRT 执行器因此被回收，而录好的 CUDA Graph 仍指向它的
    #     显存——重放时进程崩溃。现在 b1、b32 的对象分开命名、全程持有引用，b1 全部测完再测 b32。
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        # ---------------- batch 1 ----------------
        x1 = torch.randn(1, 3, 224, 224, device="cuda", dtype=torch.float16)
        trt1 = TRTRunner(ENG / "vits14_mean_224_b1_fp16.engine")
        with torch.inference_mode():
            for _ in range(5):
                comp(x1)
        run_comp1 = lambda: comp(x1)                   # noqa: E731
        run_trt1 = lambda: trt1(x1)                    # noqa: E731
        replay1, err = graph_wrap(trt1, x1)
        if replay1 is None:
            print(f"b1  TRT FP16 + CUDA Graph：录制失败 {err}")
            out["trt_graph_error"] = err
        variants_b1 = {"compile-RO-fp16": run_comp1, "trt-fp16": run_trt1}
        if replay1 is not None:
            variants_b1["trt-fp16+cudagraph"] = replay1

        pairs = [("compile-RO-fp16", "trt-fp16")]
        if replay1 is not None:
            pairs += [("trt-fp16", "trt-fp16+cudagraph"), ("compile-RO-fp16", "trt-fp16+cudagraph")]
        for a, b in pairs:
            r = _paired_rounds(variants_b1[a], variants_b1[b], 10, 30, 300)
            sp = r["a_median_ms"] / r["b_median_ms"]
            out["latency"].append({"batch": 1, "a": a, "b": b, **r, "speedup_b_over_a": round(sp, 3)})
            print(f"b1   {a:<20} {r['a_median_ms']:7.3f} ms → {b:<20} {r['b_median_ms']:7.3f} ms  "
                  f"×{sp:.2f}（后者更快的轮数 {r['rounds_b_faster']}/10）")

        for name, fn in (("compile-RO-fp16", run_comp1), ("trt-fp16", run_trt1)):
            p = kernels_per_forward(fn)
            out["profile"].append({"batch": 1, "variant": name, **p})
            print(f"b1   剖析 {name:<18} {p['kernels_per_forward']:>6.0f} 个 kernel / 前向 | "
                  f"GPU 工作量 {p['kernel_ms_per_forward']:.3f} ms")

        print("\n--- b1 稳定性（1000 次迭代）---")
        with torch.inference_mode():
            for name, fn in variants_b1.items():
                r = time_cuda(fn, warmup=50, iters=1000, sample_conditions=False)
                out["stability"].append({"variant": name, "p50_ms": round(r.median_ms, 4),
                                         "p99_ms": round(r.p99_ms, 4),
                                         "p99_over_p50": round(r.jitter_ratio, 3)})
                print(f"  {name:<20} p50 {r.median_ms:.3f} ms | p99 {r.p99_ms:.3f} ms | "
                      f"p99/p50 {r.jitter_ratio:.2f}")

        # ---------------- batch 32 ----------------
        x32 = torch.randn(32, 3, 224, 224, device="cuda", dtype=torch.float16)
        trt32 = TRTRunner(ENG / "vits14_mean_224_b32_fp16.engine")
        with torch.inference_mode():
            for _ in range(5):
                comp(x32)                              # 固定形状：b32 会触发一次重编译
        r = _paired_rounds(lambda: comp(x32), lambda: trt32(x32), 6, 30, 100)
        sp = r["a_median_ms"] / r["b_median_ms"]
        out["latency"].append({"batch": 32, "a": "compile-RO-fp16", "b": "trt-fp16", **r,
                               "speedup_b_over_a": round(sp, 3)})
        print(f"\nb32  compile-RO-fp16 {r['a_median_ms']:8.3f} ms → trt-fp16 {r['b_median_ms']:8.3f} ms  "
              f"×{sp:.2f}（后者更快的轮数 {r['rounds_b_faster']}/6）")

    path = RESULTS / "trt_r4.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
