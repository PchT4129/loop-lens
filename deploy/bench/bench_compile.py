"""阶段 4a · R2：torch.compile 各模式 vs eager —— 延迟（配对）、显存（单独驻留）、编译耗时。

三种模式各是什么
----------------
  default          把模型捕获成计算图，交给 Inductor 生成**融合后**的 kernel（主要减少 kernel 数和访存）
  reduce-overhead  在 default 基础上再用 CUDA Graph：整串 kernel 录制一次、以后一次性重放
  max-autotune     在 reduce-overhead 基础上，为每个算子多试几种实现挑最快的。
                   ⚠️ 本卡只有 46 个 SM，低于 Inductor 的 68 SM 门槛，**矩阵乘不参与自动调优**

测量设计
--------
* 延迟：eager 与编译版本**配对交替**多轮（阶段 3：batch 1 有间歇性的 CPU 侧慢状态，配对才公平）
* 显存：每个变体**单独驻留**时测（阶段 2 的教训：同时驻留会把峰值加在一起）
* 编译耗时：每个 (模式, batch) 第一次调用的墙钟。运行时用全新的 TORCHINDUCTOR_CACHE_DIR，
  所以是冷启动；同一进程内后编译的模式可能复用先编译模式的部分产物，按编译顺序如实记录
"""

from __future__ import annotations

import copy
import json
import statistics
import sys
import time
import warnings
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from deploy.bench.bench_torch import _paired_rounds, weight_mb        # noqa: E402
from deploy.bench.timing import ramp_clocks, time_cuda                # noqa: E402
from deploy.eval.runtimes import DT, _base_fp32                       # noqa: E402

warnings.filterwarnings("ignore")
RESULTS = REPO / "deploy" / "results"
MODES = ("default", "reduce-overhead", "max-autotune")


def compile_model(m, mode):
    # 第一次运行在第 9 个编译版本时崩了：Dynamo 对同一段代码（VPRDescriptor.forward）最多缓存 8 个
    # 编译版本（recompile_limit=8），每个 (模型实例, 精度, 形状) 都占一个。真实部署只编译一个版本，
    # 这是测量脚本把太多变体塞进一个进程造成的。每编译一个新变体前清空缓存，各变体互不干扰。
    torch._dynamo.reset()
    return torch.compile(m, mode=None if mode == "default" else mode, fullgraph=True, dynamic=False)


def first_call_s(fn, x) -> float:
    t0 = time.perf_counter()
    with torch.inference_mode():
        fn(x)
        torch.cuda.synchronize()
    return round(time.perf_counter() - t0, 2)


def latency(precision: str, base) -> list[dict]:
    rows = []
    eager = copy.deepcopy(base).to(DT[precision]).eval()
    for mode in MODES:
        comp = compile_model(copy.deepcopy(eager), mode)
        for batch, rounds, iters in ((1, 10, 300), (32, 6, 100)):
            x = torch.randn(batch, 3, 224, 224, device="cuda", dtype=DT[precision])
            c_s = first_call_s(comp, x)
            with torch.inference_mode():
                for _ in range(5):                    # CUDA Graph 需要先热身再录制
                    comp(x)
                torch.cuda.synchronize()
            r = _paired_rounds(lambda: eager(x), lambda: comp(x), rounds, 30, iters)
            speed = r["a_median_ms"] / r["b_median_ms"]
            rows.append({"precision": precision, "mode": mode, "batch": batch,
                         "compile_first_call_s": c_s, **r, "speedup": round(speed, 3)})
            print(f"  {precision} {mode:<16} b{batch:<3} 编译 {c_s:>6.1f} s | eager {r['a_median_ms']:8.3f} → "
                  f"编译 {r['b_median_ms']:8.3f} ms  ×{speed:.2f}  "
                  f"（编译更快的轮数 {r['rounds_b_faster']}/{rounds}）")
        del comp
        torch.cuda.empty_cache()
    return rows


def memory(precision: str, base) -> list[dict]:
    """单独驻留：每次只有一个模型在显存里。"""
    rows = []
    for mode in ("eager",) + MODES:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        m = copy.deepcopy(base).to(DT[precision]).eval()
        fn = m if mode == "eager" else compile_model(m, mode)
        x = torch.randn(1, 3, 224, 224, device="cuda", dtype=DT[precision])
        with torch.inference_mode():
            for _ in range(10):
                fn(x)
            torch.cuda.synchronize()
            r = time_cuda(lambda: fn(x), warmup=10, iters=100, sample_conditions=False)
        rows.append({"precision": precision, "mode": mode, "weights_mb": round(weight_mb(m), 1),
                     "peak_allocated_gb": r.peak_allocated_gb, "peak_reserved_gb": r.peak_reserved_gb})
        print(f"  {precision} {mode:<16} 权重 {weight_mb(m):5.1f} MB | 峰值 allocated {r.peak_allocated_gb * 1024:6.1f} MB"
              f" / reserved {r.peak_reserved_gb * 1024:6.1f} MB")
        del fn, m, x
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    base = _base_fp32().cpu()
    base = base.cuda()
    x0 = torch.randn(1, 3, 224, 224, device="cuda")
    with torch.inference_mode():
        time_cuda(lambda: base(x0), warmup=30, iters=50, sample_conditions=False)   # 丢弃首测
    ramp_clocks()
    out = {"latency": [], "memory": []}
    for p in ("fp16", "fp32"):
        print(f"\n--- 延迟：{p}，配对交替 ---")
        out["latency"] += latency(p, base)
    for p in ("fp16", "fp32"):
        print(f"\n--- 显存（单独驻留，batch 1）：{p} ---")
        out["memory"] += memory(p, base)
    path = RESULTS / "compile_r2.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
