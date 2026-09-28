"""阶段 3 · P6 的后续：batch 1 延迟是否取决于"前一段时间 GPU 干了什么"。

背景：P6 绑核实验里，连不绑核的对照组都不抖（12 轮全在 2.0–2.2 ms），阶段 2 的双峰没有复现。
两次实验的差别：阶段 2 里轻配置与顶着功耗墙的重配置交替运行；P6 只反复跑一个轻配置。

假设（平台功耗共享）：笔记本的 CPU 与 GPU 共享整机功耗预算。GPU 重负载时平台把功耗
挪给 GPU，CPU 降速；batch 1 恰好是 CPU 发射受限的，于是紧随重负载之后的测量变慢。

设计：主动操纵"前置负载"这一个变量，交错重复：
  A 前置重负载：fp32 b32@448 连续跑 3 秒（顶功耗墙），紧接着测轻配置
  B 前置空闲：  GPU 空闲 3 秒，再测
  C 前置轻负载：轻配置本身连续跑 3 秒，再测
轻配置 = bf16 b1@224。每种条件 8 次，按 ABC ABC ... 交错，抵消缓慢漂移。
"""

from __future__ import annotations

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

from deploy.bench.bench_torch import DTYPES, ModelCache           # noqa: E402
from deploy.bench.timing import _smi, ramp_clocks, time_cuda       # noqa: E402

warnings.filterwarnings("ignore")
RESULTS = REPO / "deploy" / "results"


def busy_for(fn, seconds: float) -> None:
    end = time.time() + seconds
    while time.time() < end:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()


def main() -> None:
    cache = ModelCache()
    light = cache.get("dinov2_vits14", 224, "bf16")
    xl = torch.randn(1, 3, 224, 224, device="cuda", dtype=DTYPES["bf16"])
    heavy = cache.get("dinov2_vits14", 448, "fp32")
    xh = torch.randn(32, 3, 448, 448, device="cuda", dtype=DTYPES["fp32"])

    run_light = lambda: light(xl)          # noqa: E731
    run_heavy = lambda: heavy(xh)          # noqa: E731

    with torch.inference_mode():
        time_cuda(run_light, warmup=30, iters=50, sample_conditions=False)    # 丢弃首测
        ramp_clocks()
        conds = {"A 前置重负载": lambda: busy_for(run_heavy, 3.0),
                 "B 前置空闲":   lambda: (torch.cuda.synchronize(), time.sleep(3.0)),
                 "C 前置轻负载": lambda: busy_for(run_light, 3.0)}
        res = {k: {"median_ms": [], "power_w_before": []} for k in conds}
        for rep in range(8):
            for name, pre in conds.items():
                pre()
                try:
                    pw = float(_smi("power.draw")[0])
                except Exception:                                  # noqa: BLE001
                    pw = float("nan")
                r = time_cuda(run_light, warmup=5, iters=150, sample_conditions=False)
                res[name]["median_ms"].append(round(r.median_ms, 3))
                res[name]["power_w_before"].append(pw)

    print(f"{'条件':<12}{'8 次中位数 ms（排序后）':<52}{'中位':>7}{'测前功耗':>9}")
    for name, d in res.items():
        v = d["median_ms"]
        print(f"{name:<12}{' '.join(f'{m:.2f}' for m in sorted(v)):<52}"
              f"{statistics.median(v):>7.3f}{statistics.median(d['power_w_before']):>8.0f}W")
    a, b = statistics.median(res["A 前置重负载"]["median_ms"]), statistics.median(res["B 前置空闲"]["median_ms"])
    c = statistics.median(res["C 前置轻负载"]["median_ms"])
    print(f"\n重负载之后 vs 空闲之后：{100 * (a / b - 1):+.1f}%    重负载之后 vs 轻负载之后：{100 * (a / c - 1):+.1f}%")

    path = RESULTS / "preceding_load_p6.json"
    path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已写入 {path}")


if __name__ == "__main__":
    main()
