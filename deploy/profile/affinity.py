"""阶段 3 · P6：小 batch 的抖动是否来自 CPU 调度（本机 CPU 为 8 大核 + 16 小核）。

CPU 亲和性（affinity）是什么
---------------------------
操作系统默认可以把一个线程放到任意 CPU 核上跑，并随时挪动它。"亲和性"就是限制
一个线程**只能**在哪些核上跑。`os.sched_setaffinity(0, {3})` = 只许在 3 号核上跑。

局限：WSL2 里的 24 个 CPU 是**虚拟 CPU**，它们到物理大核/小核的映射由 Windows 的
虚拟化层决定、且可能随时变化。所以这里能固定的只是"虚拟 CPU"，不是物理核。
详见 EXPERIMENTS.md P6 的可测性说明。

设计：bf16 b1@224（最不稳定的配置之一），对若干虚拟 CPU 各跑多轮，每轮 150 次迭代取中位数。
对照组是不绑核。看两件事：
  P6a 绑核后是否仍呈双峰（快簇 ~2.3 ms / 慢簇 >3 ms）
  P6b 不同虚拟 CPU 之间是否有系统性差异
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import warnings
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from deploy.bench.bench_torch import DTYPES, ModelCache           # noqa: E402
from deploy.bench.timing import ramp_clocks, time_cuda             # noqa: E402

warnings.filterwarnings("ignore")
RESULTS = REPO / "deploy" / "results"
SLOW_MS = 3.0          # 慢簇阈值：阶段 2 里快的那遍都在 2.0–2.3 ms


def main() -> None:
    rounds = 12
    cpus_to_try = [0, 3, 7, 8, 12, 16, 20, 23]
    all_cpus = set(range(os.cpu_count()))

    model = ModelCache().get("dinov2_vits14", 224, "bf16")
    x = torch.randn(1, 3, 224, 224, device="cuda", dtype=DTYPES["bf16"])
    with torch.inference_mode():
        time_cuda(lambda: model(x), warmup=30, iters=50, sample_conditions=False)   # 丢弃首测
    ramp_clocks()

    results = {}
    # 不绑核（对照）放在开头和结尾各一次，检查整个实验期间有没有漂移
    plan = [("不绑核-前", None)] + [(f"vCPU {c}", c) for c in cpus_to_try] + [("不绑核-后", None)]
    print(f"{'条件':<12}{'轮中位数 ms（排序后）':<62}{'慢簇占比':>8}")
    for label, cpu in plan:
        os.sched_setaffinity(0, all_cpus if cpu is None else {cpu})
        meds = []
        with torch.inference_mode():
            for _ in range(rounds):
                meds.append(time_cuda(lambda: model(x), warmup=20, iters=150,
                                      sample_conditions=False).median_ms)
        slow = sum(m > SLOW_MS for m in meds) / len(meds)
        results[label] = {"cpu": cpu, "round_medians_ms": [round(m, 3) for m in meds],
                          "median_ms": round(statistics.median(meds), 3),
                          "min_ms": round(min(meds), 3), "max_ms": round(max(meds), 3),
                          "slow_fraction": round(slow, 3)}
        print(f"{label:<12}{' '.join(f'{m:.2f}' for m in sorted(meds)):<62}{100 * slow:>7.0f}%")
    os.sched_setaffinity(0, all_cpus)

    path = RESULTS / "affinity_p6.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {path}")


if __name__ == "__main__":
    main()
