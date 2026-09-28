"""阶段 3 · P6 的后续：计时器的条件采样（后台每 150 ms 调一次 nvidia-smi）会不会干扰测量。

在 CPU 发射受限的配置里，一个周期性启动子进程、还要查询驱动的后台线程，
可能拖慢负责发射 kernel 的线程——即测量工具自身的观察者效应。
10 轮配对，轮内开/关顺序交替。
"""

import statistics

from deploy.diagnostics._common import save

import torch

from deploy.bench.bench_torch import DTYPES, ModelCache
from deploy.bench.timing import ramp_clocks, time_cuda


def main(rounds: int = 10) -> None:
    cache = ModelCache()
    out = {}
    for prec, batch in (("bf16", 1), ("fp16", 1), ("fp32", 8)):
        m = cache.get("dinov2_vits14", 224, prec)
        x = torch.randn(batch, 3, 224, 224, device="cuda", dtype=DTYPES[prec])
        on, off = [], []
        with torch.inference_mode():
            time_cuda(lambda: m(x), warmup=30, iters=50, sample_conditions=False)
            ramp_clocks(verbose=False)
            for i in range(rounds):
                order = [(True, on), (False, off)] if i % 2 == 0 else [(False, off), (True, on)]
                for flag, bucket in order:
                    bucket.append(time_cuda(lambda: m(x), warmup=20, iters=300,
                                            sample_conditions=flag).median_ms)
        so, sf = statistics.median(on), statistics.median(off)
        key = f"{prec} b{batch}"
        out[key] = {"sampler_on_ms": [round(v, 4) for v in on],
                    "sampler_off_ms": [round(v, 4) for v in off],
                    "effect_pct": round(100 * (so / sf - 1), 2)}
        print(f"{key}: 采样开 {so:.3f} / 关 {sf:.3f} ms → {100 * (so / sf - 1):+.1f}%")
        del m, x
        torch.cuda.empty_cache()
    save("sampler_effect", {"rounds": rounds, "results": out})


if __name__ == "__main__":
    main()
