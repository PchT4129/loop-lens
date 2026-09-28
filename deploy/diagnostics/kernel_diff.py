"""阶段 3 · P2 的细节：batch 1 为什么比 batch 32 多发射 ~23 个 kernel。

同为 BF16、224px，只改 batch。统计每种 kernel（短名）每次前向出现的次数，列出不同的项。
"""

import collections

from deploy.diagnostics._common import save

import torch
from torch.profiler import ProfilerActivity, profile

from deploy.bench.bench_torch import ModelCache
from deploy.profile.profile_torch import _is_kernel, short_name


def main(n: int = 4) -> None:
    cache = ModelCache()
    counts = {}
    for b in (1, 32):
        m = cache.get("dinov2_vits14", 224, "bf16")
        x = torch.randn(b, 3, 224, 224, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            for _ in range(10):
                m(x)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for _ in range(n):
                    m(x)
                torch.cuda.synchronize()
        c = collections.Counter(short_name(e.name) for e in prof.events() if _is_kernel(e))
        counts[b] = {k: v / n for k, v in c.items()}
        del m, x
        torch.cuda.empty_cache()
    names = sorted(set(counts[1]) | set(counts[32]))
    diff = {k: {"b1": counts[1].get(k, 0), "b32": counts[32].get(k, 0)}
            for k in names if counts[1].get(k, 0) != counts[32].get(k, 0)}
    for k, v in diff.items():
        print(f"   {k:<40}{v['b1']:>6.0f}{v['b32']:>6.0f}")
    tot = {b: sum(counts[b].values()) for b in (1, 32)}
    print(f"   合计{'':<36}{tot[1]:>6.0f}{tot[32]:>6.0f}")
    save("kernel_diff", {"per_forward_counts": {str(k): v for k, v in counts.items()},
                         "differences": diff, "totals": {str(k): v for k, v in tot.items()}})


if __name__ == "__main__":
    main()
