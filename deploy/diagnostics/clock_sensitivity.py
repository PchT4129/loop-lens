"""阶段 2 · 发现 4：显存降频会不会影响 ViT 的延迟。

同一配置反复测多轮，每轮记录(显存频率, 中位延迟)，按频率分桶比较。
对照组是一个明确访存受限的大张量复制——它必须对频率敏感，否则说明探针没有分辨力。

注意：显存频率是"自然"变化的，不受脚本控制。复跑时若整个过程都停在同一档频率，
就只会出现一个桶，无法比较——这不是失败，如实记录即可。
"""

import copy
import statistics

from deploy.diagnostics._common import save

import torch

from deploy.bench.timing import ramp_clocks, time_cuda
from deploy.export.wrapper import VPRDescriptor
from src.backbones import DINOv2FeatureExtractor


def main(rounds: int = 14) -> None:
    ext = DINOv2FeatureExtractor("dinov2_vits14", "mean").cuda().eval()
    m32 = VPRDescriptor(ext, 224).eval()
    m16 = copy.deepcopy(m32).to(torch.bfloat16)
    a = torch.empty(128 * 1024**2, device="cuda")
    b = torch.empty_like(a)
    x8 = torch.randn(8, 3, 224, 224, device="cuda")
    x1h = torch.randn(1, 3, 224, 224, device="cuda", dtype=torch.bfloat16)
    x8h = x8.to(torch.bfloat16)
    configs = {
        "copy 512MB（访存受限对照）": lambda: b.copy_(a),
        "ViT-S fp32 b8 224": lambda: m32(x8),
        "ViT-S bf16 b1 224": lambda: m16(x1h),
        "ViT-S bf16 b8 224": lambda: m16(x8h),
    }
    ramp_clocks(verbose=False)
    recs = {k: [] for k in configs}
    with torch.inference_mode():
        time_cuda(configs["ViT-S fp32 b8 224"], warmup=20, iters=50)       # 丢弃首测
        for _ in range(rounds):
            for name, fn in configs.items():
                r = time_cuda(fn, warmup=10, iters=80)
                recs[name].append((r.conditions.get("mem_clock_mhz_median"), r.median_ms))

    out = {}
    print(f"{'配置':<26}{'显存频率':>10}{'轮数':>6}{'中位延迟 ms':>13}{'相对最高档':>11}")
    for name, rs in recs.items():
        buckets: dict = {}
        for clk, ms in rs:
            buckets.setdefault(clk, []).append(ms)
        top = statistics.median(buckets[max(buckets)])
        out[name] = {}
        for clk in sorted(buckets, reverse=True):
            med = statistics.median(buckets[clk])
            out[name][str(int(clk))] = {"n": len(buckets[clk]), "median_ms": round(med, 4)}
            print(f"{name:<26}{clk:>10.0f}{len(buckets[clk]):>6}{med:>13.4f}{100 * (med / top - 1):>+10.1f}%")
    save("clock_sensitivity", {"rounds": rounds, "buckets": out})


if __name__ == "__main__":
    main()
