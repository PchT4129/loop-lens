"""阶段 3（压缩版）：用 torch.profiler 验证阶段 2 写下的预测 P1–P5。

profiler 是什么
---------------
PyTorch 自带的剖析器。开启后，它会记录两条"时间线"：
  * CPU 侧：每个算子、每次调用 CUDA 接口（如 cudaLaunchKernel）的开始与结束时间
  * GPU 侧：每个 kernel 实际在 GPU 上执行的开始与结束时间
GPU 侧的时间来自 NVIDIA 的 CUPTI 接口——GPU 硬件自己打的时间戳，很准。

⚠️ 观察者效应
-------------
profiler 要给每个 CPU 操作做记录，**会拖慢 CPU 侧**。而我们要验证的恰恰是"CPU 发射受限"——
带着 profiler 测的墙钟会被放大，不能用。所以：
  * kernel 执行时间 → 取自 profiler（GPU 硬件时间戳，不受 CPU 被拖慢影响）
  * 墙钟延迟       → 取自**不开 profiler** 的独立测量（CUDA Event，300 次中位数）
两者之比 = GPU 忙碌占比。远小于 1 → GPU 大部分时间在等 CPU。

用法：
    python -m deploy.profile.profile_torch            # 全部配置 + P4 + 时间线
"""

from __future__ import annotations

import json
import statistics
import sys
import warnings
from pathlib import Path

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile, record_function

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from deploy.bench.bench_torch import DTYPES, ModelCache, set_tf32   # noqa: E402
from deploy.bench.timing import ramp_clocks, time_cuda               # noqa: E402
from src.backbones import DINOv2FeatureExtractor                     # noqa: E402

warnings.filterwarnings("ignore")
RESULTS = REPO / "deploy" / "results"

# 分类规则来自对本机真实 kernel 名的观察（见 EXPERIMENTS.md 阶段 3）
CATEGORIES = [
    ("注意力", lambda n: "fmha" in n or "flash" in n or "attention" in n.lower()),
    ("卷积(patch 切分)", lambda n: "fprop" in n or "convolve" in n or "nchwToNhwc" in n
                                   or "nhwcToNchw" in n or "precompute_indices" in n),
    ("矩阵乘 GEMM", lambda n: "gemm" in n.lower()),
    ("LayerNorm", lambda n: "layer_norm" in n),
    ("GELU", lambda n: "Gelu" in n),
    ("插值(pos_embed)", lambda n: "upsample" in n or "bicubic" in n),
    ("逐元素 加/乘", lambda n: "CUDAFunctor_add" in n or "BinaryFunctor" in n or "MulFunctor" in n),
    ("拷贝/拼接", lambda n: "Copy" in n or "copy" in n or "Cat" in n),
    ("归约(mean/norm)", lambda n: "reduce_kernel" in n),
]


def short_name(name: str) -> str:
    """'void at::native::(anonymous namespace)::upsample_bicubic2d_out_frame<float,...>(...)'
    → 'upsample_bicubic2d_out_frame'"""
    head = name.replace("void ", "").replace("(anonymous namespace)::", "").split("<")[0].split("(")[0]
    return head.split("::")[-1] or head


def categorize(name: str) -> str:
    for cat, rule in CATEGORIES:
        if rule(name):
            return cat
    return "其他"


def sdpa_backend(kernel_names: set[str]) -> str:
    if any("flash" in n for n in kernel_names):
        return "flash"
    if any("MemEffAttention" in n or "fmha_cutlass" in n for n in kernel_names):
        return "memory-efficient"
    if any("softmax" in n.lower() for n in kernel_names):
        return "math"
    return "未识别"


# record_function 打的标签会在 GPU 时间线上**镜像一份**，跨越整个被标注的区间。
# 第一版把它当成 kernel 计入，导致"GPU 忙碌 229%"这种物理上不可能的数——必须排除。
ANNOTATIONS = ("forward_", "interpolate_pos_encoding")


def _is_kernel(e) -> bool:
    return (e.device_type == DeviceType.CUDA
            and not e.name.startswith(ANNOTATIONS)
            and "Memcpy" not in e.name and "Memset" not in e.name)


def profile_config(model, x, n_active: int = 10) -> tuple[dict, dict]:
    """对 n_active 次前向做剖析，返回 (统计, 其中一次前向的时间线)。"""
    with torch.inference_mode():
        for _ in range(30):
            model(x)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for i in range(n_active):
                with record_function(f"forward_{i}"):
                    model(x)
            torch.cuda.synchronize()

    events = list(prof.events())
    kernels = [e for e in events if _is_kernel(e)]
    launches = [e for e in events if e.device_type == DeviceType.CPU
                and e.name in ("cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel",
                               "cuLaunchKernelEx")]
    fwd = [e for e in events if e.name.startswith("forward_") and e.device_type == DeviceType.CPU]
    # GPU 侧的 forward_i 标注 = 这次前向在 GPU 上从第一个 kernel 开始到最后一个结束的跨度
    fwd_gpu = [e for e in events if e.name.startswith("forward_") and e.device_type == DeviceType.CUDA]

    by_cat: dict[str, float] = {}
    for k in kernels:
        c = categorize(k.name)
        by_cat[c] = by_cat.get(c, 0.0) + k.time_range.elapsed_us()
    total_k = sum(by_cat.values())

    stats = {
        "kernels_per_forward": round(len(kernels) / n_active, 1),
        "kernel_time_ms_per_forward": round(total_k / n_active / 1000, 4),
        "launch_calls_per_forward": round(len(launches) / n_active, 1),
        "launch_api_us_mean": round(statistics.fmean(e.time_range.elapsed_us() for e in launches), 2)
        if launches else None,
        "cpu_forward_ms_under_profiler": round(
            statistics.median(e.time_range.elapsed_us() for e in fwd) / 1000, 4) if fwd else None,
        "category_share_pct": {c: round(100 * t / total_k, 1)
                               for c, t in sorted(by_cat.items(), key=lambda kv: -kv[1])},
        "sdpa_backend": sdpa_backend({k.name for k in kernels}),
        # 同一次运行内：kernel 时间 / GPU 跨度。与"无 profiler 墙钟"互补：
        #   发射受限时 profiler 会拉大 kernel 间空档 → 这个比值偏低（低估忙碌）
        #   GPU 受限且顶功耗墙时，两次运行频率不同 → 跨运行的比值不可靠，此值更可信
        "gpu_span_ms_same_run": round(statistics.median(
            e.time_range.elapsed_us() for e in fwd_gpu) / 1000, 4) if fwd_gpu else None,
    }

    # 取中间一次前向的时间线：CPU 发射调用 + GPU kernel。
    # CPU 侧按 CPU 标注的跨度取；GPU 侧**按 GPU 标注的跨度取**——第一版用"到下次 CPU 前向开始"
    # 截 GPU kernel，GPU 受限时 GPU 落后于 CPU，后半段 kernel 被截掉了（b8 只剩 136/156 个）。
    idx = n_active // 2
    cpu_mid = next((e for e in fwd if e.name == f"forward_{idx}"), None)
    gpu_mid = next((e for e in fwd_gpu if e.name == f"forward_{idx}"), None)
    timeline = {}
    if cpu_mid is not None and gpu_mid is not None:
        t0 = cpu_mid.time_range.start
        c0, c1 = cpu_mid.time_range.start, cpu_mid.time_range.end
        g0, g1 = gpu_mid.time_range.start, gpu_mid.time_range.end
        cpu_l = [(e.time_range.start - t0, e.time_range.end - t0)
                 for e in launches if c0 <= e.time_range.start <= c1]
        gpu_l = [(k.time_range.start - t0, k.time_range.end - t0, categorize(k.name))
                 for k in kernels if g0 <= k.time_range.start < g1]
        timeline = {"cpu_launches_us": cpu_l, "gpu_kernels_us": gpu_l,
                    "cpu_span_us": c1 - c0, "gpu_span_us": g1 - g0,
                    "gpu_start_offset_us": g0 - t0}
    return stats, timeline


def wall_latency(model, x, rounds: int = 3) -> dict:
    """不开 profiler 的墙钟，多轮取中位数（小 batch 抖动大，见发现 5）。"""
    meds = []
    with torch.inference_mode():
        for _ in range(rounds):
            meds.append(time_cuda(lambda: model(x), warmup=30, iters=200,
                                  sample_conditions=False).median_ms)
    return {"wall_ms_rounds": [round(m, 4) for m in meds],
            "wall_ms_min": round(min(meds), 4), "wall_ms_median": round(statistics.median(meds), 4)}


def p4_posembed(precision: str) -> dict:
    """P4：原始提取器里 interpolate_pos_encoding 这一段发射了几个 kernel、CPU 花了多久。

    做法：给原模型的 interpolate_pos_encoding 套一层 record_function 标签，
    剖析后找到这个标签下（含子调用）所有的 kernel。
    """
    ext = DINOv2FeatureExtractor("dinov2_vits14", "mean").cuda().eval().to(DTYPES[precision])
    vit = ext.model
    original = vit.interpolate_pos_encoding

    def labelled(*a, **k):
        with record_function("interpolate_pos_encoding"):
            return original(*a, **k)

    vit.interpolate_pos_encoding = labelled
    x = torch.randn(1, 3, 224, 224, device="cuda", dtype=DTYPES[precision])
    with torch.inference_mode():
        for _ in range(30):
            ext(x)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(10):
                ext(x)
            torch.cuda.synchronize()

    events = list(prof.events())
    cpu_regions = [e for e in events if e.name == "interpolate_pos_encoding"
                   and e.device_type == DeviceType.CPU]
    gpu_regions = [e for e in events if e.name == "interpolate_pos_encoding"
                   and e.device_type == DeviceType.CUDA]
    kernels = [e for e in events if _is_kernel(e)]

    # GPU 侧标注的时间窗 = 该区域在 GPU 上从第一个 kernel 开始到最后一个结束。
    # 落在窗内的真实 kernel 就是这段代码发射的。
    per_count, per_busy, per_span, names = [], [], [], set()
    for g in gpu_regions:
        inside = [k for k in kernels
                  if g.time_range.start <= k.time_range.start < g.time_range.end]
        per_count.append(len(inside))
        per_busy.append(sum(k.time_range.elapsed_us() for k in inside))
        per_span.append(g.time_range.elapsed_us())
        names |= {short_name(k.name) for k in inside}

    def subtree_ops(ev):
        out = [ev.name]
        for ch in getattr(ev, "cpu_children", []):
            out += subtree_ops(ch)
        return out

    ops = subtree_ops(cpu_regions[0]) if cpu_regions else []
    vit.interpolate_pos_encoding = original
    return {
        "precision": precision,
        "kernels_in_region": round(statistics.fmean(per_count), 1) if per_count else None,
        "kernel_names": sorted(names),
        "kernel_gpu_busy_us": round(statistics.median(per_busy), 2) if per_busy else None,
        "gpu_span_us": round(statistics.median(per_span), 1) if per_span else None,
        "cpu_us_in_region_under_profiler": round(statistics.median(
            r.time_range.elapsed_us() for r in cpu_regions), 1) if cpu_regions else None,
        "aten_ops_in_region": [o for o in ops if o.startswith("aten::")],
    }


def main() -> None:
    cache = ModelCache()
    configs = [("fp32", 1, 224), ("fp16", 1, 224), ("bf16", 1, 224),
               ("fp32", 8, 224), ("bf16", 8, 224), ("bf16", 32, 224),
               ("fp32", 8, 448), ("bf16", 8, 448)]
    out: dict = {"configs": [], "timelines": {}, "p4": []}

    # 丢弃首测
    m0 = cache.get("dinov2_vits14", 224, "fp32")
    x0 = torch.randn(1, 3, 224, 224, device="cuda")
    with torch.inference_mode():
        time_cuda(lambda: m0(x0), warmup=30, iters=50, sample_conditions=False)
    del m0, x0
    ramp_clocks()

    print(f"\n{'配置':<13}{'kernel数':>8}{'kernel时间':>11}{'无profiler墙钟':>14}{'同次GPU跨度':>12}"
          f"{'忙碌①':>7}{'忙碌②':>7}  SDPA")
    for prec, b, res in configs:
        set_tf32(False)
        model = cache.get("dinov2_vits14", res, prec)
        x = torch.randn(b, 3, res, res, device="cuda", dtype=DTYPES[prec])
        wall = wall_latency(model, x)
        stats, tl = profile_config(model, x)
        busy = stats["kernel_time_ms_per_forward"] / wall["wall_ms_median"]
        busy_same = (stats["kernel_time_ms_per_forward"] / stats["gpu_span_ms_same_run"]
                     if stats["gpu_span_ms_same_run"] else float("nan"))
        row = {"precision": prec, "batch": b, "res": res, **wall, **stats,
               "gpu_busy_vs_unprofiled_wall": round(busy, 3),
               "gpu_busy_same_run": round(busy_same, 3)}
        out["configs"].append(row)
        out["timelines"][f"{prec}_b{b}_{res}"] = tl
        print(f"{prec} b{b} {res:<6}{stats['kernels_per_forward']:>8.0f}"
              f"{stats['kernel_time_ms_per_forward']:>9.3f}ms"
              f"{wall['wall_ms_median']:>12.3f}ms{stats['gpu_span_ms_same_run']:>10.3f}ms"
              f"{100 * busy:>6.0f}%{100 * busy_same:>6.0f}%  {stats['sdpa_backend']}")
        del model, x
        torch.cuda.empty_cache()

    print("  忙碌① = kernel 时间 / 无 profiler 墙钟（发射受限时准）"
          "   忙碌② = kernel 时间 / 同次 GPU 跨度（GPU 受限时准）")
    print("\n--- 各类 kernel 占 GPU 时间 ---")
    for r in out["configs"]:
        top = ", ".join(f"{c} {p}%" for c, p in list(r["category_share_pct"].items())[:6])
        print(f"  {r['precision']} b{r['batch']} {r['res']}: {top}")

    print("\n--- P4：原始实现里 pos_embed 插值这一段 ---")
    for prec in ("fp32", "bf16"):
        r = p4_posembed(prec)
        out["p4"].append(r)
        print(f"  {prec}: {r['kernels_in_region']} 个 kernel {r['kernel_names']}")
        print(f"        kernel 实际执行 {r['kernel_gpu_busy_us']} µs | GPU 上区间跨度 {r['gpu_span_us']} µs"
              f" | CPU 耗时 {r['cpu_us_in_region_under_profiler']} µs（profiler 下，偏大）")
        print(f"        算子序列: {' → '.join(o.replace('aten::', '') for o in r['aten_ops_in_region'])}")

    path = RESULTS / "profile_torch.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {path}")


if __name__ == "__main__":
    main()
