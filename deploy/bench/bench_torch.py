"""阶段 2：PyTorch 基线扫描——原样的模型有多快、占多少显存。

测什么、不测什么
----------------
只测 **GPU 上的模型前向**：输入是已经在显存里的张量。不含 JPEG 解码、缩放、
归一化、主机到设备拷贝。这些在真实系统里也要花时间，但它们不是本项目要优化的
对象，混进来会稀释模型本身的差异。输入用随机张量——ViT 没有数据相关的控制流，
数值大小不影响耗时。

精度模式（"改一个变量"）
------------------------
  fp32  权重与激活 float32，TF32 全关（matmul + cuDNN）
  tf32  同上，只把两个 TF32 开关打开 —— 与 fp32 只差这一个变量
  fp16  model.to(float16)，权重与激活全部半精度
  bf16  model.to(bfloat16)

注意这里用的是**整体转换**（.to(dtype)），不是 autocast。区别见 side_experiments：
autocast 保留 FP32 权重、在算子层面临时转换，所以**权重显存不会减半**——对于
"显存预算"这个动机，整体转换才是部署意义上的"跑在 FP16"。

纪律（来自阶段 0/1 的发现）
--------------------------
* 每一遍开始前把 GPU 拉到满频（发现 2）
* 每个会话第一个测量点作废（阶段 1：首测偏低 35%）
* 双遍 A/B/A：第二遍倒序，两遍差 >5% 的配置标记为不稳定（发现 3）
* 每个点记录显存频率与峰值显存；超可用显存的点标记不可信（发现 1）

用法：
    python -m deploy.bench.bench_torch --sweep           # 主扫描（ViT-S 72 点 + ViT-B 6 点，双遍）
    python -m deploy.bench.bench_torch --side            # 两个单变量对照实验
    python -m deploy.bench.bench_torch --sweep --quick   # 冒烟：每维只取两档
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import sys
import warnings
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from deploy.bench.timing import ramp_clocks, time_cuda          # noqa: E402
from deploy.export.wrapper import VPRDescriptor                 # noqa: E402
from src.backbones import DINOv2FeatureExtractor                # noqa: E402

warnings.filterwarnings("ignore", message=".*xFormers.*")
RESULTS = REPO / "deploy" / "results"

DTYPES = {"fp32": torch.float32, "tf32": torch.float32,
          "fp16": torch.float16, "bf16": torch.bfloat16}


def set_tf32(enabled: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


class ModelCache:
    """每个 (variant, 分辨率) 只从 hub 加载一次，基座放在 CPU 上。

    为什么放 CPU：若基座常驻显存，它会被算进每个测量点的峰值显存里，
    让所有数字都虚高一个模型的大小。每次测量时 deepcopy 一份到 GPU 并转精度。
    """

    def __init__(self) -> None:
        self._base: dict[tuple[str, int], VPRDescriptor] = {}

    def get(self, variant: str, image_size: int, precision: str) -> VPRDescriptor:
        key = (variant, image_size)
        if key not in self._base:
            ext = DINOv2FeatureExtractor(variant=variant, aggregation="mean").cuda().eval()
            # pos_embed 在 GPU 上插值（与等价性闸门的条件一致），再整体移到 CPU 存放
            self._base[key] = VPRDescriptor(ext, image_size).eval().cpu()
            del ext
            torch.cuda.empty_cache()
        model = copy.deepcopy(self._base[key]).to(device="cuda", dtype=DTYPES[precision])
        return model.eval()


def weight_mb(model: torch.nn.Module) -> float:
    n_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    n_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return n_bytes / 1024**2


def measure(cache: ModelCache, variant: str, precision: str, batch: int, res: int,
            pass_idx: int, warmup: int, iters: int) -> dict:
    row = {"variant": variant, "precision": precision, "batch": batch, "res": res,
           "tokens": (res // 14) ** 2 + 1, "pass": pass_idx}
    set_tf32(precision == "tf32")
    model = x = None
    try:
        model = cache.get(variant, res, precision)
        x = torch.randn(batch, 3, res, res, device="cuda", dtype=DTYPES[precision])
        row["weights_mb"] = round(weight_mb(model), 1)

        with torch.inference_mode():
            r = time_cuda(lambda: model(x), label=f"{variant} {precision} b{batch} {res}",
                          warmup=warmup, iters=iters)
        row.update(r.as_dict())
        row["throughput_img_s"] = round(batch / (r.median_ms / 1e3), 1)
        row["status"] = "ok"
        cond = r.conditions
        flags = ("" if r.mem_clock_ok else " ⚠️降频") + ("" if r.vram_headroom_ok else " ⚠️溢出")
        print(f"  {variant[-6:]:>6} {precision:>4} b{batch:<3} {res}px  "
              f"中位 {r.median_ms:8.3f} ms  p99 {r.p99_ms:8.3f}  "
              f"吞吐 {row['throughput_img_s']:8.1f} img/s  "
              f"显存 {r.peak_allocated_gb:5.2f}/{r.peak_reserved_gb:5.2f} GB  "
              f"显存频率 {cond.get('mem_clock_mhz_median', float('nan')):.0f}{flags}")
    except torch.OutOfMemoryError:
        row["status"] = "oom"
        print(f"  {variant[-6:]:>6} {precision:>4} b{batch:<3} {res}px  OOM")
    finally:
        del model, x
        torch.cuda.empty_cache()
        set_tf32(False)
    return row


def build_grid(quick: bool) -> list[tuple[str, str, int, int]]:
    precisions = ["fp32", "tf32", "fp16", "bf16"]
    batches = [1, 8] if quick else [1, 2, 4, 8, 16, 32]
    resolutions = [224, 448] if quick else [224, 336, 448]
    grid = [("dinov2_vits14", p, b, r) for r in resolutions for b in batches for p in precisions]
    if not quick:
        # ViT-B 只作参照点（深度优先）。阶段 1 发现 BF16 比 FP16 快，所以两者都测
        grid += [("dinov2_vitb14", p, b, 224) for b in (1, 8) for p in ("fp32", "fp16", "bf16")]
    return grid


def sweep(quick: bool, passes: int, warmup: int, iters: int) -> list[dict]:
    cache = ModelCache()
    grid = build_grid(quick)
    rows: list[dict] = []

    print("\n--- 丢弃首测（会话里第一个测量点不可靠）---")
    measure(cache, "dinov2_vits14", "fp32", 1, 224, -1, warmup, 50)

    for pass_idx in range(passes):
        order = grid if pass_idx % 2 == 0 else list(reversed(grid))
        print(f"\n--- 第 {pass_idx + 1} 遍（{'正序' if pass_idx % 2 == 0 else '倒序'}，"
              f"{len(order)} 点）---")
        ramp_clocks()
        for variant, precision, batch, res in order:
            rows.append(measure(cache, variant, precision, batch, res, pass_idx, warmup, iters))
    return rows


def combine(rows: list[dict]) -> list[dict]:
    """把两遍合并成一行：取两遍中位数的均值，并报告两遍差异。"""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["variant"], r["precision"], r["batch"], r["res"]), []).append(r)

    out = []
    for (variant, precision, batch, res), rs in groups.items():
        ok = [r for r in rs if r["status"] == "ok"]
        base = {"variant": variant, "precision": precision, "batch": batch, "res": res,
                "tokens": rs[0]["tokens"]}
        if not ok:
            out.append({**base, "status": "oom"})
            continue
        meds = [r["median_ms"] for r in ok]
        spread = 100 * (max(meds) - min(meds)) / max(meds) if len(meds) > 1 else 0.0
        median_ms = statistics.fmean(meds)
        out.append({
            **base,
            "status": "ok",
            "median_ms": round(median_ms, 4),
            "p90_ms": round(statistics.fmean(r["p90_ms"] for r in ok), 4),
            "p99_ms": round(statistics.fmean(r["p99_ms"] for r in ok), 4),
            "pass_medians_ms": [round(m, 4) for m in meds],
            "pass_spread_pct": round(spread, 1),
            "throughput_img_s": round(batch / (median_ms / 1e3), 1),
            "peak_allocated_gb": max(r["peak_allocated_gb"] for r in ok),
            "peak_reserved_gb": max(r["peak_reserved_gb"] for r in ok),
            "weights_mb": ok[0]["weights_mb"],
            "all_mem_clock_ok": all(r["mem_clock_ok"] for r in ok),
            "all_vram_headroom_ok": all(r["vram_headroom_ok"] for r in ok),
            "stable": spread <= 5.0,
            "power_w_median": statistics.fmean(
                r["conditions"].get("power_w_median", float("nan")) for r in ok),
            "sm_clock_mhz_median": statistics.fmean(
                r["conditions"].get("sm_clock_mhz_median", float("nan")) for r in ok),
        })
    return out


# ------------------------------------------------------------ 两个单变量对照
def _paired_rounds(fn_a, fn_b, rounds: int, warmup: int, iters: int) -> dict:
    """多轮交替测 A、B，按轮配对求差。

    为什么不用单次 A/B/A：第一版就是 A/B/A，结果出现"删掉一次插值反而慢 8.7%"
    这种机制上说不通的数——单次 A/B/A 对"恰好发生在 B 测量期间的短暂降速"没有
    抵抗力。配对设计里每一轮的 A 和 B 背靠背测，共享同一段机器状态，差值对缓慢
    漂移免疫；再取多轮差值的中位数，对个别异常轮免疫。
    轮内的先后顺序也交替（奇数轮先 B），抵消"谁先测谁吃亏"的顺序效应。
    """
    a_ms, b_ms, diffs = [], [], []
    for i in range(rounds):
        order = [("a", fn_a), ("b", fn_b)] if i % 2 == 0 else [("b", fn_b), ("a", fn_a)]
        got = {}
        for tag, fn in order:
            with torch.inference_mode():
                got[tag] = time_cuda(fn, warmup=warmup, iters=iters,
                                     sample_conditions=False).median_ms
        a_ms.append(got["a"]); b_ms.append(got["b"]); diffs.append(got["a"] - got["b"])
    diffs_sorted = sorted(diffs)
    q1 = diffs_sorted[len(diffs) // 4]
    q3 = diffs_sorted[(3 * len(diffs)) // 4]
    return {
        "a_median_ms": round(statistics.median(a_ms), 4),
        "b_median_ms": round(statistics.median(b_ms), 4),
        "paired_diff_median_ms": round(statistics.median(diffs), 4),
        "paired_diff_iqr_ms": [round(q1, 4), round(q3, 4)],
        "rounds_b_faster": sum(d > 0 for d in diffs),
        "rounds": rounds,
    }


def _isolated_peak(build, x_shape, dtype, warmup: int, iters: int) -> tuple[float, float]:
    """显存要**单独驻留**时测。第一版让两个模型同时在显存里，峰值成了二者之和，比较无效。"""
    torch.cuda.empty_cache()
    model = build()
    x = torch.randn(*x_shape, device="cuda", dtype=dtype)
    with torch.inference_mode():
        r = time_cuda(lambda: model(x), warmup=warmup, iters=iters, sample_conditions=False)
    w = weight_mb(model)
    del model, x
    torch.cuda.empty_cache()
    return r.peak_allocated_gb, w


def side_experiments(warmup: int, iters: int, rounds: int = 10) -> list[dict]:
    """对照 1：原始提取器 vs 包装（唯一差异 = pos_embed 是否每次插值）
    对照 2：autocast vs 整体转换（唯一差异 = 权重是否仍为 FP32）
    """
    rows: list[dict] = []
    set_tf32(False)
    print("\n--- 丢弃首测 ---")
    ext0 = DINOv2FeatureExtractor("dinov2_vits14", "mean").cuda().eval()
    x0 = torch.randn(1, 3, 224, 224, device="cuda")
    with torch.inference_mode():
        time_cuda(lambda: ext0(x0), warmup=warmup, iters=50, sample_conditions=False)
    del ext0, x0
    ramp_clocks()

    print(f"\n--- 对照 1：pos_embed 每次插值（原始 A） vs 固化常量（包装 B），{rounds} 轮配对 ---")
    for precision in ("fp32", "bf16"):
        dtype = DTYPES[precision]
        orig = DINOv2FeatureExtractor("dinov2_vits14", "mean").cuda().eval()
        wrap = VPRDescriptor(orig, 224).eval()           # 共享权重，只有 pos_embed 路径不同
        orig, wrap = orig.to(dtype), wrap.to(dtype)
        for batch in (1, 8):
            x = torch.randn(batch, 3, 224, 224, device="cuda", dtype=dtype)
            res = _paired_rounds(lambda: orig(x), lambda: wrap(x), rounds, warmup, iters)
            saved = res["paired_diff_median_ms"]
            rows.append({"experiment": "posembed_bake", "precision": precision, "batch": batch,
                         **res, "saved_pct": round(100 * saved / res["a_median_ms"], 2)})
            print(f"  {precision} b{batch}: 原始 {res['a_median_ms']:.3f} → 包装 {res['b_median_ms']:.3f} ms | "
                  f"配对差中位 {saved:+.3f} ms ({100 * saved / res['a_median_ms']:+.1f}%) "
                  f"IQR [{res['paired_diff_iqr_ms'][0]:+.3f}, {res['paired_diff_iqr_ms'][1]:+.3f}] | "
                  f"包装更快的轮数 {res['rounds_b_faster']}/{rounds}")
        del orig, wrap
        torch.cuda.empty_cache()

    print(f"\n--- 对照 2：autocast(bf16, FP32 权重) A vs 整体转换 .to(bf16) B ---")
    base = DINOv2FeatureExtractor("dinov2_vits14", "mean").cuda().eval()
    fp32_wrap = VPRDescriptor(base, 224).eval()
    bf16_wrap = copy.deepcopy(fp32_wrap).to(torch.bfloat16)

    def run_autocast(m, x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return m(x)

    for batch in (1, 8):
        x32 = torch.randn(batch, 3, 224, 224, device="cuda")
        x16 = x32.to(torch.bfloat16)
        res = _paired_rounds(lambda: run_autocast(fp32_wrap, x32), lambda: bf16_wrap(x16),
                             rounds, warmup, iters)
        rows.append({"experiment": "autocast_vs_cast", "batch": batch, **res})
        print(f"  b{batch} 延迟: autocast {res['a_median_ms']:.3f} → 整体转换 {res['b_median_ms']:.3f} ms | "
              f"配对差中位 {res['paired_diff_median_ms']:+.3f} ms | "
              f"整体转换更快的轮数 {res['rounds_b_faster']}/{rounds}")

    # 显存：两个模型不能同时驻留
    fp32_cpu = fp32_wrap.cpu()
    del bf16_wrap, fp32_wrap, base
    torch.cuda.empty_cache()
    for batch in (1, 8):
        pk_a, w_a = _isolated_peak(
            lambda: _AutocastWrapper(copy.deepcopy(fp32_cpu).cuda()), (batch, 3, 224, 224),
            torch.float32, warmup, 100)
        pk_b, w_b = _isolated_peak(
            lambda: copy.deepcopy(fp32_cpu).to(device="cuda", dtype=torch.bfloat16),
            (batch, 3, 224, 224), torch.bfloat16, warmup, 100)
        rows.append({"experiment": "autocast_vs_cast_memory", "batch": batch,
                     "autocast_weights_mb": round(w_a, 1), "autocast_peak_alloc_gb": pk_a,
                     "cast_weights_mb": round(w_b, 1), "cast_peak_alloc_gb": pk_b})
        print(f"  b{batch} 显存（单独驻留）: autocast 权重 {w_a:.0f} MB / 峰值 {pk_a:.3f} GB   |   "
              f"整体转换 权重 {w_b:.0f} MB / 峰值 {pk_b:.3f} GB")
    return rows


class _AutocastWrapper(torch.nn.Module):
    def __init__(self, m: torch.nn.Module):
        super().__init__()
        self.m = m

    def forward(self, x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.m(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--side", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=300)
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        rows = sweep(args.quick, args.passes, args.warmup, args.iters)
        combined = combine(rows)
        tag = "_quick" if args.quick else ""
        (RESULTS / f"baseline_torch{tag}_raw.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        (RESULTS / f"baseline_torch{tag}.json").write_text(
            json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
        fields = ["variant", "precision", "batch", "res", "tokens", "status", "median_ms",
                  "p90_ms", "p99_ms", "throughput_img_s", "peak_allocated_gb",
                  "peak_reserved_gb", "weights_mb", "pass_spread_pct", "stable",
                  "all_mem_clock_ok", "all_vram_headroom_ok", "power_w_median",
                  "sm_clock_mhz_median"]
        with open(RESULTS / f"baseline_torch{tag}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(combined)
        unstable = [c for c in combined if c["status"] == "ok" and not c["stable"]]
        low_clock = [c for c in combined if c["status"] == "ok" and not c["all_mem_clock_ok"]]
        # 显存降频只作记录、不作有效性判据：频率敏感性探针实测 ViT-S@224 在
        # 14001→9001 MHz 下延迟变化 ≤3.7%（访存受限对照组 +52%）。
        # 336/448 未直接探针，靠双遍差异间接把关。见 EXPERIMENTS.md 发现 4。
        print(f"\n汇总：{len(combined)} 个配置 | 两遍差 >5%（不稳定）{len(unstable)} 个 | "
              f"OOM {sum(c['status'] == 'oom' for c in combined)} 个 | "
              f"含显存降频样本 {len(low_clock)} 个（仅记录，见发现 4）")
        print(f"已写入 {RESULTS / f'baseline_torch{tag}.csv'}")

    if args.side:
        rows = side_experiments(args.warmup, args.iters)
        (RESULTS / "baseline_side_experiments.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"已写入 {RESULTS / 'baseline_side_experiments.json'}")


if __name__ == "__main__":
    main()
