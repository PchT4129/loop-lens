"""阶段 1：量出这台机器的物理上限，作为 roofline 的两条渐近线。

为什么必须先做这件事
--------------------
Roofline 是一张"上界图"：横轴是**算术强度**（每搬 1 字节数据能做多少次浮点运算，
FLOP/byte），纵轴是实际达到的算力。一个算子能跑多快，受两条线夹住：

    可达算力      ——  水平的"屋顶"，再多算术强度也突破不了
    可达带宽 × AI ——  斜的"屋檐"，搬数据速度决定的上界

两条线的交点叫**脊点**（ridge point）。脊点左边是**访存受限**（memory-bound，
瓶颈在搬数据），右边是**算力受限**（compute-bound，瓶颈在算）。

关键是这两条线必须用**实测可达值**，不能用规格书。本机标称带宽 672 GB/s，
实测只有七成左右。若拿 672 当屋檐，每个算子都会显得"效率很低"，诊断全错。

两个微基准
----------
1. 带宽：故意选算术强度极低的操作（复制、triad），让瓶颈只可能是搬数据
2. 算力：大方阵矩阵乘，算术强度极高，瓶颈只可能是算力

用法：
    python -m deploy.bench.machine_limits              # 全部
    python -m deploy.bench.machine_limits --bandwidth  # 只测带宽
    python -m deploy.bench.machine_limits --compute    # 只测算力
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from deploy.bench.timing import ramp_clocks, time_cuda

RESULTS = Path(__file__).resolve().parents[1] / "results"


# ----------------------------------------------------------------- 带宽
def bandwidth_sweep(
    sizes_mb: tuple[int, ...] = (64, 128, 256, 512, 1024, 1536),
    passes: int = 2,
) -> list[dict]:
    """扫不同工作集大小，找带宽平台期。

    为什么要扫而不是测一个尺寸：小工作集会**部分命中 L2 缓存**，测出的"带宽"
    虚高（那是缓存带宽不是显存带宽）。必须放大到明显超过缓存容量，读数才收敛
    到真实的显存带宽——收敛后的那个平台值才是 roofline 的屋檐。

    两种访存模式（字节数算法不同，要分别报）：
      copy   y = x          读 N + 写 N        = 2N
      triad  y = a + 3.0*b  读 2N + 写 N       = 3N   （经典 STREAM triad）

    ⭐ 为什么要跑两遍、第二遍倒序（A/B/A 纪律）
    ------------------------------------------
    第一版只按尺寸从小到大扫了一遍，结果"越大越慢"。但这个结论有致命的混杂：
    尺寸是递增的，时间也是递增的，所以"大尺寸慢"和"后面测的慢（降频）"
    两种解释**完全重合**，无法区分。

    倒序再扫一遍就能判别：
      * 若大尺寸在倒序里依然慢 → 真的是尺寸效应
      * 若大尺寸在倒序里变快了 → 是时间漂移（降频/热），第一遍的结论是假的

    同时打开频率采样，让降频能被直接看到而不是靠推断。
    """
    rows: list[dict] = []
    for pass_idx in range(passes):
        order = tuple(sizes_mb) if pass_idx % 2 == 0 else tuple(reversed(sizes_mb))
        direction = "升序" if pass_idx % 2 == 0 else "倒序"
        print(f"\n--- 带宽扫描 第{pass_idx + 1}遍（{direction}）---")
        print(f"{'工作集':>10} {'模式':>7} {'搬运量':>10} {'中位耗时':>10} {'带宽':>12} "
              f"{'SM':>7} {'显存频率':>9} {'功耗':>8}")
        rows += _bandwidth_pass(order, pass_idx, direction)
    return rows


def _bandwidth_pass(sizes_mb: tuple[int, ...], pass_idx: int, direction: str) -> list[dict]:
    rows: list[dict] = []
    for size_mb in sizes_mb:
        n = size_mb * 1024**2 // 4                    # float32，每元素 4 字节
        try:
            x = torch.randn(n, device="cuda")
            y = torch.empty_like(x)
            b = torch.randn(n, device="cuda")
        except torch.OutOfMemoryError:
            print(f"{size_mb:>8} MB  显存不足，跳过")
            continue

        nbytes = n * 4
        for mode, fn, moved in (
            ("copy", lambda: y.copy_(x), 2 * nbytes),
            ("triad", lambda: torch.add(x, b, alpha=3.0, out=y), 3 * nbytes),
        ):
            r = time_cuda(fn, label=f"{mode} {size_mb}MB", warmup=20, iters=100,
                          sample_conditions=True)
            gbs = moved / (r.median_ms / 1e3) / 1e9
            cond = r.conditions
            rows.append({
                "kind": "bandwidth", "mode": mode, "working_set_mb": size_mb,
                "bytes_moved": moved, "gb_per_s": round(gbs, 1),
                "pass": pass_idx, "direction": direction,
                **r.as_dict(),
            })
            print(f"{size_mb:>8} MB {mode:>7} {moved / 1024**2:>8.0f} MiB "
                  f"{r.median_ms:>9.3f} ms {gbs:>10.1f} GB/s "
                  f"{cond.get('sm_clock_mhz_median', float('nan')):>7.0f} "
                  f"{cond.get('mem_clock_mhz_median', float('nan')):>9.0f} "
                  f"{cond.get('power_w_median', float('nan')):>7.0f}W")

        del x, y, b
        torch.cuda.empty_cache()

    return rows


# ----------------------------------------------------------------- 算力
def compute_sweep(sizes: tuple[int, ...] = (4096, 8192, 12288), passes: int = 2) -> list[dict]:
    """大方阵矩阵乘，分精度测可达算力。

    FLOP 怎么数：M×K 乘 K×N 的矩阵乘，输出 M×N 个元素，每个要做 K 次乘和 K 次加，
    所以是 2·M·N·K 次浮点运算。方阵 n 就是 2n³。

    四种精度各是什么：
      FP32  —— 32 位标准单精度。GPU 上走普通 CUDA core
      TF32  —— NVIDIA 的"阉割版 FP32"：尾数只留 10 位（FP32 是 23 位），
               指数范围不变，能走 Tensor Core，快很多而接口仍是 float32。
               ⚠️ torch 2.x 里 matmul 的 TF32 **默认关闭**，必须显式开，
               所以 FP32 与 TF32 是一组干净的单变量对照
      FP16  —— 16 位半精度。指数只有 5 位，动态范围窄，容易上下溢出
      BF16  —— 也是 16 位，但把位数分配给指数（8 位，和 FP32 一样宽），
               牺牲尾数精度换动态范围，所以比 FP16 稳，不太需要 loss scaling
    """
    rows: list[dict] = []

    # (标签, dtype, allow_tf32)
    configs = [
        ("fp32", torch.float32, False),
        ("tf32", torch.float32, True),
        ("fp16", torch.float16, False),
        ("bf16", torch.bfloat16, False),
    ]

    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
      for pass_idx in range(passes):
        order = tuple(sizes) if pass_idx % 2 == 0 else tuple(reversed(sizes))
        direction = "升序" if pass_idx % 2 == 0 else "倒序"
        print(f"\n--- 算力扫描 第{pass_idx + 1}遍（{direction}）---")
        print(f"{'尺寸':>7} {'精度':>6} {'中位耗时':>11} {'算力':>12} {'峰值显存':>9} "
              f"{'SM':>7} {'功耗':>7} {'温度':>6}")
        for n in order:
            for label, dtype, tf32 in configs:
                torch.backends.cuda.matmul.allow_tf32 = tf32
                try:
                    a = torch.randn(n, n, device="cuda", dtype=dtype)
                    b = torch.randn(n, n, device="cuda", dtype=dtype)
                    c = torch.empty(n, n, device="cuda", dtype=dtype)
                except torch.OutOfMemoryError:
                    print(f"{n:>7} {label:>6}   显存不足，跳过")
                    continue

                r = time_cuda(lambda: torch.matmul(a, b, out=c),
                              label=f"matmul {n} {label}", warmup=10, iters=50)
                tflops = (2 * n ** 3) / (r.median_ms / 1e3) / 1e12
                rows.append({
                    "kind": "compute", "dtype": label, "n": n,
                    "flops": 2 * n ** 3, "tflop_per_s": round(tflops, 2),
                    "allow_tf32": tf32, "pass": pass_idx, "direction": direction,
                    **r.as_dict(),
                })
                cond = r.conditions
                flag = "" if r.vram_headroom_ok else " ⚠️"
                print(f"{n:>7} {label:>6} {r.median_ms:>10.3f} ms {tflops:>10.2f} TFLOP/s "
                      f"{r.peak_reserved_gb:>7.2f} GB "
                      f"{cond.get('sm_clock_mhz_median', float('nan')):>7.0f} "
                      f"{cond.get('power_w_median', float('nan')):>6.0f}W "
                      f"{cond.get('temp_c_max', float('nan')):>5.0f}C{flag}")

                del a, b, c
                torch.cuda.empty_cache()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_tf32

    return rows


def summarise(rows: list[dict]) -> dict:
    """取每类的最佳实测值作为 roofline 的渐近线。"""
    bw = [r for r in rows if r["kind"] == "bandwidth"]
    comp = [r for r in rows if r["kind"] == "compute"]

    summary: dict = {}
    if bw:
        # ⭐ 漂移判定：同一 (尺寸, 模式) 在两遍之间的差异
        drift: dict = {}
        for r in bw:
            key = f"{r['mode']}@{r['working_set_mb']}MB"
            drift.setdefault(key, {})[f"pass{r.get('pass', 0)}"] = r["gb_per_s"]
        summary["drift_check"] = {
            k: {**v,
                "spread_pct": round(
                    100 * (max(v.values()) - min(v.values())) / max(v.values()), 1
                ) if len(v) > 1 else None}
            for k, v in drift.items()
        }
        # ⭐ 屋檐只能从【显存满频】的样本里取。
        # 实测显存频率会在 11001/12501/14001 MHz 间掉档（与温度、功耗墙均无关），
        # 带宽随之差 33–36%。混入降频样本，roofline 的上界本身就是脏的。
        full = [r for r in bw if r.get("mem_clock_ok")]
        triad_full = [r for r in full if r["mode"] == "triad"]
        copy_full = [r for r in full if r["mode"] == "copy"]
        low_triad = [r for r in bw if not r.get("mem_clock_ok") and r["mode"] == "triad"]

        pool = triad_full or [r for r in bw if r["mode"] == "triad"]
        summary["achieved_bandwidth_gb_s"] = max(r["gb_per_s"] for r in pool)
        summary["achieved_bandwidth_copy_gb_s"] = max(
            (r["gb_per_s"] for r in (copy_full or bw) if r["mode"] == "copy"), default=None
        )
        summary["spec_bandwidth_gb_s"] = 672.0
        summary["bandwidth_efficiency"] = round(
            summary["achieved_bandwidth_gb_s"] / 672.0, 3
        )
        summary["bandwidth_clock_gating"] = {
            "n_full_clock": len(full),
            "n_downclocked": len(bw) - len(full),
            "roof_from_full_clock_only": bool(triad_full),
            "triad_median_full_clock": round(
                statistics.median([r["gb_per_s"] for r in triad_full]), 1
            ) if triad_full else None,
            "triad_median_downclocked": round(
                statistics.median([r["gb_per_s"] for r in low_triad]), 1
            ) if low_triad else None,
        }
    if comp:
        cdrift: dict = {}
        for r in comp:
            key = f"{r['dtype']}@{r['n']}"
            cdrift.setdefault(key, {})[f"pass{r.get('pass', 0)}"] = r["tflop_per_s"]
        summary["compute_drift_check"] = {
            k: {**v, "spread_pct": round(
                100 * (max(v.values()) - min(v.values())) / max(v.values()), 1
            ) if len(v) > 1 else None}
            for k, v in cdrift.items()
        }
        peaks = {}
        for dtype in ("fp32", "tf32", "fp16", "bf16"):
            vals = [r["tflop_per_s"] for r in comp if r["dtype"] == dtype]
            if vals:
                peaks[dtype] = max(vals)
        summary["achieved_tflops"] = peaks
        if bw and peaks:
            # 脊点 = 可达算力 / 可达带宽，单位 FLOP/byte
            summary["ridge_point_flop_per_byte"] = {
                k: round(v * 1e12 / (summary["achieved_bandwidth_gb_s"] * 1e9), 1)
                for k, v in peaks.items()
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bandwidth", action="store_true")
    parser.add_argument("--compute", action="store_true")
    parser.add_argument("--out", type=str, default=str(RESULTS / "machine_limits.json"))
    args = parser.parse_args()

    run_all = not (args.bandwidth or args.compute)

    print("=" * 78)
    print("阶段 1：机器极限实测")
    print("=" * 78)
    props = torch.cuda.get_device_properties(0)
    print(f"{props.name} | sm_{torch.cuda.get_device_capability(0)[0]}"
          f"{torch.cuda.get_device_capability(0)[1]} | {props.multi_processor_count} SM")

    # ⭐ 先把 GPU 从低功耗档拉起来，否则测的是空闲态（见 EXPERIMENTS.md 发现 2）
    print("\n--- 预热：把 GPU 拉到满频 ---")
    mem_clock = ramp_clocks()

    rows: list[dict] = []
    if run_all or args.bandwidth:
        rows += bandwidth_sweep()
    if run_all or args.compute:
        rows += compute_sweep()

    summary = summarise(rows)
    summary["mem_clock_after_ramp_mhz"] = mem_clock

    print("\n" + "=" * 78)
    print("汇总（roofline 的两条渐近线取这里的值）")
    print("=" * 78)
    if "achieved_bandwidth_gb_s" in summary:
        print(f"可达带宽 (triad 读写混合) : {summary['achieved_bandwidth_gb_s']:.1f} GB/s"
              f"   = 标称 672 的 {summary['bandwidth_efficiency'] * 100:.0f}%")
        if summary.get("achieved_bandwidth_copy_gb_s"):
            print(f"可达带宽 (copy 纯搬运)   : {summary['achieved_bandwidth_copy_gb_s']:.1f} GB/s")
    for dtype, tflops in summary.get("achieved_tflops", {}).items():
        ridge = summary.get("ridge_point_flop_per_byte", {}).get(dtype)
        print(f"可达算力 {dtype:>5}              : {tflops:7.2f} TFLOP/s"
              + (f"   脊点 {ridge} FLOP/byte" if ridge else ""))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2,
                              ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
