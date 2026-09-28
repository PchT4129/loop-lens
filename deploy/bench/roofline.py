"""阶段 1：把实测的机器极限画成 roofline 图。

Roofline 在读什么
-----------------
横轴 **算术强度**（arithmetic intensity, AI）= 每从显存搬运 1 字节，能做多少次
浮点运算，单位 FLOP/byte。它是算子的**固有属性**，由算法和数据类型决定。
纵轴是实际达到的算力 TFLOP/s。

任何算子都被两条线夹住：

    水平的"屋顶"      = 可达算力            —— 再高的 AI 也突破不了
    斜的"屋檐"        = 可达带宽 × AI       —— 搬数据速度决定的上界

交点叫**脊点**（ridge point），位置 = 可达算力 / 可达带宽。
脊点左边 = 访存受限（memory-bound，加速要减少搬数据）；
右边 = 算力受限（compute-bound，加速要用更快的数学单元，比如降精度上 Tensor Core）。

**为什么降精度不一定能加速**：低精度把屋顶抬高（FP16 的屋顶是 FP32 的 3 倍），
但屋檐不动。如果算子本来就在脊点左边（访存受限），抬屋顶毫无用处——
这正是本项目要验证的核心假设。

用法：
    python -m deploy.bench.roofline
    python -m deploy.bench.roofline --points deploy/results/op_intensity.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

RESULTS = Path(__file__).resolve().parents[1] / "results"

# --- 配色：来自已通过校验的分类色板（4 槽，相邻对 CVD ΔE 9.1 / 常视 22.9 均过闸）---
# aqua 与 yellow 在浅色底上对比度低于 3:1，按 relief 规则**必须**直接标注，
# 所以每条屋顶都带文字标签——roofline 本来也该这样画。
SERIES = {
    "fp32": "#2a78d6",   # 槽1 blue
    "tf32": "#eb6834",   # 槽2 orange
    "fp16": "#1baf7a",   # 槽3 aqua
    "bf16": "#eda100",   # 槽4 yellow
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"


def _use_cjk_font() -> None:
    """注册 Windows 侧的 SimHei，否则中文会渲染成方框。"""
    for path in ("/mnt/c/Windows/Fonts/simhei.ttf", "/mnt/c/Windows/Fonts/msyh.ttc"):
        if Path(path).exists():
            try:
                font_manager.fontManager.addfont(path)
                name = font_manager.FontProperties(fname=path).get_name()
                plt.rcParams["font.family"] = [name, "DejaVu Sans"]
                plt.rcParams["axes.unicode_minus"] = False
                # SimHei 没有数学字体的 U+2212 减号，对数轴的 10^-1 会缺字形
                plt.rcParams["mathtext.fontset"] = "dejavusans"
                return
            except Exception:                                 # noqa: BLE001
                continue
    print("  ⚠️ 未找到中文字体，图内改用英文标签")


def plot_roofline(summary: dict, op_points: list[dict], out_path: Path) -> None:
    bw_gbs = summary["achieved_bandwidth_gb_s"]
    peaks = summary["achieved_tflops"]
    bw_tflop_per_flopbyte = bw_gbs / 1000.0          # GB/s -> TFLOP/s per (FLOP/byte)

    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=170)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ai = np.logspace(-1.1, 3.2, 400)

    # --- 屋檐：可达带宽（中性色，它不是"某个序列"，是所有精度共享的访存上界）---
    ax.plot(ai, ai * bw_tflop_per_flopbyte, lw=2.0, color=INK_SECONDARY,
            zorder=3, solid_capstyle="round")
    # 沿屋檐斜着标注。位置要避开左下角的 triad 锚点标注，所以放在 x≈3 处
    # 沿屋檐斜标注。文字抬离线体一点，且只写关键量，百分比留给脚注
    ax.text(1.6, 1.6 * bw_tflop_per_flopbyte * 1.45,
            f"实测可达带宽 {bw_gbs:.0f} GB/s",
            fontsize=9, color=INK_SECONDARY, rotation=31,
            rotation_mode="anchor", ha="left", va="bottom")

    # --- 屋顶：各精度可达算力 + 脊点 ---
    # 屋顶值很接近时（FP16 54.5 与 BF16 59.8）标注会重叠，所以按序错开上下
    label_side = {"fp32": "below", "tf32": "above", "fp16": "below", "bf16": "above"}
    for dtype, color in SERIES.items():
        if dtype not in peaks:
            continue
        peak = peaks[dtype]
        ridge = peak / bw_tflop_per_flopbyte
        above = label_side.get(dtype) == "above"

        ax.plot([ridge, ai[-1]], [peak, peak], lw=2.0, color=color,
                zorder=4, solid_capstyle="round", label=f"{dtype.upper()}")
        # 脊点标记（>=8px）
        ax.plot([ridge], [peak], marker="o", ms=8, color=color,
                mec=SURFACE, mew=2.0, zorder=5)
        # 直接标注（relief 规则要求，也是 roofline 的常规读法）
        # ⚠️ 标注文字用墨色而非序列色：yellow 在浅底上对比度仅 2.11:1、aqua 2.74:1，
        # 用彩色写字会读不清。身份由紧邻的彩色屋顶线承载，文字不必再上色。
        ax.text(ai[-1] * 0.88, peak * (1.09 if above else 0.84),
                f"{dtype.upper()}  {peak:.1f} TFLOP/s",
                fontsize=9, color=INK, ha="right",
                va="bottom" if above else "top", fontweight="medium")
        ax.text(ridge * (1.14 if above else 0.88), peak * (1.02 if above else 0.97),
                f"脊点 {ridge:.0f}", fontsize=7.5, color=INK_SECONDARY,
                ha="left" if above else "right", va="bottom" if above else "top")

    # --- 实测锚点：triad 微基准应当正好落在屋檐上 ---
    triad_ai = 2 / 12          # 每元素 2 FLOP、搬 12 字节（读2写1 的 float32）
    triad_tflops = triad_ai * bw_tflop_per_flopbyte
    ax.plot([triad_ai], [triad_tflops], marker="D", ms=8, color=INK,
            mec=SURFACE, mew=1.8, zorder=6)
    ax.annotate(f"triad 微基准 {triad_ai:.2f} FLOP/byte\n落在屋檐上（锚定上界）",
                xy=(triad_ai, triad_tflops),
                xytext=(triad_ai * 1.25, triad_tflops * 4.2),
                fontsize=8, color=INK_SECONDARY, ha="left", linespacing=1.4,
                arrowprops=dict(arrowstyle="-", lw=0.9, color=INK_MUTED))

    # --- 阶段 3 会把 ViT 各算子标进来 ---
    for pt in op_points:
        ax.plot([pt["ai"]], [pt["tflops"]], marker="s", ms=7,
                color=pt.get("color", INK), mec=SURFACE, mew=1.5, zorder=6)
        ax.annotate(pt["label"], xy=(pt["ai"], pt["tflops"]), fontsize=7.5,
                    color=INK_SECONDARY, xytext=(4, 4), textcoords="offset points")

    # --- 两个区域的说明 ---
    fp32_ridge = peaks["fp32"] / bw_tflop_per_flopbyte
    ax.axvspan(ai[0], fp32_ridge, color=INK_MUTED, alpha=0.05, zorder=0)
    ax.text(0.105, 110, "← 访存受限\n瓶颈在搬数据\n降精度基本无用",
            fontsize=8.5, color=INK_SECONDARY, va="top", linespacing=1.5)
    ax.text(900, 0.9, "算力受限 →\n瓶颈在算\n降精度有效",
            fontsize=8.5, color=INK_SECONDARY, ha="right", va="bottom", linespacing=1.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(ai[0], ai[-1])
    ax.set_ylim(0.05, 150)
    ax.set_xlabel("算术强度  (FLOP / byte)", fontsize=10, color=INK_SECONDARY)
    ax.set_ylabel("可达算力  (TFLOP/s)", fontsize=10, color=INK_SECONDARY)
    ax.set_title("RTX 5070 Ti Laptop (sm_120) 实测 Roofline",
                 fontsize=12.5, color=INK, pad=12, loc="left")

    # 网格与坐标轴保持退让
    # 显式写成普通数字，而不是 10^-1 这种指数记法：
    # 一是对读者更直观，二是绕开 SimHei 缺数学减号字形的问题
    from matplotlib.ticker import FixedFormatter, FixedLocator
    ax.xaxis.set_major_locator(FixedLocator([0.1, 1, 10, 100, 1000]))
    ax.xaxis.set_major_formatter(FixedFormatter(["0.1", "1", "10", "100", "1000"]))
    ax.yaxis.set_major_locator(FixedLocator([0.05, 0.2, 1, 5, 20, 100]))
    ax.yaxis.set_major_formatter(FixedFormatter(["0.05", "0.2", "1", "5", "20", "100"]))

    ax.grid(True, which="major", lw=0.6, color="#e3e2de", zorder=1)
    ax.grid(True, which="minor", lw=0.35, color="#eeedea", zorder=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d6d5d1")
    ax.tick_params(colors=INK_SECONDARY, labelsize=8.5)

    # 多序列必须有图例（同时已直接标注，二者并存）
    ax.legend(loc="lower right", frameon=True, fontsize=8.5, ncols=4,
              facecolor=SURFACE, edgecolor="#e3e2de", title="可达算力屋顶",
              title_fontsize=8.5, bbox_to_anchor=(1.0, 0.0))

    fig.text(0.008, 0.012,
             f"屋檐 = triad 实测 {bw_gbs:.0f} GB/s，为标称 672 的 {bw_gbs / 672 * 100:.0f}%，"
             f"仅取显存满频样本；屋顶 = 大方阵 matmul 实测，双向 A/B/A 两遍取最优。"
             f"条件：插电、显存 14001 MHz、功耗 134–149 W、温度 ≤79°C",
             fontsize=7, color=INK_MUTED)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    print(f"已写入 {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limits", type=str, default=str(RESULTS / "machine_limits.json"))
    parser.add_argument("--points", type=str, default=None,
                        help="可选：算子标点 JSON，[{label, ai, tflops}]（阶段 3 用）")
    parser.add_argument("--out", type=str, default=str(RESULTS / "roofline.png"))
    args = parser.parse_args()

    _use_cjk_font()
    data = json.loads(Path(args.limits).read_text(encoding="utf-8"))
    summary = data["summary"]

    missing = [k for k in ("achieved_bandwidth_gb_s", "achieved_tflops") if k not in summary]
    if missing:
        raise SystemExit(
            f"{args.limits} 缺少 {missing}——需要先把带宽和算力都跑完："
            f"python -m deploy.bench.machine_limits"
        )

    op_points = json.loads(Path(args.points).read_text(encoding="utf-8")) if args.points else []

    print("Roofline 参数：")
    print(f"  屋檐 可达带宽 {summary['achieved_bandwidth_gb_s']:.1f} GB/s")
    for d, v in summary["achieved_tflops"].items():
        print(f"  屋顶 {d:>5} {v:6.2f} TFLOP/s   脊点 "
              f"{v / (summary['achieved_bandwidth_gb_s'] / 1000):6.1f} FLOP/byte")

    plot_roofline(summary, op_points, Path(args.out))


if __name__ == "__main__":
    main()
