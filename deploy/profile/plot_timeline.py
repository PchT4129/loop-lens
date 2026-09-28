"""阶段 3：一次前向的 CPU 发射 vs GPU 执行时间线——第 A.7 节两张示意图的真实版本。

上下两格只差一个变量：batch（1 vs 32），同为 BF16、224px。
  * batch 1：GPU 上的 kernel 条之间布满空隙——GPU 在等 CPU（发射受限）
  * batch 32：CPU 很早就发射完了，GPU 条连成一片——CPU 在等 GPU（GPU 受限）

⚠️ 图中数据取自 profiler 运行，而 profiler 会拖慢 CPU 侧（观察者效应），所以
batch 1 的空隙比真实情况更宽。真实（无 profiler）的忙碌占比见 EXPERIMENTS.md。
两格横轴刻度各自独立（它们的总时长差 3 倍，共用刻度会让 batch 1 的空隙看不见）。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deploy.bench.roofline import INK, INK_MUTED, INK_SECONDARY, SURFACE, _use_cjk_font

RESULTS = Path(__file__).resolve().parents[1] / "results"
# 类别色：分类色板槽 1、2；"其他"用中性灰——它是剩余项，不是一个需要强调的序列
COLORS = {"矩阵乘 GEMM": "#2a78d6", "注意力": "#eb6834"}
OTHER = "#b9b8b3"
CPU_INK = "#52514e"


def draw(ax, tl: dict, title: str, note: str) -> None:
    ax.set_facecolor(SURFACE)
    y_cpu, y_gpu, h = 1.0, 0.0, 0.62
    for s, e in tl["cpu_launches_us"]:
        ax.barh(y_cpu, max(e - s, 1.5) / 1000, left=s / 1000, height=h, color=CPU_INK,
                lw=0, zorder=3)
    for s, e, cat in tl["gpu_kernels_us"]:
        ax.barh(y_gpu, max(e - s, 1.5) / 1000, left=s / 1000, height=h,
                color=COLORS.get(cat, OTHER), lw=0, zorder=3)

    kt = sum(e - s for s, e, _ in tl["gpu_kernels_us"])
    busy = kt / tl["gpu_span_us"]
    ax.set_yticks([y_gpu, y_cpu])
    ax.set_yticklabels(["GPU：执行 kernel", "CPU：发射 kernel"], fontsize=9, color=INK)
    ax.set_ylim(-0.6, 1.6)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=6)
    ax.text(1.0, 1.02, note.format(busy=100 * busy, kt=kt / 1000, span=tl["gpu_span_us"] / 1000,
                                   cpu=tl["cpu_span_us"] / 1000),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=INK_SECONDARY)
    ax.set_xlabel("时间（ms，从这次前向的 CPU 端开始计）", fontsize=8.5, color=INK_SECONDARY)
    ax.grid(True, axis="x", lw=0.5, color="#e3e2de", zorder=1)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d6d5d1")
    ax.tick_params(axis="x", colors=INK_SECONDARY, labelsize=8)
    ax.tick_params(axis="y", length=0)


def main() -> None:
    _use_cjk_font()
    d = json.loads((RESULTS / "profile_torch.json").read_text(encoding="utf-8"))
    t1, t32 = d["timelines"]["bf16_b1_224"], d["timelines"]["bf16_b32_224"]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 5.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    draw(a1, t1, "batch 1：GPU 在等 CPU（发射受限）",
         "GPU 实际干活 {kt:.2f} ms / 跨度 {span:.2f} ms → 忙碌 {busy:.0f}%，其余是空隙")
    draw(a2, t32, "batch 32：CPU 在等 GPU（GPU 受限）",
         "CPU {cpu:.1f} ms 就发射完了；GPU 干活 {kt:.1f} ms / 跨度 {span:.1f} ms → 忙碌 {busy:.0f}%")
    a1.set_xlim(0, max(t1["gpu_span_us"] + t1["gpu_start_offset_us"], t1["cpu_span_us"]) / 1000 * 1.02)
    a2.set_xlim(0, max(t32["gpu_span_us"] + t32["gpu_start_offset_us"], t32["cpu_span_us"]) / 1000 * 1.02)

    # batch 32 的 GPU 泳道前段是空白，但那不是 GPU 空闲：CPU 一路领先，GPU 此时正在执行
    # 之前几次前向排队的 kernel（图里只画了"这一次"前向的 kernel）。不标注会被误读。
    off = t32["gpu_start_offset_us"] / 1000
    depth = t32["gpu_start_offset_us"] / t32["gpu_span_us"]
    a2.barh(0.0, off, left=0, height=0.62, color="#e9e8e4", hatch="////", edgecolor="#cfcec9",
            lw=0, zorder=2)
    a2.text(off / 2, 0.0, f"GPU 并没有闲着：它正在执行此前排队的约 {depth:.1f} 次前向（未画出）",
            ha="center", va="center", fontsize=8.5, color=INK_SECONDARY, zorder=4)
    a2.annotate("", xy=(off, 1.0), xytext=(t32["cpu_span_us"] / 1000, 1.0),
                arrowprops=dict(arrowstyle="->", lw=0.9, color=INK_MUTED))
    a2.text((t32["cpu_span_us"] / 1000 + off) / 2, 1.2,
            f"CPU 领先 GPU 约 {off:.0f} ms——任务队列里积压着活", ha="center", va="bottom",
            fontsize=8.5, color=INK_SECONDARY)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()]
    handles += [plt.Rectangle((0, 0), 1, 1, color=OTHER), plt.Rectangle((0, 0), 1, 1, color=CPU_INK)]
    fig.legend(handles, list(COLORS) + ["其他（LayerNorm、GELU、加/乘…）", "CPU 发射调用"],
               loc="upper right", ncols=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.995, 1.0))
    fig.suptitle("一次前向的时间线：ViT-S/14 · BF16 · 224px（两格只差 batch）",
                 x=0.012, ha="left", fontsize=12.5, color=INK)
    fig.text(0.012, 0.012,
             "数据来自 torch.profiler。profiler 会拖慢 CPU 侧，batch 1 的空隙比真实情况更宽"
             "（无 profiler 时 GPU 忙碌约 30%）。两格横轴刻度各自独立。",
             fontsize=7, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94), h_pad=2.2)
    out = RESULTS / "timeline_b1_vs_b32.png"
    fig.savefig(out, facecolor=SURFACE)
    print(f"已写入 {out}")


if __name__ == "__main__":
    main()
