"""阶段 2：把基线扫描画成"延迟 vs batch"小多图（三档分辨率各一格）。

读图要点：
* 纵轴对数、三格共享同一刻度——同一个量，放在同一把尺子上比
* 色带 = 两遍测量的范围。色带宽 = 这个点不稳定（两遍差 >5%）
* 大 batch 时各线平行上升（GPU 受限，延迟 ∝ 工作量）；
  小 batch 的半精度线贴着一个与分辨率无关的"地板"——这是命题 (b) 的图像特征

配色与 roofline 图完全一致（颜色跟随精度，不跟随排名）。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator

from deploy.bench.roofline import (INK, INK_MUTED, INK_SECONDARY, SERIES, SURFACE,
                                   _use_cjk_font)

RESULTS = Path(__file__).resolve().parents[1] / "results"
BATCHES = [1, 2, 4, 8, 16, 32]
RESES = [224, 336, 448]
V = "dinov2_vits14"


def main() -> None:
    _use_cjk_font()
    raw = json.loads((RESULTS / "baseline_torch_raw.json").read_text(encoding="utf-8"))
    comb = json.loads((RESULTS / "baseline_torch.json").read_text(encoding="utf-8"))
    c = {(r["variant"], r["precision"], r["batch"], r["res"]): r for r in comb}
    passes: dict[tuple, list[float]] = {}
    for r in raw:
        if r["status"] == "ok" and r["pass"] >= 0:
            passes.setdefault((r["variant"], r["precision"], r["batch"], r["res"]), []).append(r["median_ms"])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2), dpi=160, sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, res in zip(axes, RESES):
        ax.set_facecolor(SURFACE)
        for prec, color in SERIES.items():
            ys = [c[(V, prec, b, res)]["median_ms"] for b in BATCHES]
            lo = [min(passes[(V, prec, b, res)]) for b in BATCHES]
            hi = [max(passes[(V, prec, b, res)]) for b in BATCHES]
            ax.fill_between(BATCHES, lo, hi, color=color, alpha=0.18, lw=0, zorder=2)
            ax.plot(BATCHES, ys, color=color, lw=2.0, zorder=3, label=prec.upper(),
                    marker="o", ms=6, mec=SURFACE, mew=1.5, solid_capstyle="round")
            if res == RESES[-1]:
                # 直接标注（4 槽色板的 relief 规则）；文字用墨色，身份由相邻线条承载。
                # FP16 与 BF16 在右端几乎重合（~65 ms），上下错开
                dy = {"fp16": -7, "bf16": 7}.get(prec, 0)
                ax.annotate(prec.upper(), xy=(BATCHES[-1], ys[-1]), xytext=(7, dy),
                            textcoords="offset points", fontsize=9, color=INK,
                            va="center", fontweight="medium")

        tokens = (res // 14) ** 2 + 1
        ax.set_title(f"{res}px（{tokens} token）", fontsize=10.5, color=INK, loc="left")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(FixedLocator(BATCHES))
        ax.xaxis.set_major_formatter(FixedFormatter([str(b) for b in BATCHES]))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.set_xlim(0.85, 32 * (1.55 if res == RESES[-1] else 1.18))
        ax.set_xlabel("batch", fontsize=9.5, color=INK_SECONDARY)
        ax.grid(True, which="major", lw=0.6, color="#e3e2de", zorder=1)
        ax.grid(True, which="minor", axis="y", lw=0.35, color="#eeedea", zorder=1)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#d6d5d1")
        ax.tick_params(colors=INK_SECONDARY, labelsize=8.5)

    yt = [2, 5, 10, 20, 50, 100, 200]
    axes[0].yaxis.set_major_locator(FixedLocator(yt))
    axes[0].yaxis.set_major_formatter(FixedFormatter([str(v) for v in yt]))
    axes[0].set_ylim(1.6, 330)
    axes[0].set_ylabel("中位延迟（ms，对数）", fontsize=9.5, color=INK_SECONDARY)

    # 地板参考线：只标在 224 格，说明它的来历，不在另两格重复
    axes[0].axhline(2.05, color=INK_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    # 说明文字放在左上空白处，避开 x 轴刻度
    axes[0].text(1.0, 230, "虚线 ≈ 2 ms 地板：\n小 batch 的半精度线贴着它，\n且与分辨率无关"
                 "（对照 336/448 两格）\n→ 疑似 CPU 发射 kernel 受限，\n    阶段 3 用 profiler 验证",
                 fontsize=7.8, color=INK_SECONDARY, va="top", linespacing=1.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncols=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.995, 1.0))
    fig.suptitle("ViT-S/14 基线延迟 vs batch（PyTorch eager，整体转换精度）",
                 x=0.012, ha="left", fontsize=12.5, color=INK)
    fig.text(0.012, 0.012,
             "色带 = 两遍 A/B/A 测量的范围（宽 = 不稳定）。每点 300 次迭代的中位数，两遍取均值。"
             "只测 GPU 前向，不含解码/预处理/拷贝。插电，功耗 67–140 W。",
             fontsize=7, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    out = RESULTS / "baseline_latency.png"
    fig.savefig(out, facecolor=SURFACE)
    print(f"已写入 {out}")


if __name__ == "__main__":
    main()
