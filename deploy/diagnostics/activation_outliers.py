"""阶段 4c · R5 崩溃之后：INT8 为什么把 DINOv2 的特征打坏了。

INT8（隐式量化，Entropy 校准）让特征余弦均值跌到 0.74、held-out R@1 从 0.933 掉到 0.633。
解码校准缓存发现：某些 LayerNorm 输出的真实最大值远大于 Entropy 选的截断点（例如 layer_norm_1：
MinMax 117.7 vs Entropy 15.7）。假设：**少数通道上的巨大激活（离群值）是模型必需的信息，
逐张量 INT8 要么把它们截掉（Entropy），要么为了容纳它们把其余数值的分辨率压得很粗（MinMax）。**

两部分：
  ① 观察：FP32 模型上挂钩子，统计 25 个 LayerNorm 输出的 max|x|、p99.9|x|、离群通道
  ② 操纵（单变量）：只把这 25 个 LayerNorm 输出做逐张量 INT8 伪量化，缩放因子直接取 TensorRT 自己的
     校准缓存（Entropy / MinMax），其余全部保持 FP32。若仅此一处就复现出崩溃的量级，机制成立。

伪量化（fake quantization）：x → round(x / s) 截断到 [-127, 127] → 再乘回 s。数值仍是浮点，
但只剩 INT8 能表示的那 255 个值——在 PyTorch 里模拟"这一步变成 INT8 会损失多少"。
"""

import struct

from deploy.diagnostics._common import REPO, save

import torch

from deploy.eval.compare_features import compare
from deploy.eval.extract_with_runtime import extract
from deploy.eval.runtimes import _base_fp32
from deploy.export.calibrate import load_calibration_images

ENG = REPO / "deploy" / "results" / "engines"
OUT = REPO / "deploy" / "results" / "runtime"


def load_amax(cache_name: str) -> dict[str, float]:
    """TensorRT 校准缓存：每行 `张量名: 缩放因子的大端 float32 十六进制`。amax = scale × 127。"""
    out = {}
    for line in (ENG / cache_name).read_text().splitlines()[1:]:
        name, hexval = line.rsplit(":", 1)
        out[name.strip()] = struct.unpack("!f", bytes.fromhex(hexval.strip()))[0] * 127.0
    return out


def layernorms(model) -> list:
    """按 ONNX 导出顺序排列：layer_norm = block0.norm1, layer_norm_1 = block0.norm2, ..., layer_norm_24 = 最后的 norm。"""
    mods = []
    for blk in model.blocks:
        mods += [blk.norm1, blk.norm2]
    return mods + [model.norm]


def onnx_name(i: int) -> str:
    return "layer_norm" if i == 0 else f"layer_norm_{i}"


def observe(model) -> list[dict]:
    stats: dict[int, dict] = {}

    def hook(i):
        def fn(_m, _inp, out):
            a = out.detach().abs().float()                         # [1, 257, 384]
            s = stats.setdefault(i, {"max": 0.0, "q": [], "ch": torch.zeros(a.shape[-1], device=a.device)})
            s["max"] = max(s["max"], float(a.max()))
            s["q"].append(float(torch.quantile(a.flatten(), 0.999)))
            s["ch"] = torch.maximum(s["ch"], a.amax(dim=(0, 1)))
        return fn

    hs = [m.register_forward_hook(hook(i)) for i, m in enumerate(layernorms(model))]
    with torch.inference_mode():
        for x in load_calibration_images("day"):
            model(x.cuda())
    for h in hs:
        h.remove()

    rows = []
    for i, s in sorted(stats.items()):
        p999 = sum(s["q"]) / len(s["q"])
        top = torch.topk(s["ch"], 3)
        rows.append({"layer": onnx_name(i), "max_abs": round(s["max"], 2), "p99_9_abs": round(p999, 2),
                     "max_over_p99_9": round(s["max"] / p999, 1),
                     "top_channels": top.indices.tolist(), "top_channel_max": [round(v, 1) for v in top.values.tolist()],
                     "median_channel_max": round(float(s["ch"].median()), 2)})
    return rows


def simulate(model, amax: dict[str, float], tag: str) -> dict:
    def hook(i):
        s = amax[onnx_name(i)] / 127.0

        def fn(_m, _inp, out):
            return torch.clamp(torch.round(out / s), -127, 127) * s
        return fn

    hs = [m.register_forward_hook(hook(i)) for i, m in enumerate(layernorms(model))]
    try:
        extract(lambda x: model(x), {"kind": "simulation", "what": f"fake-quant LayerNorm outputs only ({tag})"},
                OUT / f"sim_lnq_{tag}", batch_size=1)
    finally:
        for h in hs:
            h.remove()
    return compare(OUT / f"sim_lnq_{tag}")


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = _base_fp32()

    print("① 观察：25 个 LayerNorm 输出（50 张校准图）")
    rows = observe(model)
    ent, mm = load_amax("calib_entropy_day.cache"), load_amax("calib_minmax_day.cache")
    print(f"   {'层':<14}{'max|x|':>8}{'p99.9':>8}{'倍数':>7}{'TRT MinMax':>12}{'TRT Entropy':>13}   最大的 3 个通道（中位通道 max）")
    for r in rows:
        print(f"   {r['layer']:<14}{r['max_abs']:>8}{r['p99_9_abs']:>8}{r['max_over_p99_9']:>7}"
              f"{mm[r['layer']]:>12.1f}{ent[r['layer']]:>13.1f}   {r['top_channels']} {r['top_channel_max']} ({r['median_channel_max']})")

    print("② 操纵：只伪量化 LayerNorm 输出，其余 FP32")
    sims = {}
    for tag, amax in (("entropy", ent), ("minmax", mm)):
        r = simulate(model, amax, tag)
        sims[tag] = r
        c, a, g = r["cosine"], r["retrieval_all"], r["frozen_gate_test"]
        print(f"   {tag:<8} 1−cos {c['one_minus_mean']:.2e}  最小 cos {c['min']:.4f}  top-1 翻转 {a['top1_flips']}/200"
              f"  最大分数漂移 {a['score_shift_max_abs']:.3f}  决策翻转 {g['decision_flips']}/50")
    save("activation_outliers", {"layernorm_stats": rows, "simulation": sims})


if __name__ == "__main__":
    main()
