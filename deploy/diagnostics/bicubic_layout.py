"""阶段 3 · P4 被推翻之后：DINOv2 的 pos_embed 插值为什么在 GPU 上要 ~720 µs。

模型内测到插值那段 GPU 执行 ~720 µs，而同样尺寸的插值单独测只要 ~92 µs。
逐个变量拆开：
  ① 固定输出 16×16，改变通道数           —— 看耗时是否与通道数成正比
  ② 固定 384 通道，改变输出尺寸
  ③ 2×2：输入布局（DINOv2 的不连续视图 vs 连续 NCHW）× 尺寸参数（带偏移 scale_factor vs size）
     并检查两种布局的输出是否逐位相同
"""

from deploy.diagnostics._common import save

import torch
import torch.nn.functional as F

from deploy.bench.timing import time_cuda

M, D, OUT = 37, 384, 16              # 518px 训练的 37×37 patch 网格；224px 推理的 16×16


def t_us(fn, iters: int = 300) -> float:
    return time_cuda(fn, warmup=20, iters=iters, sample_conditions=False).median_ms * 1000


def main() -> None:
    res: dict = {}

    print("① 固定输出 16×16，改变通道数（连续 NCHW）")
    res["channels"] = {}
    for c in (48, 96, 192, 384, 768):
        x = torch.randn(1, c, M, M, device="cuda")
        us = t_us(lambda: F.interpolate(x, size=(OUT, OUT), mode="bicubic", align_corners=False))
        res["channels"][c] = round(us, 1)
        print(f"   C={c:<4} {us:8.1f} µs   每通道 {us / c:.2f} µs")

    print("② 固定 384 通道，改变输出尺寸（连续 NCHW）")
    res["out_size"] = {}
    x = torch.randn(1, D, M, M, device="cuda")
    for o in (8, 16, 32, 64):
        us = t_us(lambda: F.interpolate(x, size=(o, o), mode="bicubic", align_corners=False))
        res["out_size"][o] = round(us, 1)
        print(f"   输出 {o}×{o}  {us:8.1f} µs")

    print("③ 布局 × 尺寸参数")
    src = torch.randn(1, M * M, D, device="cuda")
    perm = src.reshape(1, M, M, D).permute(0, 3, 1, 2)     # DINOv2 的实际写法
    cont = perm.contiguous()
    s = (OUT + 0.1) / M                                    # interpolate_offset = 0.1
    cells = {
        "不连续视图 + scale_factor": lambda: F.interpolate(perm, scale_factor=(s, s), mode="bicubic"),
        "不连续视图 + size": lambda: F.interpolate(perm, size=(OUT, OUT), mode="bicubic"),
        "连续 NCHW + scale_factor": lambda: F.interpolate(cont, scale_factor=(s, s), mode="bicubic"),
        "连续 NCHW + size": lambda: F.interpolate(cont, size=(OUT, OUT), mode="bicubic"),
    }
    res["layout"] = {k: round(t_us(fn), 1) for k, fn in cells.items()}
    for k, v in res["layout"].items():
        print(f"   {k:<24}{v:8.1f} µs")
    res["layout_slowdown_x"] = round(res["layout"]["不连续视图 + scale_factor"]
                                     / res["layout"]["连续 NCHW + scale_factor"], 2)
    res["noncontig_strides"] = list(perm.stride())
    a = F.interpolate(perm, scale_factor=(s, s), mode="bicubic").contiguous()
    b = F.interpolate(cont, scale_factor=(s, s), mode="bicubic").contiguous()
    res["outputs_bit_identical"] = bool(torch.equal(a, b))
    print(f"   只改布局慢 {res['layout_slowdown_x']}×；输出逐位相同 = {res['outputs_bit_identical']}")
    save("bicubic_layout", res)


if __name__ == "__main__":
    main()
