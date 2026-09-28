"""VPRDescriptor：把 DINOv2 + 聚合 + L2 归一化封装成"输入图像张量，输出描述子张量"的纯函数。

为什么不能直接用 src.backbones.DINOv2FeatureExtractor
---------------------------------------------------
它对训练/评测完全够用，但对部署有两个问题：

1. **forward_features 返回 dict**。ONNX / TensorRT 只认张量输入输出，
   dict 里还混着 `masks=None` 这种非张量值，导出时要么报错要么被静默丢弃。

2. **位置编码每次前向都在重新插值**。DINOv2 的 pos_embed 按 518px 训练，
   形状 (1, 1370, 384) = 37×37 个 patch + 1 个 cls。我们推理用 224px，只有
   16×16 = 256 个 patch，于是 `interpolate_pos_encoding` 的快速返回条件
   `npatch == N` 永远不成立，每次都要跑一次 bicubic 插值。

   实测 hub 上的 dinov2_vits14 用的是 antialias=False、interpolate_offset=0.1。
   offset 意味着传给插值的缩放系数是 (16+0.1)/37 而不是 16/37——PyTorch 用这个
   带偏移的系数做坐标映射。ONNX 的 Resize 算子虽然支持 cubic 模式，但导出器
   要精确复现"带偏移系数 + half_pixel 坐标变换"的组合，这是数值偏差的常见来源。

   而推理期分辨率是固定的，插值结果是个常量。所以这里**预先算一次、存成 buffer**：
   数值与原实现逐位一致（就是同一次 PyTorch 插值的结果），同时消掉这个算子，
   导出图里也就不存在这个风险点了。

契约：输入 [B, 3, S, S]（已按 ImageNet 均值方差归一化），输出 [B, D]，已 L2 归一化。
与仓库 {features, paths, meta} 下游契约完全一致。

用法：
    python -m deploy.export.wrapper --self-check     # 等价性闸门（阶段 2 必须先过）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.backbones import DINOv2FeatureExtractor          # noqa: E402

RESULTS = REPO / "deploy" / "results"
PATCH_SIZE = 14


class VPRDescriptor(nn.Module):
    """复刻 DINOv2FeatureExtractor 的前向，唯一改动是 pos_embed 固化为常量。

    不复制任何权重：patch_embed / cls_token / blocks / norm 都直接引用原模型的子模块。
    这样"包装"不可能在权重层面引入差异，等价性只取决于前向逻辑是否复刻正确。
    """

    def __init__(self, extractor: DINOv2FeatureExtractor, image_size: int):
        super().__init__()
        if image_size % PATCH_SIZE:
            raise ValueError(f"image_size={image_size} 不是 patch size {PATCH_SIZE} 的倍数")

        vit = extractor.model
        if vit.num_register_tokens:
            # 带 register 的变体 token 排布不同，这里没有复刻，宁可显式拒绝
            raise NotImplementedError("暂不支持 *_reg 变体")

        self.patch_embed = vit.patch_embed
        self.cls_token = vit.cls_token
        self.blocks = vit.blocks
        self.norm = vit.norm
        self.aggregation = extractor.aggregation
        self.gem = extractor.gem
        self.image_size = image_size
        self.feature_dim = extractor.feature_dim

        # ⭐ 核心：调用原模型自己的 interpolate_pos_encoding 算一次，结果存成 buffer。
        # 用原函数而不是自己重写插值，保证与原实现逐位一致。
        n_patch = (image_size // PATCH_SIZE) ** 2
        probe = torch.zeros(1, n_patch + 1, vit.embed_dim,
                            dtype=vit.pos_embed.dtype, device=vit.pos_embed.device)
        with torch.no_grad():
            pos = vit.interpolate_pos_encoding(probe, image_size, image_size)
        self.register_buffer("pos_embed", pos.detach().clone(), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 等价于 prepare_tokens_with_masks(x, masks=None)，只是 pos_embed 换成常量
        b = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(b, -1, -1), x), dim=1)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        cls, patches = x[:, 0], x[:, 1:]
        if self.aggregation == "cls":
            feats = cls
        elif self.aggregation == "mean":
            feats = patches.mean(dim=1)
        elif self.aggregation == "gem":
            feats = self.gem(patches.transpose(1, 2).unsqueeze(-1)).flatten(1)
        elif self.aggregation == "cls+gem":
            gem = self.gem(patches.transpose(1, 2).unsqueeze(-1)).flatten(1)
            feats = torch.cat([cls, gem], dim=1)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation!r}")
        return F.normalize(feats, p=2, dim=1)


def build_descriptor(
    variant: str = "dinov2_vits14",
    aggregation: str = "mean",
    image_size: int = 224,
    device: str = "cuda",
) -> tuple[VPRDescriptor, DINOv2FeatureExtractor]:
    """返回 (包装后的模型, 原始提取器)。二者共享同一份权重。"""
    extractor = DINOv2FeatureExtractor(variant=variant, aggregation=aggregation).to(device).eval()
    wrapped = VPRDescriptor(extractor, image_size).to(device).eval()
    return wrapped, extractor


# ------------------------------------------------------------------ 等价性闸门
def self_check(variant: str, aggregation: str, image_size: int, batch_size: int = 32) -> dict:
    """⭐ 阶段 2 的硬闸门：包装后必须与原提取器逐位（或在浮点误差内）一致。

    为什么这是硬闸门：后面每一档优化（compile / ONNX / TRT / INT8）都会拿特征去跑
    VPR 协议、和 FP32 参考比余弦相似度。如果包装本身就引入了差异，那些差异里就
    混进了"包装写错了"这个变量，再也分不清是量化的代价还是我的 bug。

    用全部 300 张真实图像（而不是随机张量）比：随机输入可能碰不到某些数值区间。
    """
    from torch.utils.data import DataLoader
    from src.dataset import ImageFolderDataset, get_default_transform

    wrapped, extractor = build_descriptor(variant, aggregation, image_size)
    report: dict = {"variant": variant, "aggregation": aggregation, "image_size": image_size}

    for split in ("database", "query"):
        ds = ImageFolderDataset(REPO / "data" / "gardens_point" / split,
                                transform=get_default_transform(image_size))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
        ref_all, new_all = [], []
        with torch.no_grad():
            for images, _ in loader:
                images = images.cuda()
                ref_all.append(extractor(images).float().cpu())
                new_all.append(wrapped(images).float().cpu())
        ref, new = torch.cat(ref_all), torch.cat(new_all)
        cos = F.cosine_similarity(ref, new, dim=1)
        report[split] = {
            "n": len(ref),
            "max_abs_diff": float((ref - new).abs().max()),
            "min_cosine": float(cos.min()),
            "bit_identical": bool(torch.equal(ref, new)),
        }

    report["passed"] = all(
        report[s]["max_abs_diff"] < 1e-5 and report[s]["min_cosine"] > 1 - 1e-6
        for s in ("database", "query")
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--variant", default="dinov2_vits14")
    parser.add_argument("--aggregation", default="mean")
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    if not args.self_check:
        parser.print_help()
        return

    torch.backends.cuda.matmul.allow_tf32 = False     # 纯 FP32 比对，排除 TF32 干扰
    torch.backends.cudnn.allow_tf32 = False

    print(f"等价性闸门：VPRDescriptor vs DINOv2FeatureExtractor  "
          f"({args.variant}, {args.aggregation}, {args.image_size}px, FP32)")
    report = self_check(args.variant, args.aggregation, args.image_size)
    for split in ("database", "query"):
        r = report[split]
        print(f"  {split:<9} n={r['n']:<4} max|Δ|={r['max_abs_diff']:.3e}  "
              f"min cos={r['min_cosine']:.9f}  逐位相同={r['bit_identical']}")
    print("结论：", "✅ 通过" if report["passed"] else "❌ 未通过 —— 禁止进入后续阶段")

    out = RESULTS / f"wrapper_selfcheck_{args.variant}_{args.aggregation}_{args.image_size}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已写入 {out}")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
