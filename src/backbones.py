"""Backbone 工厂：统一 ResNet18 和 DINOv2 的接口。

所有 backbone 都遵守同一个契约：
    输入 [B, 3, H, W]  ->  输出 [B, D]，且已 L2 归一化

这样 `{features, paths}` 那个下游契约完全不用变，
`retrieve.py` / `evaluate.py` / `visualize.py` 一行都不用改。

为什么要试 DINOv2
-----------------
本项目量化出的核心痛点是【跨域泛化】：昼夜变化让 Recall@1 从 96% 掉到 51%，
而在小数据上微调 ResNet18 会过拟合（泛化间隙 34.8 点）。

DINOv2 是自监督训练的 ViT，AnyLoc (RA-L'23) 发现直接拿它的特征做 VPR，
【零训练】就能在跨域场景上超过专门训练的模型。所以这里要验证的命题是：

    对只有 70 个训练样本的场景，"换一个泛化能力本来就强的特征"
    是不是比"在小数据上微调"更对症？
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import GeMPooling, ResNet18FeatureExtractor


class DINOv2FeatureExtractor(nn.Module):
    """DINOv2 全局描述子。

    Args:
        variant: "dinov2_vits14"(384维) / "dinov2_vitb14"(768维) / "dinov2_vitl14"(1024维)
        aggregation: 怎么把 ViT 的输出变成一个向量
            "cls"      —— 直接用 CLS token。ViT 自带的全局表示
            "gem"      —— 对 patch token 做 GeM 池化
            "mean"     —— 对 patch token 做平均（等价于 GAP）
            "cls+gem"  —— 两者拼接

            AnyLoc 的结论是【patch token 聚合优于 CLS】，因为 CLS 是为
            分类/自监督目标训练的，patch token 保留了更多局部结构信息。
        gem_p: aggregation 含 gem 时的 p 值

    ⚠️ patch size = 14，所以输入的 H、W 必须是 14 的倍数。
       224 = 16×14 ✓，但 384 = 27.4×14 ✗（会被 DINOv2 内部裁掉一部分）。
       用 448 = 32×14 作为"高分辨率"档。
    """

    PATCH_SIZE = 14

    def __init__(
        self,
        variant: str = "dinov2_vits14",
        aggregation: str = "gem",
        gem_p: float = 3.0,
    ):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", variant, verbose=False)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)          # 零训练：整个 backbone 冻结

        self.aggregation = aggregation
        self.gem = GeMPooling(p=gem_p, learnable=False) if "gem" in aggregation else None

        embed_dim = self.model.embed_dim
        self.feature_dim = embed_dim * 2 if aggregation == "cls+gem" else embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        if h % self.PATCH_SIZE or w % self.PATCH_SIZE:
            raise ValueError(
                f"DINOv2 的 patch size 是 {self.PATCH_SIZE}，输入尺寸 {h}x{w} "
                f"不是它的倍数。建议用 224 (=16x14) 或 448 (=32x14)"
            )

        out = self.model.forward_features(x)
        cls = out["x_norm_clstoken"]                    # [B, D]
        patches = out["x_norm_patchtokens"]             # [B, N, D]

        if self.aggregation == "cls":
            features = cls
        elif self.aggregation == "mean":
            features = patches.mean(dim=1)
        elif self.aggregation == "gem":
            features = self._gem_pool(patches)
        elif self.aggregation == "cls+gem":
            features = torch.cat([cls, self._gem_pool(patches)], dim=1)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation!r}")

        return F.normalize(features, p=2, dim=1)

    def _gem_pool(self, patches: torch.Tensor) -> torch.Tensor:
        # GeMPooling 期望 [B,C,H,W]；把 [B,N,D] 转成 [B,D,N,1] 复用同一份实现
        x = patches.transpose(1, 2).unsqueeze(-1)
        return self.gem(x).flatten(1)


def build_backbone(
    name: str = "resnet18",
    pretrained: bool = True,
    pooling: str = "gap",
    gem_p: float = 3.0,
    dinov2_aggregation: str = "gem",
):
    """统一入口。

    name:
        "resnet18"        —— 项目原有的 CNN backbone
        "dinov2_vits14"   —— DINOv2 ViT-S/14，384 维，零训练
        "dinov2_vitb14"   —— DINOv2 ViT-B/14，768 维
    """
    if name == "resnet18":
        return ResNet18FeatureExtractor(
            pretrained=pretrained, pooling=pooling, gem_p=gem_p
        )
    if name.startswith("dinov2"):
        return DINOv2FeatureExtractor(
            variant=name, aggregation=dinov2_aggregation, gem_p=gem_p
        )
    raise ValueError(f"Unknown backbone: {name!r}")
