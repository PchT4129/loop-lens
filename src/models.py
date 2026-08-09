import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class GeMPooling(nn.Module):
    """广义均值池化 (Generalized Mean Pooling)。

        f_c = ( mean_{i,j} x[c,i,j]^p ) ^ (1/p)

    它是 GAP 和 max pooling 之间的连续插值：
        p = 1    -> 退化成 GAP（算术平均）
        p -> inf -> 退化成 max pooling
        p 居中   -> 放大高响应区域的权重

    为什么对 VPR 有用：GAP 把整张特征图求平均，天空、路面这类占据大量像素
    但没什么判别力的区域会稀释真正有用的响应。p > 1 时高响应被指数放大，
    等于自动降低了这些大面积低信息区域的权重。

    p 设为可学习参数，让网络自己决定"该多接近 max"。
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6, learnable: bool = True):
        super().__init__()
        if learnable:
            self.p = nn.Parameter(torch.tensor(float(p)))
        else:
            self.register_buffer("p", torch.tensor(float(p)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # clamp 是为了让 pow 在数值上安全（负底数的分数次幂无定义）。
        # ResNet 的 layer4 输出经过 ReLU 恒非负，所以 clamp 实际不改变任何值，
        # 这也是 p=1 时能与 GAP 数值等价的前提。
        return x.clamp(min=self.eps).pow(self.p).mean(dim=(2, 3), keepdim=True).pow(1.0 / self.p)

    def extra_repr(self) -> str:
        return f"p={float(self.p):.4f}, eps={self.eps}"


class ResNet18FeatureExtractor(nn.Module):
    """ResNet18 全局描述子：backbone -> 池化 -> flatten -> L2 归一化。

    Args:
        pretrained: 是否加载 ImageNet 预训练权重。加载 checkpoint 时可设为 False
            （权重会被完全覆盖），但注意这依赖 load_state_dict 的 strict=True。
        pooling: "gap" 用全局平均池化，"gem" 用广义均值池化。
        gem_p: GeM 的初始 p 值。
    """

    def __init__(
        self,
        pretrained: bool = True,
        pooling: str = "gap",
        gem_p: float = 3.0,
    ):
        super().__init__()

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)

        # 注意这里是 [:-2] 而不是 [:-1]：同时去掉 avgpool 和 fc，
        # 把池化拆出来单独作为一个可替换的模块。
        #
        # 这【不会破坏已有 checkpoint 的兼容性】——avgpool 没有任何参数和 buffer，
        # 所以 backbone 的 state_dict 键名与 [:-1] 版本完全相同。
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        if pooling == "gap":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif pooling == "gem":
            self.pool = GeMPooling(p=gem_p)
        else:
            raise ValueError(f"Unknown pooling: {pooling!r}, expected 'gap' or 'gem'")

        self.pooling = pooling
        self.feature_dim = resnet.fc.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)                      # [B, 512, H/32, W/32]
        features = self.pool(features)                   # [B, 512, 1, 1]
        features = torch.flatten(features, start_dim=1)  # [B, 512]
        features = F.normalize(features, p=2, dim=1)     # 点积即余弦
        return features
