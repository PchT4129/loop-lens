"""度量学习损失。

除了 PyTorch 自带的 TripletMarginLoss，这里实现 InfoNCE——它把度量学习
从"判断题"（a 离 p 是不是比离 n 更近）变成"选择题"（N+1 个候选里哪个是正样本）。

InfoNCE 相对 triplet 的三个优势：
  1. 一步用上 batch 内【全部】负样本，而不是随机挑 1 个
  2. softmax 永远有梯度；triplet 满足 margin 后 loss 归零
     （实测本项目的随机负样本采样有 24% 的三元组零梯度）
  3. 【自带难样本加权】——每个负样本收到的梯度恰好等于它的 softmax 概率，
     相似度越高权重越大。不需要手写 semi-hard mining
"""

import torch
import torch.nn.functional as F


def build_invalid_negative_mask(
    anchor_index: torch.Tensor,
    positive_index: torch.Tensor,
    negative_gap: int,
) -> torch.Tensor:
    """标出 batch 内哪些 (anchor, positive) 配对【不能当负样本用】。

    InfoNCE 默认把 batch 内其他所有 positive 都当作负样本。但在 VPR 里这是
    危险的：batch 里可能恰好抽到两个帧号只差 5 的样本——它们物理上很近、
    看起来很像，却会被当负样本推开。

    ⚠️ 而且 InfoNCE 给难样本的权重最大，**这种矛盾信号反而会被放大**——
    因为它们正是相似度最高的那些"负样本"。

    所以沿用 triplet 的 negative_gap 语义：只有帧号差 >= negative_gap 的
    才算合格负样本，其余（包括"其实也是正样本"和灰色地带）全部屏蔽。

    Args:
        anchor_index:   [B]，每个 anchor 的帧号
        positive_index: [B]，每个 positive 在 database 里的帧号
        negative_gap:   帧号差达到多少才算"明确不同的地点"

    Returns:
        [B, B] 的 bool 张量，True 表示该位置应从 softmax 分母中排除。
        对角线恒为 False（那是正样本本身，必须保留）。
    """
    delta = (anchor_index[:, None] - positive_index[None, :]).abs()
    invalid = delta < negative_gap
    invalid.fill_diagonal_(False)
    return invalid


def info_nce_loss(
    anchor_features: torch.Tensor,
    positive_features: torch.Tensor,
    tau: float = 0.07,
    invalid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """InfoNCE / NT-Xent 损失。

        L = -log  exp(sim(a_i, p_i)/tau) / sum_j exp(sim(a_i, p_j)/tau)

    本质就是 softmax + 交叉熵：把相似度除以温度当 logits，正样本当"第 i 类"。

    Args:
        anchor_features:   [B, D]，已 L2 归一化
        positive_features: [B, D]，已 L2 归一化。第 i 行是第 i 个 anchor 的正样本
        tau: 温度。**难样本聚焦程度的旋钮**——
             tau 越小 softmax 越尖锐、越只盯最难的负样本（趋近 hardest mining），
             tau 越大越一视同仁。SimCLR 用 0.1，MoCo 用 0.07。
        invalid_mask: [B, B] bool，True 的位置从分母中排除（见上面那个函数）

    Returns:
        标量 loss
    """
    # 特征已归一化，点积即余弦
    logits = anchor_features @ positive_features.T / tau

    if invalid_mask is not None:
        logits = logits.masked_fill(invalid_mask, float("-inf"))

    labels = torch.arange(len(logits), device=logits.device)
    return F.cross_entropy(logits, labels)


@torch.no_grad()
def infonce_diagnostics(
    anchor_features: torch.Tensor,
    positive_features: torch.Tensor,
    tau: float,
    invalid_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """训练时的诊断信息，用来验证 InfoNCE 确实在按预期工作。

    返回：
        valid_negatives: 平均每个 anchor 有多少个合格负样本
                         （batch 太小时这个数会很低，是个真实的限制）
        top1_acc:        正样本被排到第一的比例
        effective_negs:  梯度权重 > 1% 的负样本平均个数
                         （对比 triplet 的"1 个负样本、其中 24% 零梯度"）
    """
    logits = anchor_features @ positive_features.T / tau
    if invalid_mask is not None:
        logits = logits.masked_fill(invalid_mask, float("-inf"))

    batch = len(logits)
    labels = torch.arange(batch, device=logits.device)

    if invalid_mask is None:
        valid = torch.full((batch,), batch - 1.0, device=logits.device)
    else:
        valid = (~invalid_mask).float().sum(dim=1) - 1.0  # 减掉对角线的正样本

    probs = logits.softmax(dim=1)
    off_diagonal = probs.clone()
    off_diagonal[labels, labels] = 0.0

    return {
        "valid_negatives": valid.mean().item(),
        "top1_acc": (logits.argmax(dim=1) == labels).float().mean().item(),
        "effective_negs": (off_diagonal > 0.01).float().sum(dim=1).mean().item(),
    }
