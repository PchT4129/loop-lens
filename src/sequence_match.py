"""序列匹配：利用轨迹的时序连续性对检索结果重排。

动机
----
单帧匹配把每一帧当作独立事件，但机器人是【连续运动】的，这个约束被白白扔掉了。

核心假设：
    如果 query 第 t 帧匹配 database 第 j 帧，
    那么 query 第 t+1 帧应该匹配 database 第 j+1 帧。

所以正确的匹配会在相似度矩阵上沿对角线形成一条【连续的高分带】，
而误匹配是孤立的亮点。沿对角线做窗口聚合就能把前者放大、后者压制。

这是 SeqSLAM (Milford & Wyeth, ICRA 2012) 的简化版：
真正的 SeqSLAM 还会在多个"速度"（对角线斜率）上搜索，并做局部对比度归一化。

前提与局限
----------
1. 假设两次采集的行进【速度比为 1】——query 第 t 帧正好对应 database 第 j+t 帧。
   Gardens Point 满足（同一路线步行采集）。速度不一致时需要多速度搜索。
2. 需要 query 是【连续轨迹】。随机单张图的定位场景用不了。
3. 引入延迟：因果模式下要"已经走过 window 帧"才有完整窗口。
   机器人原地转向、倒着走、走岔路时序列假设会失效。
"""

import torch


def sequence_rerank(
    similarities: torch.Tensor,
    window: int,
    causal: bool = False,
) -> torch.Tensor:
    """沿相似度矩阵的对角线方向做窗口聚合。

    Args:
        similarities: [M, N] 的相似度矩阵，M 个 query × N 个 database。
            **要求 query 按时间顺序排列，database 也按顺序排列。**
        window: 窗口半径。causal=False 时用 [-window, +window]（共 2*window+1 帧），
            causal=True 时只用 [-window, 0]（共 window+1 帧）。
        causal: 是否只使用【过去】的帧。在线 SLAM 必须为 True，
            因为未来帧还没被观测到。

    Returns:
        [M, N] 的重排后分数矩阵。语义仍是"越大越相似"，可以直接喂给 topk。

    实现要点：
        acc[i, j] = mean over d of similarities[i+d, j+d]

        源块必须相对目标块偏移 d —— 如果源和目标用同一个切片，
        就退化成"把同一个矩阵重复加几次"，排序完全不变。
    """
    if window <= 0:
        return similarities

    num_queries, num_database = similarities.shape

    accumulated = torch.zeros_like(similarities)
    counts = torch.zeros_like(similarities)

    offsets = range(-window, 1) if causal else range(-window, window + 1)

    for offset in offsets:
        # 目标块：所有满足 0 <= i < M 且 0 <= i+offset < M 的 i
        i_start, i_end = max(0, -offset), min(num_queries, num_queries - offset)
        j_start, j_end = max(0, -offset), min(num_database, num_database - offset)

        if i_end <= i_start or j_end <= j_start:
            continue

        accumulated[i_start:i_end, j_start:j_end] += similarities[
            i_start + offset:i_end + offset,
            j_start + offset:j_end + offset,
        ]
        counts[i_start:i_end, j_start:j_end] += 1

    # 边界处窗口不完整，除以实际参与的项数而不是固定窗口大小
    return accumulated / counts.clamp(min=1)
