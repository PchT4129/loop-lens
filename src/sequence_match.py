"""序列匹配：利用轨迹的时序连续性对检索结果重排。

动机
----
单帧匹配把每一帧当作独立事件，但机器人是【连续运动】的，这个约束被白白扔掉了。

核心假设：
    如果 query 第 t 帧匹配 database 第 j 帧，
    那么 query 第 t+1 帧应该匹配 database 第 j+v 帧。

其中 v 是【速度比】——两次采集的行进速度之比。正确的匹配会在相似度矩阵上
沿一条斜率为 v 的直线形成连续高分带，而误匹配是孤立的亮点。

这是 SeqSLAM (Milford & Wyeth, ICRA 2012) 的简化版：真正的 SeqSLAM 还会做
局部对比度归一化，这里省略了。

关于速度比
----------
早期版本固定假设 v = 1（两次采集速度完全一致）。Gardens Point 恰好满足，
但真实场景里机器人不会每次都用同样的速度走。`velocities` 参数支持在多个速度
上搜索并取最优——代价是计算量线性增长。
"""

import torch


def _aggregate_along_line(
    similarities: torch.Tensor,
    window: int,
    velocity: float,
    causal: bool,
) -> torch.Tensor:
    """沿斜率为 velocity 的直线做窗口聚合。

    acc[i, j] = mean over d of similarities[i + d, j + round(velocity * d)]

    实现要点：源块必须相对目标块偏移。如果源和目标用同一个切片，
    就退化成"把同一个矩阵重复加几次"，排序完全不变。
    """
    num_queries, num_database = similarities.shape

    accumulated = torch.zeros_like(similarities)
    counts = torch.zeros_like(similarities)

    offsets = range(-window, 1) if causal else range(-window, window + 1)

    for offset in offsets:
        row_shift = offset
        col_shift = int(round(velocity * offset))

        # 目标块：所有满足 0 <= i < M 且 0 <= i+row_shift < M 的 i（列同理）
        i_start = max(0, -row_shift)
        i_end = min(num_queries, num_queries - row_shift)
        j_start = max(0, -col_shift)
        j_end = min(num_database, num_database - col_shift)

        if i_end <= i_start or j_end <= j_start:
            continue

        accumulated[i_start:i_end, j_start:j_end] += similarities[
            i_start + row_shift:i_end + row_shift,
            j_start + col_shift:j_end + col_shift,
        ]
        counts[i_start:i_end, j_start:j_end] += 1

    # 边界处窗口不完整，除以实际参与的项数而非固定窗口大小
    return accumulated / counts.clamp(min=1)


def sequence_rerank(
    similarities: torch.Tensor,
    window: int,
    causal: bool = False,
    velocities: tuple[float, ...] = (1.0,),
) -> torch.Tensor:
    """沿相似度矩阵的对角线方向做窗口聚合，对候选重排。

    Args:
        similarities: [M, N] 相似度矩阵，M 个 query × N 个 database。
            **要求 query 按时间顺序排列，database 也按顺序排列。**
        window: 窗口半径。causal=False 时用 [-window, +window]（共 2*window+1 帧），
            causal=True 时只用 [-window, 0]（共 window+1 帧）。
        causal: 是否只使用【过去】的帧。在线 SLAM 必须为 True，
            因为未来帧还没被观测到。
        velocities: 要搜索的速度比列表。默认 (1.0,) 即假设两次采集速度一致。
            传多个值时对每个速度各算一遍，**逐元素取最大**——
            也就是"这条轨迹在任何一个合理速度下能得到的最好分数"。

    Returns:
        [M, N] 的重排后分数矩阵，语义仍是"越大越相似"，可直接喂给 topk。

    前提与局限
    ----------
    1. 需要 query 是【连续轨迹】。随机单张图的定位场景用不了。
    2. 因果模式开头有 history warm-up；非因果模式还需要等待未来 window 帧。
    3. 机器人原地转向、倒着走、走岔路时序列假设会失效。
    """
    if window <= 0:
        return similarities

    if not velocities:
        raise ValueError("velocities 不能为空")

    best = None
    for velocity in velocities:
        scored = _aggregate_along_line(similarities, window, velocity, causal)
        best = scored if best is None else torch.maximum(best, scored)

    return best


def parse_velocities(spec: str) -> tuple[float, ...]:
    """把命令行传的速度规格解析成元组。

    支持两种写法：
        "1.0"                单一速度
        "0.8,0.9,1.0,1.1"    显式列举
    """
    return tuple(float(v) for v in spec.split(",") if v.strip())
