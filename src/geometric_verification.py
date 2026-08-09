"""几何验证：用局部特征 + RANSAC 确认检索候选是否真的是同一地点。

在 SLAM 的回环检测流水线里，全局描述子检索只是第一级：

    ① 候选检索（CNN 全局描述子）  ← 快，宽松，追求高 recall
    ② 几何验证（本模块）           ← 慢，严格，剔除误检
    ③ 一致性检查（连续多帧确认）
    ④ 位姿图优化

全局描述子只能说"这两张图像是同一个地方"，但【说不出图像里哪个点对应哪个点】。
几何验证补的正是这一步：如果两张图真的拍的是同一个三维场景，那么它们的
局部特征之间必然存在一个【一致的几何关系】（对极几何约束）。

RANSAC 的作用就是在一堆含噪声的匹配中，找出符合同一几何模型的最大子集：
    1. 随机抽最小点集（基础矩阵需 8 对）
    2. 用它算一个候选模型
    3. 拿这个模型检验【所有】匹配，统计内点数
    4. 重复多次，取内点最多的模型
正确的匹配都服从同一个几何关系（来自同一个真实三维场景），
而错误匹配是随机散落的——所以"能让最多点对同时满足的模型"就是真的。

⚠️ 实测结论：ORB 在昼夜配对上【完全失效】
------------------------------------------
在本数据集上做过参数扫描（15 个采样点，正确配对 vs 随机错误配对）：

    Query 域                    正确配对内点   错误配对内点   可分性
    day_right  (仅视角变化)         134.7          26.1        100%
    night_right(昼夜变化)            28.7          27.0      40~53%

也就是说：在纯视角变化下几何验证判别力完美；但在昼夜配对上，
正确与错误配对的内点数分布几乎完全重合，**判别力等同于随机猜测**。

原因是 ORB 依赖图像梯度构造二值描述子，昼夜之间梯度结构本身就变了，
能通过 ratio test 的匹配大多是虚假的；而基础矩阵有 7 个自由度，
在几十个随机匹配里凑出一个"看似成立"的模型并不困难。

这个结果的意义：它从一个独立的角度验证了整个项目的前提——
**传统手工特征扛不住光照剧变，这正是需要学习型描述子的原因**。
下一步应该换成 SuperPoint + LightGlue 这类学习型局部特征。

工程结论：几何验证对同域（白天-白天）配对有效，对跨昼夜配对不应启用，
否则会把本来正确的检索结果重排乱。
"""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

# ORB 特征点数量。参数扫描显示 4000 明显优于 2000：
# day_right 上正确/错误配对内点数 134.7 vs 26.1，可分性从 87% 提到 100%。
DEFAULT_N_FEATURES = 4000

# Lowe ratio test 阈值。越小越严格，保留的匹配越少但越可靠。
# ORB 是二值描述子，距离分布比 SIFT 更集中，0.75 太严会导致匹配数只剩个位数，
# 此时基础矩阵 RANSAC 极易在噪声上凑出虚假模型。实测 0.90 最优。
DEFAULT_RATIO = 0.90

# RANSAC 判定内点的重投影误差阈值（像素）
DEFAULT_RANSAC_THRESHOLD = 3.0


@lru_cache(maxsize=512)
def _detect(image_path: str, n_features: int):
    """提取 ORB 关键点与描述子。带缓存——同一张 database 图会被反复用到。"""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(image, None)

    if descriptors is None or len(keypoints) < 8:
        return np.empty((0, 2), dtype=np.float32), None

    points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    return points, descriptors


def verify_pair(
    query_path: str | Path,
    database_path: str | Path,
    n_features: int = DEFAULT_N_FEATURES,
    ratio: float = DEFAULT_RATIO,
    ransac_threshold: float = DEFAULT_RANSAC_THRESHOLD,
) -> dict:
    """对一个 (query, candidate) 配对做几何验证。

    Returns:
        dict，含：
            num_keypoints_query / num_keypoints_database: 各自检出的特征点数
            num_matches:  通过 ratio test 的匹配数
            num_inliers:  RANSAC 后符合几何模型的内点数  ← 主判据
            inlier_ratio: 内点数 / 匹配数
    """
    empty = {
        "num_keypoints_query": 0,
        "num_keypoints_database": 0,
        "num_matches": 0,
        "num_inliers": 0,
        "inlier_ratio": 0.0,
    }

    query_points, query_desc = _detect(str(query_path), n_features)
    db_points, db_desc = _detect(str(database_path), n_features)

    empty["num_keypoints_query"] = len(query_points)
    empty["num_keypoints_database"] = len(db_points)

    if query_desc is None or db_desc is None:
        return empty

    # ORB 是二值描述子，用汉明距离。knn=2 是为了做 ratio test。
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = matcher.knnMatch(query_desc, db_desc, k=2)

    # Lowe ratio test：最近邻明显优于次近邻才算可靠匹配，
    # 能滤掉大量重复纹理造成的歧义匹配。
    good = [
        m
        for pair in knn_matches
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < ratio * n.distance
    ]

    empty["num_matches"] = len(good)

    # 求基础矩阵至少需要 8 对点
    if len(good) < 8:
        return empty

    src = np.array([query_points[m.queryIdx] for m in good], dtype=np.float32)
    dst = np.array([db_points[m.trainIdx] for m in good], dtype=np.float32)

    # 用基础矩阵而非本质矩阵：本质矩阵需要相机内参，而这个数据集没有标定信息。
    # 基础矩阵同样能施加对极约束，足以判断"是不是同一个场景"。
    _, mask = cv2.findFundamentalMat(
        src, dst, cv2.FM_RANSAC, ransac_threshold, 0.99
    )

    if mask is None:
        return empty

    num_inliers = int(mask.sum())
    return {
        "num_keypoints_query": len(query_points),
        "num_keypoints_database": len(db_points),
        "num_matches": len(good),
        "num_inliers": num_inliers,
        "inlier_ratio": num_inliers / max(len(good), 1),
    }


def verify_candidates(
    query_path: str | Path,
    candidate_paths: list[str],
    **kwargs,
) -> list[int]:
    """对一个 query 的全部候选做几何验证，返回每个候选的内点数。"""
    return [
        verify_pair(query_path, candidate, **kwargs)["num_inliers"]
        for candidate in candidate_paths
    ]


def rerank_by_inliers(
    candidate_paths: list[str],
    inliers: list[int],
) -> list[int]:
    """按内点数从多到少重排，返回排序后的【原下标】。

    用稳定排序：内点数相同时保持原有的 CNN 相似度排序。
    这一点很重要——ORB 在昼夜配对上经常大面积返回 0 内点，
    此时应该退化回原始检索顺序，而不是随机打乱。
    """
    order = sorted(range(len(inliers)), key=lambda i: -inliers[i])
    return order
