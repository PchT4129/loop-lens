"""学习型局部特征的几何验证：DISK + LightGlue + RANSAC。

为什么需要它
------------
`geometric_verification.py` 用的是 ORB（手工设计的二值描述子）。在本项目上
实测出一个明确的失效边界：

    Query 域                    正确配对内点   错误配对内点   可分性
    day_right  (仅视角变化)         134.7          26.1        100%
    night_right(昼夜变化)            28.7          27.0      40~53%   ← 等同随机

原因是 ORB 依赖图像梯度构造描述子，昼夜之间梯度结构本身就变了，
能通过 ratio test 的匹配大多是虚假的。而基础矩阵只有 7 个自由度，
在几十个随机匹配里凑出一个"看似成立"的模型并不困难——
**不是 RANSAC 失效了，是喂给它的匹配本身就是噪声。**

所以本模块换成学习型局部特征，验证"换特征能否救回判别力"这个命题。

⚠️ 设计约束：判据必须和 ORB 路径【完全一致】
--------------------------------------------
都是"匹配 -> cv2.findFundamentalMat + RANSAC -> 数内点"。
只有特征提取和匹配这两步不同，否则对比不公平。
"""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch

# 和 geometric_verification.py 保持一致的 RANSAC 判据
DEFAULT_RANSAC_THRESHOLD = 3.0
DEFAULT_N_FEATURES = 2048

_MODELS: dict = {}


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_models():
    """惰性加载 DISK 和 LightGlue（首次调用时才下权重）。"""
    if "disk" not in _MODELS:
        import kornia.feature as KF

        dev = _device()
        _MODELS["disk"] = KF.DISK.from_pretrained("depth").to(dev).eval()
        # LightGlue 的 'disk' 配置对应 DISK 的 128 维描述子
        _MODELS["matcher"] = KF.LightGlueMatcher("disk").to(dev).eval()
    return _MODELS["disk"], _MODELS["matcher"]


@lru_cache(maxsize=900)
def _detect(image_path: str, n_features: int):
    """提取 DISK 关键点与描述子。带缓存——同一张 database 图会被反复用到。

    与 ORB 版本对齐：同样返回 (points[N,2], descriptors) 或 (empty, None)。
    """
    import kornia as K

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    dev = _device()
    tensor = K.image.image_to_tensor(image, False).float() / 255.0
    tensor = K.color.bgr_to_rgb(tensor).to(dev)

    disk, _ = _get_models()
    with torch.no_grad():
        features = disk(
            tensor, n=n_features, window_size=5,
            score_threshold=0.0, pad_if_not_divisible=True,
        )[0]

    keypoints = features.keypoints          # [N, 2]
    descriptors = features.descriptors      # [N, 128]

    if len(keypoints) < 8:
        return np.empty((0, 2), dtype=np.float32), None

    return keypoints.cpu().numpy().astype(np.float32), descriptors


def verify_pair(
    query_path: str | Path,
    database_path: str | Path,
    n_features: int = DEFAULT_N_FEATURES,
    ransac_threshold: float = DEFAULT_RANSAC_THRESHOLD,
    **_ignored,
) -> dict:
    """对一个 (query, candidate) 配对做几何验证。

    接口与 `geometric_verification.verify_pair` 完全相同，
    所以 `evaluate.py` 只需切换实现即可。
    """
    empty = {
        "num_keypoints_query": 0,
        "num_keypoints_database": 0,
        "num_matches": 0,
        "num_inliers": 0,
        "inlier_ratio": 0.0,
        "fundamental_matrix": None,
    }

    q_pts, q_desc = _detect(str(query_path), n_features)
    d_pts, d_desc = _detect(str(database_path), n_features)

    empty["num_keypoints_query"] = len(q_pts)
    empty["num_keypoints_database"] = len(d_pts)

    if q_desc is None or d_desc is None:
        return empty

    import kornia.feature as KF

    _, matcher = _get_models()
    # LightGlue 需要 laf（局部仿射框架）来提供关键点位置信息
    dev = _device()
    q_t = torch.from_numpy(q_pts).to(dev)
    d_t = torch.from_numpy(d_pts).to(dev)
    lafs_q = KF.laf_from_center_scale_ori(q_t[None], torch.ones(1, len(q_t), 1, 1, device=dev))
    lafs_d = KF.laf_from_center_scale_ori(d_t[None], torch.ones(1, len(d_t), 1, 1, device=dev))

    with torch.no_grad():
        _, idxs = matcher(q_desc, d_desc, lafs_q, lafs_d)

    num_matches = len(idxs)
    empty["num_matches"] = num_matches

    if num_matches < 8:
        return empty

    idxs = idxs.cpu().numpy()
    src = q_pts[idxs[:, 0]]
    dst = d_pts[idxs[:, 1]]

    # ⚠️ 这一步必须和 ORB 路径完全一致，否则对比不公平
    fundamental_matrix, mask = cv2.findFundamentalMat(
        src, dst, cv2.FM_RANSAC, ransac_threshold, 0.99
    )
    if mask is None:
        return empty

    num_inliers = int(mask.sum())
    return {
        "num_keypoints_query": len(q_pts),
        "num_keypoints_database": len(d_pts),
        "num_matches": num_matches,
        "num_inliers": num_inliers,
        "inlier_ratio": num_inliers / max(num_matches, 1),
        "fundamental_matrix": fundamental_matrix.tolist(),
    }


def verify_candidates(query_path, candidate_paths: list[str], **kwargs) -> list[dict]:
    """对一个 query 的全部候选做几何验证，返回完整验证统计。"""
    return [
        verify_pair(query_path, candidate, **kwargs)
        for candidate in candidate_paths
    ]


def match_pair(
    query_path: str | Path,
    database_path: str | Path,
    n_features: int = DEFAULT_N_FEATURES,
) -> tuple[np.ndarray, np.ndarray]:
    """返回一一对应的匹配像素坐标 (uv_query [N,2], uv_database [N,2])。

    `verify_pair` 只回报内点数；位姿图需要的是【对应关系本身】，
    好把它们喂给 PnP 求相对位姿（见 `rgbd_pose.py`）。
    两者共用同一套 DISK + LightGlue 与同一个特征缓存。
    """
    import kornia.feature as KF

    q_pts, q_desc = _detect(str(query_path), n_features)
    d_pts, d_desc = _detect(str(database_path), n_features)

    empty = (np.empty((0, 2), np.float32), np.empty((0, 2), np.float32))
    if q_desc is None or d_desc is None:
        return empty

    dev = _device()
    _, matcher = _get_models()
    q_t = torch.from_numpy(q_pts).to(dev)
    d_t = torch.from_numpy(d_pts).to(dev)
    lafs_q = KF.laf_from_center_scale_ori(q_t[None], torch.ones(1, len(q_t), 1, 1, device=dev))
    lafs_d = KF.laf_from_center_scale_ori(d_t[None], torch.ones(1, len(d_t), 1, 1, device=dev))

    with torch.no_grad():
        _, idxs = matcher(q_desc, d_desc, lafs_q, lafs_d)

    if len(idxs) == 0:
        return empty

    idxs = idxs.cpu().numpy()
    return q_pts[idxs[:, 0]], d_pts[idxs[:, 1]]
