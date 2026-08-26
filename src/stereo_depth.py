"""双目视差 -> 深度图。KITTI 没有深度传感器，回环边的三维点从这里来。

为什么需要
----------
`rgbd_pose.py` 的 PnP 需要一侧的三维点。TUM 有 Kinect 深度图可以直接用，
KITTI 只有左右目灰度图，得自己算：

    视差 d = x_left - x_right          （同一个三维点在两幅图上的横坐标差）
    深度 Z = f * b / d                 （f 焦距，b 基线）

直觉：物体越近，左右眼看到的位置差越大。视差和深度成【反比】。
所以远处的深度对视差噪声极其敏感——KITTI 基线 0.54 m、焦距 719 px，
视差 1 px 的误差在 10 m 处约 0.26 m，在 50 m 处就是 6.5 m。
这就是 `KITTI_00.max_depth = 60` 的由来：再远的点不可信，直接丢掉。

⚠️ 用 SGBM 而不是 BM：
   BM 只做局部块匹配，弱纹理区域（KITTI 里大片的路面、天空）会大面积失效；
   SGBM 加了沿多个方向的动态规划平滑代价，稠密得多也稳得多。
"""

from functools import lru_cache

import cv2
import numpy as np

# KITTI 序列 00-02（Examples/Stereo/KITTI00-02.yaml）
KITTI_FX = 718.856
KITTI_BASELINE = 0.53716          # 米。ORB-SLAM3 yaml 里的 Stereo.b

_MATCHER = None


def _get_matcher(num_disparities: int = 128, block_size: int = 5):
    """SGBM 参数取自 OpenCV 对 KITTI 这类户外场景的常用设置。"""
    global _MATCHER
    if _MATCHER is None:
        _MATCHER = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities,        # 必须是 16 的倍数
            blockSize=block_size,
            P1=8 * block_size ** 2,                # 视差变化 1 的惩罚
            P2=32 * block_size ** 2,               # 视差变化 >1 的惩罚（要更大）
            disp12MaxDiff=1,                       # 左右一致性检查
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,   # 比全 8 方向快，质量接近
        )
    return _MATCHER


@lru_cache(maxsize=400)
def depth_from_stereo(
    left_path: str, right_path: str,
    fx: float = KITTI_FX, baseline: float = KITTI_BASELINE,
) -> np.ndarray:
    """由左右目图像算深度图（单位：米）。无效处为 0。

    带缓存——同一个关键帧会被多个回环候选反复用到。
    """
    left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    if left is None or right is None:
        raise ValueError(f"读不到图像: {left_path} / {right_path}")

    # SGBM 输出的是 定点数视差 x16，要除回去
    disparity = _get_matcher().compute(left, right).astype(np.float32) / 16.0

    depth = np.zeros_like(disparity)
    valid = disparity > 0.5                       # 视差过小 = 太远 = 不可信
    depth[valid] = fx * baseline / disparity[valid]
    return depth
