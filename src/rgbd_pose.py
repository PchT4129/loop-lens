"""从一对 RGB-D 关键帧估计相对位姿 —— 回环边的来源。

这一步补上了 Gardens Point 阶段做不到的事
------------------------------------------
在 Gardens Point 上，几何验证只能数内点（`geometric_verification.py`），
因为那个数据集【没有相机内参、也没有深度】，基础矩阵转不成有物理意义的位姿。
所以当时只测到了几何验证的一半价值：能判断"是不是同一地点"，
但给不出"位姿图需要的那条边"。

TUM RGB-D 两样都有，于是这里能把另一半补上：

    ① 特征匹配（沿用几何验证那一套）
    ② 用【深度】把第 i 帧的匹配点反投影成三维点
    ③ solvePnPRansac：三维点(帧 i) <-> 二维点(帧 j)  =>  T_ji
    ④ 内点数 + 重投影误差 双重把关

⚠️ 为什么用 PnP(3D-2D) 而不是 3D-3D 配准：
   3D-3D 要求两帧的深度都可靠，噪声进来两次；
   PnP 只用一侧的深度，另一侧用像素坐标（精度高得多）。
"""

import cv2
import numpy as np

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraModel:
    """针孔内参 + 深度图的单位换算 + 有效量程。

    把这些做成参数而不是常量，是因为同一套 PnP 逻辑要跑两个数据集：
    TUM 的深度来自 Kinect（uint16 / 5000 = 米，量程几米），
    KITTI 的深度由双目视差算出（已经是米，量程几十米）。
    """

    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float = 1.0        # 深度图数值 / depth_scale = 米
    min_depth: float = 0.3
    max_depth: float = 6.0

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]])


# TUM freiburg3（configs/TUM3_*.yaml），深度来自 Kinect
TUM_FR3 = CameraModel(535.4, 539.2, 320.1, 247.6,
                      depth_scale=5000.0, min_depth=0.3, max_depth=6.0)

# KITTI 序列 00-02（Examples/Stereo/KITTI00-02.yaml），深度由双目视差算得（单位：米）
KITTI_00 = CameraModel(718.856, 718.856, 607.1928, 185.2157,
                       depth_scale=1.0, min_depth=1.0, max_depth=60.0)


def backproject(
    uv: np.ndarray, depth_img: np.ndarray, cam: CameraModel = TUM_FR3,
) -> tuple[np.ndarray, np.ndarray]:
    """把像素坐标 + 深度图 变成相机坐标系下的三维点。

    Returns:
        (points3d [M,3], valid_mask [N])  —— 深度无效的点会被剔除
    """
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    h, w = depth_img.shape
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)

    z = np.zeros(len(uv), dtype=np.float64)
    z[inside] = depth_img[v[inside], u[inside]] / cam.depth_scale
    valid = inside & (z > cam.min_depth) & (z < cam.max_depth) & np.isfinite(z)

    zz = z[valid]
    pts = np.stack([
        (uv[valid, 0] - cam.cx) * zz / cam.fx,
        (uv[valid, 1] - cam.cy) * zz / cam.fy,
        zz,
    ], axis=1)
    return pts, valid


def relative_pose(
    uv_i: np.ndarray,
    uv_j: np.ndarray,
    depth_i: np.ndarray,
    cam: CameraModel = TUM_FR3,
    min_inliers: int = 25,
    reproj_threshold: float = 3.0,
) -> dict:
    """由匹配点对 + 帧 i 的深度，估计 T_ji（把帧 i 坐标系的点变到帧 j）。

    Args:
        uv_i, uv_j: [N,2] 一一对应的匹配像素坐标
        depth_i:    帧 i 的深度图（uint16，TUM 原始格式）

    Returns:
        dict(success, T_ji 4x4, num_inliers, inlier_ratio, reproj_error)
    """
    fail = {"success": False, "T_ji": None, "num_inliers": 0,
            "inlier_ratio": 0.0, "reproj_error": float("inf")}

    if len(uv_i) < 6:
        return fail

    pts3d, valid = backproject(uv_i, depth_i, cam)
    uv_j_valid = uv_j[valid]
    if len(pts3d) < 6:
        return fail

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d.astype(np.float64),
        uv_j_valid.astype(np.float64),
        cam.K, None,
        reprojectionError=reproj_threshold,
        confidence=0.999,
        iterationsCount=500,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < min_inliers:
        return {**fail, "num_inliers": 0 if inliers is None else len(inliers)}

    idx = inliers.ravel()
    # 用全部内点做一次非线性精化（EPnP 只是初值）
    rvec, tvec = cv2.solvePnPRefineLM(
        pts3d[idx].astype(np.float64), uv_j_valid[idx].astype(np.float64),
        cam.K, None, rvec, tvec,
    )

    proj, _ = cv2.projectPoints(pts3d[idx], rvec, tvec, cam.K, None)
    reproj = float(np.linalg.norm(proj.reshape(-1, 2) - uv_j_valid[idx], axis=1).mean())

    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()

    return {
        "success": True,
        "T_ji": T,
        "num_inliers": int(len(idx)),
        "inlier_ratio": float(len(idx) / max(len(pts3d), 1)),
        "reproj_error": reproj,
    }
