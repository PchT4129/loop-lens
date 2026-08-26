"""SE(3) 李群工具：位姿图优化的地基。

为什么需要它
------------
位姿不能直接当向量做加减——旋转矩阵的集合 SO(3) 不是向量空间，
两个旋转矩阵相加的结果一般不再是旋转矩阵。

解决办法是在【李代数】上做优化：
    - 李群 SE(3)：4x4 变换矩阵，有约束（R 正交、det=1）
    - 李代数 se(3)：6 维向量 ξ = [ρ, φ]，【无约束】，可以自由加减
    - exp / log 在两者之间来回

优化时把增量 δ 放在李代数里算，再用 exp 映射回群上做乘法更新：
    X <- X · exp(δ^)          （右乘扰动）
这样更新后的 X 【永远】还是合法的位姿。
"""

import numpy as np

_EPS = 1e-10


def hat(v: np.ndarray) -> np.ndarray:
    """3 维向量 -> 反对称矩阵。满足 hat(a) @ b == np.cross(a, b)。"""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def vee(M: np.ndarray) -> np.ndarray:
    """hat 的逆。"""
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


def so3_exp(phi: np.ndarray) -> np.ndarray:
    """so(3) -> SO(3)，罗德里格斯公式。"""
    theta = np.linalg.norm(phi)
    K = hat(phi)
    if theta < _EPS:
        # 小角度用二阶泰勒，避免 0/0
        return np.eye(3) + K + 0.5 * K @ K
    return (
        np.eye(3)
        + (np.sin(theta) / theta) * K
        + ((1.0 - np.cos(theta)) / theta**2) * (K @ K)
    )


def so3_log(R: np.ndarray) -> np.ndarray:
    """SO(3) -> so(3)。"""
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))          # 数值上可能越界
    theta = np.arccos(c)
    if theta < _EPS:
        return vee(R - np.eye(3)) * 0.5
    if abs(theta - np.pi) < 1e-5:
        # theta 接近 pi 时 sin(theta)->0，改从 R+I 提取轴
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.maximum(np.diag(A), 0.0))
        k = int(np.argmax(axis))
        if axis[k] > _EPS:
            axis = A[:, k] / axis[k]
        axis = axis / (np.linalg.norm(axis) + _EPS)
        # 符号由 R 的反对称部分决定
        if np.dot(vee(R - R.T), axis) < 0:
            axis = -axis
        return axis * theta
    return vee(R - R.T) * (theta / (2.0 * np.sin(theta)))


def _V(phi: np.ndarray) -> np.ndarray:
    """SE(3) exp 里平移部分的左雅可比 V。"""
    theta = np.linalg.norm(phi)
    K = hat(phi)
    if theta < _EPS:
        return np.eye(3) + 0.5 * K + (1.0 / 6.0) * K @ K
    return (
        np.eye(3)
        + ((1.0 - np.cos(theta)) / theta**2) * K
        + ((theta - np.sin(theta)) / theta**3) * (K @ K)
    )


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """se(3) 6 维向量 -> SE(3) 4x4 矩阵。xi = [rho(3), phi(3)]。"""
    rho, phi = xi[:3], xi[3:]
    T = np.eye(4)
    T[:3, :3] = so3_exp(phi)
    T[:3, 3] = _V(phi) @ rho
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    """SE(3) -> se(3)。"""
    phi = so3_log(T[:3, :3])
    rho = np.linalg.solve(_V(phi), T[:3, 3])
    return np.concatenate([rho, phi])


def se3_inv(T: np.ndarray) -> np.ndarray:
    """位姿求逆。比 np.linalg.inv 快且数值更稳（利用 R 正交）。"""
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def se3_adjoint(T: np.ndarray) -> np.ndarray:
    """伴随矩阵 Ad(T)，把一个坐标系下的 se(3) 增量搬到另一个坐标系。

    满足 T · exp(xi^) · T^-1 == exp((Ad(T) xi)^)
    """
    R, t = T[:3, :3], T[:3, 3]
    A = np.zeros((6, 6))
    A[:3, :3] = R
    A[:3, 3:] = hat(t) @ R
    A[3:, 3:] = R
    return A


def se3_ad(xi: np.ndarray) -> np.ndarray:
    """李代数上的 ad 算子（伴随的微分），用于右雅可比的一阶近似。"""
    rho, phi = xi[:3], xi[3:]
    A = np.zeros((6, 6))
    A[:3, :3] = hat(phi)
    A[:3, 3:] = hat(rho)
    A[3:, 3:] = hat(phi)
    return A


def right_jacobian_inv(xi: np.ndarray) -> np.ndarray:
    """右雅可比的逆，一阶近似 J_r^-1 ≈ I + 0.5 * ad(xi)。

    位姿图收敛时残差 xi -> 0，这个近似在收敛邻域内足够准，
    且比完整公式（含 Q 矩阵）便宜得多。
    """
    return np.eye(6) + 0.5 * se3_ad(xi)


def pose_from_quat(t: np.ndarray, q: np.ndarray) -> np.ndarray:
    """由平移 + 四元数 (qx,qy,qz,qw) 构造 4x4 位姿。TUM 格式就是这个顺序。"""
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    T = np.eye(4)
    T[:3, :3] = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    T[:3, 3] = t
    return T


def quat_from_pose(T: np.ndarray) -> np.ndarray:
    """4x4 位姿 -> 四元数 (qx,qy,qz,qw)。"""
    R = T[:3, :3]
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x, y, z = 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x, y, z = (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x, y, z = (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return np.array([x, y, z, w])
