"""位姿图优化：把回环约束的误差沿整条轨迹摊平。

问题
----
里程计只提供【相邻帧】之间的相对位姿。每一段都有小误差，累加起来就是漂移——
走一圈回到原点，估计出的位置却偏出去几十厘米甚至几米。

回环检测提供了一条额外的边：「第 j 帧和第 i 帧是同一个地方，它们的相对位姿是 Z_ij」。
这条边和里程计链条形成一个【环】，环上的误差必须被摊掉。

数学形式
--------
待优化量：每个关键帧的位姿 X_0 .. X_{n-1} ∈ SE(3)
每条边给一个测量 Z_ij，残差是

    e_ij = log( Z_ij^-1 · X_i^-1 · X_j )   ∈ R^6

目标：min Σ e_ij^T Ω_ij e_ij

用高斯-牛顿迭代。位姿不能直接加减（见 se3.py），所以增量放在李代数上：

    X_i <- X_i · exp(δ_i^)      （右乘扰动）

对应的雅可比（右扰动约定）：

    ∂e/∂δ_i = -J_r^-1(e) · Ad(X_j^-1 X_i)
    ∂e/∂δ_j = +J_r^-1(e)

⚠️ 规范自由度：整张图整体平移旋转不改变任何残差，所以 H 是奇异的。
   必须固定一个节点（这里固定 X_0），否则解不唯一。
"""

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from src.se3 import right_jacobian_inv, se3_adjoint, se3_exp, se3_inv, se3_log


@dataclass
class Edge:
    """一条位姿图的边。"""

    i: int
    j: int
    Z: np.ndarray                                    # 4x4，测得的 X_i^-1 X_j
    information: np.ndarray = field(default=None)    # 6x6，测量的置信度（信息矩阵）
    kind: str = "odom"                               # "odom" | "loop"

    def __post_init__(self):
        if self.information is None:
            self.information = np.eye(6)


def _residual(Xi: np.ndarray, Xj: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return se3_log(se3_inv(Z) @ se3_inv(Xi) @ Xj)


def total_error(poses: list[np.ndarray], edges: list[Edge]) -> float:
    """总的加权平方误差 Σ e^T Ω e。"""
    return float(sum(
        (e := _residual(poses[ed.i], poses[ed.j], ed.Z)) @ ed.information @ e
        for ed in edges
    ))


def optimize(
    poses: list[np.ndarray],
    edges: list[Edge],
    iterations: int = 30,
    fixed: int = 0,
    lm_lambda: float = 1e-4,
    tol: float = 1e-9,
    verbose: bool = True,
) -> tuple[list[np.ndarray], list[float]]:
    """高斯-牛顿 + LM 阻尼的位姿图优化。

    Args:
        poses: 初值，n 个 4x4（通常就是里程计的漂移轨迹）
        edges: 里程计边 + 回环边
        fixed: 固定哪个节点（消除规范自由度）

    Returns:
        (优化后的位姿, 每轮的总误差)
    """
    poses = [p.copy() for p in poses]
    n = len(poses)
    history = [total_error(poses, edges)]

    if verbose:
        print(f"  初始误差 {history[0]:.6f}  ({n} 节点, {len(edges)} 边)")

    for it in range(iterations):
        # 稀疏 H 用 COO 三元组累加，最后一次性组装
        rows, cols, vals = [], [], []
        b = np.zeros(6 * n)

        for ed in edges:
            Xi, Xj = poses[ed.i], poses[ed.j]
            e = _residual(Xi, Xj, ed.Z)
            Jr_inv = right_jacobian_inv(e)
            Ji = -Jr_inv @ se3_adjoint(se3_inv(Xj) @ Xi)
            Jj = Jr_inv
            Om = ed.information

            for (a, Ja) in ((ed.i, Ji), (ed.j, Jj)):
                b[6 * a:6 * a + 6] -= Ja.T @ Om @ e
                for (c, Jc) in ((ed.i, Ji), (ed.j, Jj)):
                    block = Ja.T @ Om @ Jc
                    r = np.repeat(np.arange(6 * a, 6 * a + 6), 6)
                    cc = np.tile(np.arange(6 * c, 6 * c + 6), 6)
                    rows.append(r); cols.append(cc); vals.append(block.ravel())

        H = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(6 * n, 6 * n),
        ).tocsr()

        # 固定一个节点：把它对应的 6 行 6 列换成单位阵，b 置零
        idx = np.arange(6 * fixed, 6 * fixed + 6)
        H = H.tolil()
        H[idx, :] = 0
        H[:, idx] = 0
        H[idx, idx] = 1.0
        H = H.tocsr()
        b[idx] = 0.0

        # LM 阻尼：H + λ·diag(H)，避免 H 病态时步长炸掉
        H = H + sp.diags(lm_lambda * H.diagonal())

        try:
            dx = spla.spsolve(H, b)
        except Exception as exc:                      # pragma: no cover
            print(f"  线性求解失败：{exc}")
            break

        if not np.all(np.isfinite(dx)):
            print("  增量出现 nan/inf，停止")
            break

        for k in range(n):
            if k == fixed:
                continue
            poses[k] = poses[k] @ se3_exp(dx[6 * k:6 * k + 6])

        err = total_error(poses, edges)
        history.append(err)
        if verbose:
            print(f"  iter {it + 1:2d}  误差 {err:.6f}  |dx| {np.linalg.norm(dx):.3e}")

        if abs(history[-2] - err) < tol * max(1.0, history[-2]):
            if verbose:
                print(f"  收敛于第 {it + 1} 轮")
            break

    return poses, history


def build_odometry_edges(poses: list[np.ndarray], information: np.ndarray = None) -> list[Edge]:
    """由一条轨迹生成相邻帧之间的里程计边。

    ⚠️ 关键：测量值取自【漂移的里程计轨迹本身】。
       里程计的局部相对位姿是准的，错的是它们累加起来的全局位姿。
       位姿图要做的就是在保持这些局部关系的前提下，让回环边也能被满足。
    """
    return [
        Edge(k, k + 1, se3_inv(poses[k]) @ poses[k + 1], information, "odom")
        for k in range(len(poses) - 1)
    ]
