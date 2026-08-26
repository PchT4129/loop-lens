"""把 VPR 前端接到真实 SLAM 轨迹上：检测回环 -> 求相对位姿 -> 位姿图优化。

这一步补的是什么
----------------
在 Gardens Point 阶段，整个项目只能回答"检索准不准"（Recall@K）。
但 SLAM 里真正关心的是【回环把轨迹修正了多少】。这个模块把链路补全：

    ORB-SLAM3（关闭回环）-> 纯里程计轨迹（漂移上界）
              ↓
    ① DINOv2 全局描述子检索           ← 项目的"现代"检索路径
    ② DISK + LightGlue 匹配           ← 项目的"现代"几何验证路径
    ③ PnP + RANSAC 求相对位姿          ← Gardens Point 上做不到（无内参/深度）
              ↓
    位姿图优化（src/pose_graph.py，自己实现的高斯-牛顿）
              ↓
    ATE 对比：里程计 vs 本方案 vs ORB-SLAM3 自带的 DBoW2 回环

⚠️ 因果约束：只允许和【更早】的关键帧闭环，且必须隔开 min_gap 帧。
   否则会把时间上相邻、本来就该相似的帧当成"回环"，那是自欺欺人。

支持两个数据集，差别都收敛到 SequenceSpec 里：
    TUM RGB-D  —— 轨迹 TUM 格式，深度来自 Kinect 深度图
    KITTI 里程计 —— 轨迹 KITTI 格式（3x4 无时间戳），深度由双目视差算出
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from src.backbones import build_backbone
from src.dataset import get_default_transform
from src.learned_matching import match_pair
from src.pose_graph import Edge, optimize
from src.rgbd_pose import KITTI_00, TUM_FR3, CameraModel, relative_pose
from src.se3 import pose_from_quat, quat_from_pose, se3_inv


# ---------------------------------------------------------------- 轨迹 I/O

def load_trajectory(path, fmt: str) -> tuple[np.ndarray, list[np.ndarray]]:
    """读轨迹。tum: `ts tx ty tz qx qy qz qw`；kitti: 3x4 行主序 12 个数。"""
    data = np.loadtxt(path)
    if fmt == "tum":
        return data[:, 0], [pose_from_quat(r[1:4], r[4:8]) for r in data]
    poses = []
    for row in data:
        T = np.eye(4)
        T[:3, :] = row.reshape(3, 4)
        poses.append(T)
    return np.arange(len(poses), dtype=float), poses


def save_trajectory(path, stamps: np.ndarray, poses: list[np.ndarray], fmt: str) -> None:
    with open(path, "w") as fh:
        for t, T in zip(stamps, poses):
            if fmt == "tum":
                q = quat_from_pose(T)
                fh.write(f"{t:.6f} {T[0,3]:.9f} {T[1,3]:.9f} {T[2,3]:.9f} "
                         f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n")
            else:
                fh.write(" ".join(f"{v:.9e}" for v in T[:3, :].ravel()) + "\n")


# ---------------------------------------------------------------- 数据集描述

@dataclass
class SequenceSpec:
    """一条序列需要的全部信息。两个数据集的差异都收在这里。"""

    rgb_paths: list[str]
    camera: CameraModel
    traj_format: str
    depth_of: callable                  # index -> 深度图（ndarray）


def _tum_spec(seq_dir: Path, assoc_path: Path, kf_stamps: np.ndarray, tol=0.02):
    """TUM：按时间戳把关键帧对到 rgb/depth 图像。"""
    rows = [ln.split() for ln in open(assoc_path) if len(ln.split()) >= 4]
    a_st = np.array([float(r[0]) for r in rows])
    rgb, depth = [r[1] for r in rows], [r[3] for r in rows]

    rgb_paths, depth_paths, keep = [], [], []
    for k, ts in enumerate(kf_stamps):
        i = int(np.argmin(np.abs(a_st - ts)))
        if abs(a_st[i] - ts) > tol:
            continue
        rgb_paths.append(str(seq_dir / rgb[i]))
        depth_paths.append(str(seq_dir / depth[i]))
        keep.append(k)

    def depth_of(idx):
        return cv2.imread(depth_paths[idx], cv2.IMREAD_UNCHANGED)

    return SequenceSpec(rgb_paths, TUM_FR3, "tum", depth_of), np.array(keep)


def _kitti_spec(seq_dir: Path, n_poses: int, step: int):
    """KITTI：轨迹逐帧、无时间戳，第 k 行就是第 k 张图。深度由双目现算。"""
    from src.stereo_depth import depth_from_stereo

    keep = np.arange(0, n_poses, step)
    left = [str(seq_dir / "image_0" / f"{i:06d}.png") for i in keep]
    right = [str(seq_dir / "image_1" / f"{i:06d}.png") for i in keep]

    def depth_of(idx):
        return depth_from_stereo(left[idx], right[idx])

    return SequenceSpec(left, KITTI_00, "kitti", depth_of), keep


# ---------------------------------------------------------------- 检索

@torch.no_grad()
def extract_descriptors(image_paths, backbone="dinov2_vits14", aggregation="gem",
                        image_size=224, batch_size=16, verbose=True) -> torch.Tensor:
    """对关键帧抽全局描述子。输出已 L2 归一化。"""
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_backbone(backbone, dinov2_aggregation=aggregation).to(device).eval()
    tf = get_default_transform(image_size=image_size)

    feats = []
    for s in range(0, len(image_paths), batch_size):
        batch = torch.stack([tf(Image.open(p).convert("RGB"))
                             for p in image_paths[s:s + batch_size]]).to(device)
        feats.append(model(batch).cpu())
        if verbose and (s // batch_size) % 20 == 0:
            print(f"      {min(s + batch_size, len(image_paths))}/{len(image_paths)}", end="\r")
    return torch.cat(feats)


def detect_loop_candidates(features, min_gap=40, top_k=5, min_similarity=0.0):
    """因果回环候选：对每个关键帧 j，在 i <= j - min_gap 里取 top-K。"""
    sim = (features @ features.T).numpy()
    out = []
    for j in range(min_gap, len(features)):
        window = sim[j, : j - min_gap + 1]
        if not len(window):
            continue
        for i in np.argsort(-window)[:top_k]:
            if window[i] >= min_similarity:
                out.append((j, int(i), float(window[i])))
    return out


# ---------------------------------------------------------------- 回环边

def make_information(rot_weight: float, scale: float = 1.0) -> np.ndarray:
    """各向异性信息矩阵 diag([1,1,1,w,w,w])。

    ⚠️ 为什么不能用单位阵（实测踩过的坑）：
    残差 e = [平移(米), 旋转(弧度)]。用单位阵等于说"1 米的平移误差和 1 弧度的
    旋转误差一样糟"，但 1 弧度是 57 度——在 3.7 km 的轨迹上，一个节点转错
    1 度就能让下游整段甩出几十米。旋转必须【贵得多】。

    KITTI 00 实测（500 条真值回环边）：
        w=1     -> ATE 3.28 m
        w=100   -> ATE 2.57 m
        w=1e4   -> ATE 1.97 m     ← 之后趋于平稳
    """
    return np.diag([1.0, 1.0, 1.0, rot_weight, rot_weight, rot_weight]) * scale


def build_odometry_edges(poses, information=None):
    """相邻节点之间的里程计边。

    ⚠️ 测量值取自【漂移的里程计轨迹本身】——局部相对位姿是准的，
       错的是它们累加起来的全局位姿。
    """
    return [Edge(k, k + 1, se3_inv(poses[k]) @ poses[k + 1], information, "odom")
            for k in range(len(poses) - 1)]


def build_loop_edges(candidates, spec: SequenceSpec, min_inliers=40,
                     max_reproj=2.0, information=None, verbose=True):
    """对候选做匹配 + PnP，通过的变成回环边。

    ⚠️ 位姿约定：
        PnP 给出 T_ji（把帧 i 的三维点变到帧 j 的相机系）。
        位姿图的边要的是 Z_ij = X_i^-1 X_j = T_ci_cj，正是 T_ji 的【逆】。
    """
    edges, records = [], []
    for n, (j, i, sim) in enumerate(candidates):
        uv_j, uv_i = match_pair(spec.rgb_paths[j], spec.rgb_paths[i])
        if len(uv_i) >= 6:
            res = relative_pose(uv_i, uv_j, spec.depth_of(i), spec.camera,
                                min_inliers=min_inliers)
            records.append({"j": j, "i": i, "similarity": sim, "num_matches": len(uv_i),
                            **{k: v for k, v in res.items() if k != "T_ji"}})
            if res["success"] and res["reproj_error"] <= max_reproj:
                edges.append(Edge(i, j, se3_inv(res["T_ji"]), information, "loop"))
                if verbose:
                    print(f"  ✓ 回环 {i:4d} <-> {j:4d}  sim={sim:.3f} "
                          f"内点={res['num_inliers']:4d} 重投影={res['reproj_error']:.2f}px")
        if verbose and (n + 1) % 300 == 0:
            print(f"  ... 已验证 {n + 1}/{len(candidates)}，已接受 {len(edges)}")
    return edges, records


# ---------------------------------------------------------------- 主流程

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["tum", "kitti"], required=True)
    ap.add_argument("--trajectory", required=True, help="关闭回环的里程计轨迹")
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--associations", help="仅 TUM 需要")
    ap.add_argument("--output", required=True)
    ap.add_argument("--frame-step", type=int, default=1, help="仅 KITTI：每 N 帧取一个节点")
    ap.add_argument("--backbone", default="dinov2_vits14")
    ap.add_argument("--aggregation", default="gem", choices=["cls", "mean", "gem", "cls+gem"])
    ap.add_argument("--min-gap", type=int, default=40)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--min-inliers", type=int, default=40)
    ap.add_argument("--max-reproj", type=float, default=2.0)
    ap.add_argument("--loop-info", type=float, default=10.0,
                    help="回环边信息矩阵相对里程计边的倍数")
    ap.add_argument("--rot-info", type=float, default=1e4,
                    help="旋转分量相对平移分量的权重。见 make_information 的说明")
    ap.add_argument("--iterations", type=int, default=30)
    args = ap.parse_args()

    seq_dir = Path(args.sequence)
    fmt = "tum" if args.dataset == "tum" else "kitti"

    print("[1/5] 读取里程计轨迹")
    stamps, poses = load_trajectory(args.trajectory, fmt)
    if args.dataset == "tum":
        spec, keep = _tum_spec(seq_dir, Path(args.associations), stamps)
    else:
        spec, keep = _kitti_spec(seq_dir, len(poses), args.frame_step)
    stamps = stamps[keep]
    poses = [poses[k] for k in keep]
    length = sum(np.linalg.norm(poses[k + 1][:3, 3] - poses[k][:3, 3])
                 for k in range(len(poses) - 1))
    print(f"      {len(poses)} 个节点，轨迹长 {length:.1f} m")

    print(f"[2/5] 抽取 {args.backbone} 描述子（{args.aggregation}）")
    feats = extract_descriptors(spec.rgb_paths, args.backbone, args.aggregation)
    print(f"      {tuple(feats.shape)}")

    print(f"[3/5] 因果回环检索（min_gap={args.min_gap}, top_k={args.top_k}）")
    cands = detect_loop_candidates(feats, args.min_gap, args.top_k)
    print(f"      {len(cands)} 个候选")

    print("[4/5] LightGlue 匹配 + PnP 求相对位姿")
    odom_info = make_information(args.rot_info)
    loop_info = make_information(args.rot_info, args.loop_info)
    loop_edges, records = build_loop_edges(
        cands, spec, args.min_inliers, args.max_reproj, loop_info)
    print(f"      接受 {len(loop_edges)} 条回环边"
          f"（通过率 {len(loop_edges)/max(len(cands),1)*100:.1f}%）")

    print("[5/5] 位姿图优化")
    edges = build_odometry_edges(poses, odom_info) + loop_edges
    opt, hist = optimize(poses, edges, iterations=args.iterations, verbose=False)
    print(f"      误差 {hist[0]:.4f} -> {hist[-1]:.6f}（{len(hist)-1} 轮）")

    save_trajectory(args.output, stamps, opt, fmt)
    np.save(Path(args.output).with_suffix(".records.npy"), np.array(records, dtype=object))
    shift = np.mean([np.linalg.norm(a[:3, 3] - b[:3, 3]) for a, b in zip(opt, poses)])
    print(f"\n写入 {args.output}　节点平均位移 {shift:.4f} m")


if __name__ == "__main__":
    main()
