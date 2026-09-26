import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from src.retrieve import load_feature_file
from src.sequence_match import parse_velocities, sequence_rerank
from src.confidence import apply_gate, fit_gate, gate_scores
from src.ground_truth import PairManifestGroundTruth


@dataclass
class LoopClosureProposal:
    query_path: str
    candidate_path: str
    retrieval_score: float
    sequence_score: float
    num_matches: int = 0
    num_inliers: int = 0
    inlier_ratio: float = 0.0
    accepted: bool | None = None
    fundamental_matrix: list[list[float]] | None = None


def image_index_from_path(path: str) -> int:
    match = re.search(r"Image(\d+)\.jpg$", path)
    if match is None:
        raise ValueError(f"Could not parse image index from path: {path}")

    return int(match.group(1))


def is_correct_match(
    query_path: str,
    database_path: str,
    tolerance: int,
    ground_truth: PairManifestGroundTruth | None = None,
) -> bool:
    if ground_truth is not None:
        return ground_truth.is_correct(query_path, database_path)
    query_index = image_index_from_path(query_path)
    database_index = image_index_from_path(database_path)

    return abs(query_index - database_index) <= tolerance


def evaluate_retrieval(
    query_paths: list[str],
    database_paths: list[str],
    top_indices: torch.Tensor,
    recall_ks: list[int],
    precision_k: int,
    tolerance: int,
    ground_truth: PairManifestGroundTruth | None = None,
):
    metrics = {}

    for k in recall_ks:
        num_success = 0

        for query_idx, query_path in enumerate(query_paths):
            retrieved_indices = top_indices[query_idx, :k]

            hit = any(
                is_correct_match(
                    query_path=query_path,
                    database_path=database_paths[db_idx.item()],
                    tolerance=tolerance,
                    ground_truth=ground_truth,
                )
                for db_idx in retrieved_indices
            )

            if hit:
                num_success += 1

        metrics[f"recall@{k}"] = num_success / len(query_paths)

    total_precision = 0.0

    for query_idx, query_path in enumerate(query_paths):
        retrieved_indices = top_indices[query_idx, :precision_k]

        num_correct = sum(
            is_correct_match(
                query_path=query_path,
                database_path=database_paths[db_idx.item()],
                tolerance=tolerance,
                ground_truth=ground_truth,
            )
            for db_idx in retrieved_indices
        )

        total_precision += num_correct / precision_k

    metrics[f"precision@{precision_k}"] = total_precision / len(query_paths)

    return metrics


def select_query_indices(
    query_paths: list[str],
    split_name: str,
    min_index: int | None = None,
    max_index: int | None = None,
) -> list[int]:
    selected_indices = []

    for index, path in enumerate(query_paths):
        if f"/{split_name}/" not in path:
            continue

        if min_index is not None or max_index is not None:
            image_index = image_index_from_path(path)
            if min_index is not None and image_index < min_index:
                continue
            if max_index is not None and image_index > max_index:
                continue

        selected_indices.append(index)

    return selected_indices


def recall_at_full_precision(
    query_paths: list[str],
    database_paths: list[str],
    top_indices: torch.Tensor,
    confidences: torch.Tensor,
    tolerance: int,
    ground_truth: PairManifestGroundTruth | None = None,
) -> tuple[float, float]:
    """Recall@100%Precision —— SLAM 回环检测的经典指标。

    问的是：**在一个假阳性都不产生的判定阈值下，最多能召回多少回环？**

    之所以是这个指标，是因为 M1 讲过的代价不对称：漏检只是少一次修正机会，
    而误检会把两个不同地点焊在一起，位姿图优化会把整张地图撕坏且通常不可逆。
    所以真正该问的不是"平均准确率多少"，而是"零误检的前提下能做到多少召回"。

    做法：把每个 query 的 top-1 结果按置信度从高到低排序，逐个降低阈值，
    统计此时的 precision 和 recall，找出 precision 仍为 100% 时的最大 recall。

    Args:
        confidences: 每个 query 对其 top-1 候选的置信度。开了几何验证时
            是内点数，否则是余弦相似度。
    Returns:
        (recall_at_100_precision, 对应的阈值)
    """
    records = []
    for query_idx, query_path in enumerate(query_paths):
        db_idx = top_indices[query_idx, 0].item()
        correct = is_correct_match(
            query_path, database_paths[db_idx], tolerance, ground_truth
        )
        records.append((confidences[query_idx].item(), correct))

    # 按置信度降序：阈值从高往低放，接受的样本逐个增加
    records.sort(key=lambda r: -r[0])

    total = len(records)
    num_accepted = 0
    num_correct = 0
    best_recall = 0.0
    best_threshold = float("inf")

    for confidence, correct in records:
        num_accepted += 1
        num_correct += int(correct)

        precision = num_correct / num_accepted
        recall = num_correct / total

        if precision >= 1.0 and recall > best_recall:
            best_recall = recall
            best_threshold = confidence

    return best_recall, best_threshold


def open_set_metrics(
    query_paths: list[str],
    database_paths: list[str],
    top_indices: torch.Tensor,
    confidences: torch.Tensor,
    tolerance: int,
    threshold: float | None = None,
    accepted: torch.Tensor | None = None,
    ground_truth: PairManifestGroundTruth | None = None,
) -> dict[str, float]:
    """开集评测：有些 query 在 database 里【根本没有正确答案】，系统应该拒绝它们。

    为什么需要这个
    --------------
    闭集设定下每个 query 必有正确答案，于是"拒绝"这个动作永远是错的，
    门控机制的价值完全无法衡量——这也是 v2 里 `--geometric-mode gate`
    的指标恒等于不加门控的原因。

    但真实 SLAM 里**绝大多数时刻机器人都在新地方**，"没有回环"才是常态，
    而误报一次回环就可能撕坏整张地图。所以"能不能正确地拒绝"才是关键能力。

    构造方式：把 database 的某个帧号区间移除，那么 query 里对应那段的
    就成了"应该被拒绝"的样本。不需要额外数据。

    四种结果：
                     应该接受            应该拒绝
        系统接受     TP(且检索对)        FP ← 灾难性的假阳性回环
        系统拒绝     FN                  TN
    """
    database_indices = (
        [image_index_from_path(p) for p in database_paths]
        if ground_truth is None else []
    )

    tp = fp = fn = tn = 0
    num_with_match = 0

    for query_idx, query_path in enumerate(query_paths):
        # 这个 query 在【过滤后的】database 里还有没有正确答案？
        if ground_truth is None:
            query_index = image_index_from_path(query_path)
            has_match = any(abs(d - query_index) <= tolerance for d in database_indices)
        else:
            has_match = ground_truth.has_match(query_path, database_paths)
        num_with_match += int(has_match)

        is_accepted = (
            bool(accepted[query_idx])
            if accepted is not None
            else confidences[query_idx].item() >= float(threshold)
        )
        top1_correct = is_correct_match(
            query_path, database_paths[top_indices[query_idx, 0].item()], tolerance,
            ground_truth,
        )

        if is_accepted and top1_correct:
            tp += 1
        elif is_accepted:
            fp += 1               # 断言了一个回环，但它是错的
        elif has_match:
            fn += 1               # 本该找到却拒绝了
        else:
            tn += 1               # 正确地拒绝了

    precision = tp / max(tp + fp, 1)
    recall = tp / max(num_with_match, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "open_set/precision": precision,
        "open_set/recall": recall,
        "open_set/f1": f1,
        "open_set/TP": tp,
        "open_set/FP": fp,
        "open_set/FN": fn,
        "open_set/TN": tn,
        "open_set/num_with_match": num_with_match,
        "open_set/num_without_match": len(query_paths) - num_with_match,
    }


def open_set_targets(
    query_paths: list[str],
    database_paths: list[str],
    top_indices: torch.Tensor,
    tolerance: int,
    ground_truth: PairManifestGroundTruth | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (top1_correct, has_any_valid_match) for gate fitting."""
    database_indices = (
        [image_index_from_path(path) for path in database_paths]
        if ground_truth is None else []
    )
    top1_correct = []
    has_match = []
    for query_idx, query_path in enumerate(query_paths):
        if ground_truth is None:
            query_index = image_index_from_path(query_path)
            has_match.append(
                any(abs(index - query_index) <= tolerance for index in database_indices)
            )
        else:
            has_match.append(ground_truth.has_match(query_path, database_paths))
        candidate = database_paths[top_indices[query_idx, 0].item()]
        top1_correct.append(
            is_correct_match(query_path, candidate, tolerance, ground_truth)
        )
    return torch.tensor(top1_correct), torch.tensor(has_match)


def open_set_average_precision(
    confidences: torch.Tensor,
    top1_correct: torch.Tensor,
    has_match: torch.Tensor,
) -> float:
    """Area under the open-set precision-recall curve without fitting a threshold."""
    order = torch.argsort(confidences, descending=True)
    correct = top1_correct[order].float()
    cumulative_tp = correct.cumsum(dim=0)
    precision = cumulative_tp / torch.arange(1, len(order) + 1)
    return float((precision * correct).sum() / max(int(has_match.sum()), 1))


def bootstrap_retrieval_cis(
    query_paths: list[str],
    database_paths: list[str],
    top_indices: torch.Tensor,
    recall_ks: list[int],
    tolerance: int,
    samples: int,
    seed: int = 0,
    ground_truth: PairManifestGroundTruth | None = None,
) -> dict[str, float]:
    """Non-parametric 95% CIs over queries for Recall@K."""
    if samples <= 0:
        return {}
    generator = torch.Generator().manual_seed(seed)
    num_queries = len(query_paths)
    metrics: dict[str, float] = {}
    for k in recall_ks:
        hits = torch.tensor([
            any(
                is_correct_match(
                    query_path, database_paths[index.item()], tolerance, ground_truth
                )
                for index in top_indices[query_idx, :k]
            )
            for query_idx, query_path in enumerate(query_paths)
        ], dtype=torch.float32)
        draws = torch.randint(num_queries, (samples, num_queries), generator=generator)
        estimates = hits[draws].mean(dim=1)
        metrics[f"recall@{k}/ci95_low"] = float(torch.quantile(estimates, 0.025))
        metrics[f"recall@{k}/ci95_high"] = float(torch.quantile(estimates, 0.975))
    return metrics


def bootstrap_open_set_cis(
    accepted: torch.Tensor,
    top1_correct: torch.Tensor,
    has_match: torch.Tensor,
    samples: int,
    seed: int = 0,
) -> dict[str, float]:
    if samples <= 0:
        return {}
    generator = torch.Generator().manual_seed(seed)
    num_queries = len(accepted)
    draws = torch.randint(num_queries, (samples, num_queries), generator=generator)
    sampled_accepted = accepted.bool()[draws]
    sampled_correct = top1_correct.bool()[draws]
    sampled_has_match = has_match.bool()[draws]
    tp = (sampled_accepted & sampled_correct).sum(dim=1).float()
    fp = (sampled_accepted & ~sampled_correct).sum(dim=1).float()
    precision = tp / (tp + fp).clamp(min=1)
    recall = tp / sampled_has_match.sum(dim=1).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    intervals = {}
    for name, values in {"precision": precision, "recall": recall, "f1": f1}.items():
        intervals[f"deployed/open_set/{name}/ci95_low"] = float(torch.quantile(values, 0.025))
        intervals[f"deployed/open_set/{name}/ci95_high"] = float(torch.quantile(values, 0.975))
    return intervals


def sweep_open_set(
    query_paths: list[str],
    database_paths: list[str],
    top_indices: torch.Tensor,
    confidences: torch.Tensor,
    tolerance: int,
    ground_truth: PairManifestGroundTruth | None = None,
) -> dict[str, float]:
    """扫描置信度阈值，找最佳 F1 和零假阳性下的最大召回。"""
    thresholds = sorted({c.item() for c in confidences} | {0.0})

    best_f1 = {"open_set/best_f1": 0.0, "open_set/best_f1_threshold": 0.0}
    recall_at_100p = 0.0
    threshold_at_100p = float("inf")

    for threshold in thresholds:
        m = open_set_metrics(
            query_paths, database_paths, top_indices, confidences, tolerance, threshold,
            ground_truth=ground_truth,
        )
        if m["open_set/f1"] > best_f1["open_set/best_f1"]:
            best_f1 = {
                "open_set/best_f1": m["open_set/f1"],
                "open_set/best_f1_threshold": threshold,
            }
        if m["open_set/precision"] >= 1.0 and m["open_set/recall"] > recall_at_100p:
            recall_at_100p = m["open_set/recall"]
            threshold_at_100p = threshold

    return {
        **best_f1,
        "open_set/recall@100%precision": recall_at_100p,
        "open_set/threshold@100%precision": threshold_at_100p,
    }


def apply_geometric_verification(
    query_paths: list[str],
    database_paths: list[str],
    top_indices: torch.Tensor,
    top_scores: torch.Tensor,
    mode: str = "gate",
    verifier: str = "orb",
):
    """对每个 query 的 Top-K 候选跑 ORB+RANSAC 几何验证。

    两种模式，实测表明它们的适用性差别很大：

    mode="gate"（默认，也是 ORB-SLAM 里的用法）
        不改变排序，并返回 top-1 的完整几何统计。接受/拒绝由后续的
        validation-fitted gate 统一完成。

    mode="rerank"
        按内点数重排 Top-K。⚠️ 实测这会【降低】性能：
            day_right : R@1 0.967 -> 0.833
            night_right: R@1 1.000 -> 0.667（配合序列匹配时）

    为什么重排会有害？关键在于两个任务的难度完全不同：

        任务            正确 vs 错误的内点数（实测，held-out 070-099）
        ─────────────────────────────────────────────────────────
        判别"是不是同一地点"   day: 134.7 vs 26.1  → 100% 可分
        在 Top-10 内部排序     day:  58.1 vs 40.2  → 有信号但方差大
                              night: 25.1 vs 24.4  → 基本无信号

    Top-K 候选全是相隔几帧的近似图像，ORB 内点数无法在它们之间可靠排序；
    而 CNN 的 top-1 已经相当准，用一个高方差信号去重排只会把对的挤下去。
    这就是为什么几何验证在真实 SLAM 里是【门控】而不是【排序器】。
    """
    if verifier == "lightglue":
        from src.learned_matching import verify_candidates
    else:
        from src.geometric_verification import verify_candidates

    num_queries, k = top_indices.shape
    new_indices = top_indices.clone()
    new_scores = top_scores.clone()
    top1_stats: list[dict] = []

    for query_idx, query_path in enumerate(query_paths):
        candidate_indices = (
            top_indices[query_idx]
            if mode == "rerank"
            else top_indices[query_idx, :1]
        )
        candidates = [database_paths[i.item()] for i in candidate_indices]
        stats = verify_candidates(query_path, candidates)
        inliers = [item["num_inliers"] for item in stats]

        if mode == "rerank":
            # 稳定排序：内点数相同时保持原有的 CNN 相似度顺序
            order = sorted(range(k), key=lambda i: -inliers[i])
            for rank, src in enumerate(order):
                new_indices[query_idx, rank] = top_indices[query_idx, src]
                new_scores[query_idx, rank] = top_scores[query_idx, src]
            best_stats = stats[order[0]]
        else:
            best_stats = stats[0]
        top1_stats.append(best_stats)

    return new_indices, new_scores, top1_stats


def build_signals(
    retrieval_scores: torch.Tensor,
    sequence_scores: torch.Tensor,
    geometric_stats: list[dict] | None,
) -> dict[str, torch.Tensor]:
    num_queries = len(sequence_scores)
    if geometric_stats is None:
        ratios = torch.zeros(num_queries)
        matches = torch.zeros(num_queries)
    else:
        ratios = torch.tensor([item["inlier_ratio"] for item in geometric_stats])
        matches = torch.tensor([item["num_matches"] for item in geometric_stats]).float()
    return {
        "retrieval_score": retrieval_scores.float(),
        "sequence_score": sequence_scores.float(),
        "inlier_ratio": ratios.float(),
        "log_num_matches": torch.log1p(matches),
    }


def write_proposals(
    output_path: str | Path,
    query_paths: list[str],
    database_paths: list[str],
    top_indices: torch.Tensor,
    signals: dict[str, torch.Tensor],
    geometric_stats: list[dict] | None,
    accepted: torch.Tensor | None,
) -> None:
    proposals = []
    for query_idx, query_path in enumerate(query_paths):
        stats = geometric_stats[query_idx] if geometric_stats is not None else {}
        proposals.append(asdict(LoopClosureProposal(
            query_path=query_path,
            candidate_path=database_paths[top_indices[query_idx, 0].item()],
            retrieval_score=float(signals["retrieval_score"][query_idx]),
            sequence_score=float(signals["sequence_score"][query_idx]),
            num_matches=int(stats.get("num_matches", 0)),
            num_inliers=int(stats.get("num_inliers", 0)),
            inlier_ratio=float(stats.get("inlier_ratio", 0.0)),
            accepted=None if accepted is None else bool(accepted[query_idx]),
            fundamental_matrix=stats.get("fundamental_matrix"),
        )))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(proposals, indent=2), encoding="utf-8")


def print_metrics(
    split_name: str,
    num_queries: int,
    tolerance: int,
    metrics: dict[str, float],
    min_index: int | None = None,
    max_index: int | None = None,
    extra_lines: list[str] | None = None,
    distance_threshold_m: float | None = None,
):
    print("=" * 80)
    print(f"Split: {split_name}")
    print(f"Num queries: {num_queries}")
    if distance_threshold_m is None:
        print(f"Tolerance: ±{tolerance} frames")
    else:
        print(f"Ground truth: distance <= {distance_threshold_m:g} m")

    if min_index is not None or max_index is not None:
        min_label = "*" if min_index is None else str(min_index)
        max_label = "*" if max_index is None else str(max_index)
        print(f"Index range: {min_label}-{max_label}")

    for line in extra_lines or []:
        print(line)

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--tolerance", type=int, default=3)
    parser.add_argument(
        "--ground-truth-manifest", type=str, default=None,
        help="CSV with query_path,database_path,distance_m for metric benchmarks",
    )
    parser.add_argument("--distance-threshold-m", type=float, default=25.0)
    parser.add_argument("--split-name", type=str, default=None)
    parser.add_argument("--min-index", type=int, default=None)
    parser.add_argument("--max-index", type=int, default=None)
    parser.add_argument(
        "--recall-ks", type=int, nargs="+", default=[1, 5, 10],
        help="要报告的 Recall@K 的 K 列表",
    )
    parser.add_argument("--precision-k", type=int, default=5)
    parser.add_argument(
        "--seq-window", type=int, default=0,
        help="序列匹配的窗口半径。0 表示关闭（单帧匹配）",
    )
    parser.add_argument(
        "--seq-causal", action="store_true",
        help="序列匹配只使用过去帧（在线 SLAM 的真实条件）",
    )
    parser.add_argument(
        "--seq-velocities", type=str, default="1.0",
        help="序列匹配搜索的速度比，逗号分隔。默认 1.0 假设两次采集速度一致；"
             "传多个值会在各速度上取最优，代价是误匹配得高分的机会也变多",
    )
    parser.add_argument(
        "--geometric-verify", action="store_true",
        help="对 Top-K 候选跑 ORB+RANSAC 几何验证",
    )
    parser.add_argument(
        "--geometric-mode", choices=["gate", "rerank"], default="gate",
        help="gate=只做接受/拒绝(默认，ORB-SLAM 的用法)；rerank=按内点数重排(实测有害)",
    )
    parser.add_argument(
        "--db-exclude-range", type=int, nargs=2, default=None, metavar=("LO", "HI"),
        help="从 database 中移除该帧号区间，构造【开集】：落在这段的 query "
             "将没有正确答案、应该被系统拒绝。不传则是闭集（每个 query 必有答案）",
    )
    parser.add_argument(
        "--verifier", choices=["orb", "lightglue"], default="orb",
        help="orb=手工特征(快，但跨昼夜判别力等同随机)；"
             "lightglue=DISK+LightGlue 学习型特征(建议 GPU，跨昼夜判别力恢复)",
    )
    parser.add_argument(
        "--gate-mode", choices=["similarity", "geometry", "joint", "logistic"],
        default="joint",
        help="接受/拒绝策略。joint 同时要求序列分数与几何内点率过阈值",
    )
    parser.add_argument(
        "--fit-thresholds", action="store_true",
        help="在当前 validation split 上拟合 gate；必须同时传 --thresholds-out",
    )
    parser.add_argument("--thresholds-in", type=str, default=None,
                        help="读取 validation 上冻结的 gate JSON，用于 test")
    parser.add_argument("--thresholds-out", type=str, default=None,
                        help="保存拟合出的 gate JSON")
    parser.add_argument("--proposals-out", type=str, default=None,
                        help="将逐帧 LoopClosureProposal 写成 JSON")
    parser.add_argument("--bootstrap-samples", type=int, default=0,
                        help=">0 时对 Recall@K 做 query bootstrap 95%% CI")
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--inlier-threshold", type=int, default=20,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.fit_thresholds and not args.thresholds_out:
        parser.error("--fit-thresholds requires --thresholds-out")
    if args.fit_thresholds and args.thresholds_in:
        parser.error("fit and apply are separate phases; do not combine threshold flags")
    if args.fit_thresholds and args.split_name is None and args.ground_truth_manifest is None:
        parser.error("threshold fitting requires one explicit --split-name")
    if args.fit_thresholds and args.db_exclude_range is None and args.ground_truth_manifest is None:
        parser.error(
            "threshold fitting requires --db-exclude-range or an open-set ground-truth manifest"
        )
    if args.db_exclude_range is not None and args.ground_truth_manifest is not None:
        parser.error("use either a metric manifest or frame-index exclusion, not both")
    if args.ground_truth_manifest is not None and (
        args.min_index is not None or args.max_index is not None
    ):
        parser.error("frame-index min/max filters are unavailable with metric manifests")

    gate_config = None
    if args.thresholds_in:
        gate_config = json.loads(Path(args.thresholds_in).read_text(encoding="utf-8"))
        args.gate_mode = gate_config["gate_mode"]
    if args.gate_mode in {"geometry", "joint", "logistic"} and not args.geometric_verify:
        if args.fit_thresholds or args.thresholds_in:
            parser.error(f"--gate-mode {args.gate_mode} requires --geometric-verify")

    pipeline_config = {
        "seq_window": args.seq_window,
        "seq_causal": args.seq_causal,
        "seq_velocities": args.seq_velocities,
        "geometric_verify": args.geometric_verify,
        "geometric_mode": args.geometric_mode,
        "verifier": args.verifier if args.geometric_verify else None,
        "ground_truth_protocol": (
            f"metric@{args.distance_threshold_m:g}m"
            if args.ground_truth_manifest else f"frame_index@{args.tolerance}"
        ),
    }
    # --- 修 M5 的静默失效 ---
    # Python/PyTorch 的切片越界【不会报错】，会静默返回它有的那部分。
    # 所以 --top-k 小于 recall_ks 最大值时，算出来的 recall@10 其实是 recall@5，
    # 但输出仍然打印 recall@10。这里改成 fail fast。
    required_k = max(max(args.recall_ks), args.precision_k)
    if args.top_k < required_k:
        raise ValueError(
            f"--top-k ({args.top_k}) 必须 >= max(recall_ks, precision_k) = {required_k}，"
            f"否则切片会被静默截断，指标是错的"
        )

    database_features, database_paths, database_meta = load_feature_file(
        args.database, include_meta=True
    )
    query_features, query_paths, query_meta = load_feature_file(
        args.query, include_meta=True
    )
    if database_meta != query_meta:
        raise ValueError(
            "database and query feature artifacts were produced by different configs: "
            f"{database_meta} != {query_meta}"
        )
    pipeline_config["feature_meta"] = query_meta or {
        "feature_dim": int(query_features.shape[1]),
        "legacy_artifact_without_meta": True,
    }
    if gate_config is not None and gate_config.get("pipeline") != pipeline_config:
        parser.error(
            "threshold file was fitted for a different pipeline or feature configuration: "
            f"expected {gate_config.get('pipeline')}, got {pipeline_config}"
        )
    ground_truth = (
        PairManifestGroundTruth.from_csv(
            args.ground_truth_manifest, args.distance_threshold_m
        )
        if args.ground_truth_manifest else None
    )

    if args.db_exclude_range is not None:
        lo, hi = args.db_exclude_range
        keep = [
            i for i, p in enumerate(database_paths)
            if not (lo <= image_index_from_path(p) <= hi)
        ]
        removed = len(database_paths) - len(keep)
        database_features = database_features[keep]
        database_paths = [database_paths[i] for i in keep]
        print(f"[开集] 从 database 移除帧号 {lo}-{hi} 共 {removed} 张，"
              f"剩余 {len(database_paths)} 张")

    if len(database_paths) < required_k:
        raise ValueError(
            f"filtered database has {len(database_paths)} images, fewer than the "
            f"{required_k} required by recall/precision metrics"
        )

    if ground_truth is not None and args.split_name is None:
        split_names = ["metric_manifest"]
    else:
        split_names = [args.split_name] if args.split_name is not None else [
            "day_right",
            "night_right",
        ]

    for split_name in split_names:
        split_indices = (
            list(range(len(query_paths)))
            if split_name == "metric_manifest"
            else select_query_indices(
                query_paths=query_paths,
                split_name=split_name,
                min_index=args.min_index,
                max_index=args.max_index,
            )
        )

        if len(split_indices) == 0:
            raise ValueError(
                f"No queries found for split={split_name}, "
                f"min_index={args.min_index}, max_index={args.max_index}"
            )

        split_query_paths = [query_paths[index] for index in split_indices]

        # 按 split 单独算相似度矩阵：序列匹配要求 query 在时间上连续且有序，
        # 跨 split 混在一起做对角线聚合是没有意义的。
        retrieval_started = time.perf_counter()
        raw_similarities = query_features[split_indices] @ database_features.T
        similarities = raw_similarities

        if args.seq_window > 0:
            similarities = sequence_rerank(
                similarities,
                window=args.seq_window,
                causal=args.seq_causal,
                velocities=parse_velocities(args.seq_velocities),
            )

        top_scores, split_top_indices = torch.topk(
            similarities, k=min(args.top_k, database_features.shape[0]), dim=1
        )
        retrieval_seconds = time.perf_counter() - retrieval_started
        sequence_scores = top_scores[:, 0].clone()
        retrieval_scores = raw_similarities.gather(
            1, split_top_indices[:, :1]
        ).squeeze(1)
        geometric_stats = None
        geometry_seconds = 0.0

        if args.geometric_verify:
            geometry_started = time.perf_counter()
            split_top_indices, top_scores, geometric_stats = apply_geometric_verification(
                query_paths=split_query_paths,
                database_paths=database_paths,
                top_indices=split_top_indices,
                top_scores=top_scores,
                mode=args.geometric_mode,
                verifier=args.verifier,
            )
            geometry_seconds = time.perf_counter() - geometry_started
            if args.geometric_mode == "rerank":
                sequence_scores = top_scores[:, 0].clone()
                retrieval_scores = raw_similarities.gather(
                    1, split_top_indices[:, :1]
                ).squeeze(1)

        signals = build_signals(retrieval_scores, sequence_scores, geometric_stats)

        metrics = evaluate_retrieval(
            query_paths=split_query_paths,
            database_paths=database_paths,
            top_indices=split_top_indices,
            recall_ks=args.recall_ks,
            precision_k=args.precision_k,
            tolerance=args.tolerance,
            ground_truth=ground_truth,
        )
        metrics.update(bootstrap_retrieval_cis(
            query_paths=split_query_paths,
            database_paths=database_paths,
            top_indices=split_top_indices,
            recall_ks=args.recall_ks,
            tolerance=args.tolerance,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
            ground_truth=ground_truth,
        ))
        metrics["runtime/retrieval_ms_per_query"] = (
            1000 * retrieval_seconds / len(split_query_paths)
        )
        if args.geometric_verify:
            metrics["runtime/geometry_ms_per_query"] = (
                1000 * geometry_seconds / len(split_query_paths)
            )

        recall_100p, threshold = recall_at_full_precision(
            query_paths=split_query_paths,
            database_paths=database_paths,
            top_indices=split_top_indices,
            confidences=signals["sequence_score"],
            tolerance=args.tolerance,
            ground_truth=ground_truth,
        )
        metrics["oracle/sequence_recall@100%precision"] = recall_100p

        top1_correct, has_match = open_set_targets(
            split_query_paths, database_paths, split_top_indices, args.tolerance,
            ground_truth,
        )
        accepted = None

        if args.db_exclude_range is not None or ground_truth is not None:
            metrics["open_set/sequence_auprc"] = open_set_average_precision(
                signals["sequence_score"], top1_correct, has_match
            )
            oracle = sweep_open_set(
                query_paths=split_query_paths,
                database_paths=database_paths,
                top_indices=split_top_indices,
                confidences=signals["sequence_score"],
                tolerance=args.tolerance,
                ground_truth=ground_truth,
            )
            metrics.update({f"oracle/{key}": value for key, value in oracle.items()})

            if args.fit_thresholds:
                gate_config, fit_metrics = fit_gate(
                    args.gate_mode, signals, top1_correct, has_match
                )
                gate_config["fitted_on"] = {
                    "split_name": split_name,
                    "min_index": args.min_index,
                    "max_index": args.max_index,
                    "db_exclude_range": args.db_exclude_range,
                    "tolerance": args.tolerance,
                }
                gate_config["pipeline"] = pipeline_config
                Path(args.thresholds_out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.thresholds_out).write_text(
                    json.dumps(gate_config, indent=2), encoding="utf-8"
                )
                metrics.update({f"validation_gate/{key}": value for key, value in fit_metrics.items()})

            if gate_config is not None and args.thresholds_in:
                accepted = apply_gate(gate_config, signals)
                gate_confidences = gate_scores(gate_config, signals)
                deployed = open_set_metrics(
                    query_paths=split_query_paths,
                    database_paths=database_paths,
                    top_indices=split_top_indices,
                    confidences=gate_confidences,
                    tolerance=args.tolerance,
                    accepted=accepted,
                    ground_truth=ground_truth,
                )
                metrics.update({f"deployed/{key}": value for key, value in deployed.items()})
                metrics.update(bootstrap_open_set_cis(
                    accepted, top1_correct, has_match,
                    samples=args.bootstrap_samples, seed=args.bootstrap_seed,
                ))

        if args.proposals_out:
            proposal_path = Path(args.proposals_out)
            if len(split_names) > 1:
                proposal_path = proposal_path.with_name(
                    f"{proposal_path.stem}_{split_name}{proposal_path.suffix or '.json'}"
                )
            write_proposals(
                proposal_path, split_query_paths, database_paths, split_top_indices,
                signals, geometric_stats, accepted,
            )

        print_metrics(
            split_name=split_name,
            num_queries=len(split_query_paths),
            tolerance=args.tolerance,
            metrics=metrics,
            min_index=args.min_index,
            max_index=args.max_index,
            extra_lines=_describe_config(args, threshold),
            distance_threshold_m=(
                args.distance_threshold_m if ground_truth is not None else None
            ),
        )


def _describe_config(args, threshold) -> list[str]:
    lines = []
    if args.seq_window > 0:
        mode = "causal(仅过去帧)" if args.seq_causal else "non-causal(前后帧)"
        lines.append(
            f"序列匹配: window=±{args.seq_window} {mode} velocities={args.seq_velocities}"
        )
    if args.geometric_verify:
        lines.append(
            f"几何验证: {args.verifier}+RANSAC mode={args.geometric_mode}"
        )
    if args.thresholds_in:
        lines.append(f"部署 gate: {args.gate_mode} thresholds={args.thresholds_in}")
    elif args.fit_thresholds:
        lines.append(f"验证集拟合 gate: {args.gate_mode} -> {args.thresholds_out}")
    if threshold != float("inf"):
        lines.append(f"oracle 零假阳性序列分数阈值: {threshold:.4f}")
    return lines


if __name__ == "__main__":
    main()
