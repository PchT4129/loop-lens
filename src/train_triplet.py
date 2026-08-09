import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import get_default_transform, get_train_transform
from src.evaluate import evaluate_retrieval, image_index_from_path
from src.extract_features import extract_with_model
from src.losses import (
    build_invalid_negative_mask,
    info_nce_loss,
    infonce_diagnostics,
)
from src.models import ResNet18FeatureExtractor
from src.retrieve import retrieve_top_k
from src.triplet_dataset import TripletPlaceDataset


def train_one_epoch(model, loader, criterion, optimizer, device, loss_type="triplet",
                    tau=0.07, negative_gap=20):
    model.train()

    total_loss = 0.0
    total_zero_grad = 0          # loss 恰好为 0 的样本数（triplet 才会出现）
    diag_sum = {"valid_negatives": 0.0, "top1_acc": 0.0, "effective_negs": 0.0}
    num_batches = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        anchor = batch["anchor"].to(device)
        positive = batch["positive"].to(device)

        # 各分支【合并成一个 batch 做一次 forward】。
        #
        # 之前的写法是分开 forward，那样有个隐蔽的问题：anchor 全是夜间图、
        # positive/negative 全是白天图，train() 模式下 BatchNorm 用的是当前 batch
        # 的统计量，于是 anchor 和 positive 实际上被【两个不同的归一化函数】处理，
        # 输出根本不在同一个空间里——而 loss 却在计算它们之间的距离。
        # 实测：分开 vs 合并，anchor 特征余弦仅 0.8655，训练信号差 24%。
        #
        # 合并之后 BN 看到的是昼夜混合分布，各分支共用同一套归一化，
        # 而且一次 forward 比多次的 GPU 利用率更高，顺带还更快。
        if loss_type == "triplet":
            negative = batch["negative"].to(device)
            merged = torch.cat([anchor, positive, negative], dim=0)
            a_f, p_f, n_f = model(merged).chunk(3, dim=0)
            loss = criterion(a_f, p_f, n_f)

            # 统计有多少三元组已满足 margin、不再产生梯度
            with torch.no_grad():
                per_sample = torch.relu(
                    (a_f - p_f).norm(dim=1) - (a_f - n_f).norm(dim=1) + criterion.margin
                )
                total_zero_grad += int((per_sample == 0).sum())
        else:
            merged = torch.cat([anchor, positive], dim=0)
            a_f, p_f = model(merged).chunk(2, dim=0)

            invalid = build_invalid_negative_mask(
                batch["anchor_index"].to(device),
                batch["positive_index"].to(device),
                negative_gap,
            )
            loss = info_nce_loss(a_f, p_f, tau=tau, invalid_mask=invalid)

            d = infonce_diagnostics(a_f, p_f, tau, invalid)
            for k in diag_sum:
                diag_sum[k] += d[k]
            num_batches += 1

        #反向传播三部曲
        optimizer.zero_grad() #清空上一次残留梯度（否则梯度累计，不是bug而是特性）
        loss.backward() #计算梯度并反向传播
        optimizer.step() #用梯度更新参数

        total_loss += loss.item() * anchor.size(0)

    n = len(loader.dataset)
    stats = {"loss": total_loss / n}
    if loss_type == "triplet":
        stats["zero_grad_ratio"] = total_zero_grad / n
    else:
        for k, v in diag_sum.items():
            stats[k] = v / max(num_batches, 1)
    return stats


@torch.no_grad()
def evaluate_model(
    model,
    database_dir: str,
    query_dir: str,
    device: str,
    min_index: int | None = None,
    max_index: int | None = None,
    tolerance: int = 3,
    recall_ks: tuple[int, ...] = (1, 5),
    precision_k: int = 5,
    batch_size: int = 16,
) -> dict[str, float]:
    """在验证段上跑一次真实的检索评测。

    为什么必须有这个函数：训练 loss 下降【不能】证明检索变好。
    一是大量三元组满足 margin 后 loss 恒为 0，会把平均值拽低；
    二是 loss 衡量的是"margin 满足程度"，不是"排序质量"。
    所以模型选择必须按真实的 Recall@K 来做。

    实现上完全复用已有组件：抽特征 → retrieve_top_k → evaluate_retrieval。
    """
    database_features, database_paths = extract_with_model(
        model, database_dir, device, batch_size, show_progress=False
    )
    query_features, query_paths = extract_with_model(
        model, query_dir, device, batch_size, show_progress=False
    )

    keep = [
        i
        for i, path in enumerate(query_paths)
        if (min_index is None or image_index_from_path(path) >= min_index)
        and (max_index is None or image_index_from_path(path) <= max_index)
    ]
    if len(keep) == 0:
        raise ValueError(
            f"验证段为空：{query_dir} 里没有 index 落在 [{min_index}, {max_index}] 的图像"
        )

    k = max(max(recall_ks), precision_k)
    _, top_indices = retrieve_top_k(
        query_features=query_features[keep],
        database_features=database_features,
        k=k,
    )

    return evaluate_retrieval(
        query_paths=[query_paths[i] for i in keep],
        database_paths=database_paths,
        top_indices=top_indices,
        recall_ks=list(recall_ks),
        precision_k=precision_k,
        tolerance=tolerance,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-dir", type=str, required=True)
    parser.add_argument("--database-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--positive-tolerance", type=int, default=3)
    parser.add_argument("--negative-gap", type=int, default=20)
    parser.add_argument("--min-index", type=int, default=None)
    parser.add_argument("--max-index", type=int, default=None)
    # --- database 区间过滤：不传时行为与旧版一致（用全部 database 图像）---
    parser.add_argument("--db-min-index", type=int, default=None)
    parser.add_argument(
        "--db-max-index", type=int, default=None,
        help="database 的最大帧号。配合训练段留出 purged gap，避免边界泄漏",
    )
    # --- 验证与模型选择 ---
    parser.add_argument(
        "--val-min-index", type=int, default=None,
        help="验证段起始帧号。不传则不做验证（退化成旧行为）",
    )
    parser.add_argument("--val-max-index", type=int, default=None)
    parser.add_argument("--tolerance", type=int, default=3)
    parser.add_argument(
        "--early-stop-patience", type=int, default=0,
        help="连续多少个 epoch 验证 R@1 没提升就停。0 表示不早停",
    )
    # --- 优化器与增强 ---
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--no-augment", action="store_true",
        help="关闭数据增强（消融实验用）",
    )
    parser.add_argument(
        "--no-schedule", action="store_true",
        help="关闭 cosine 学习率调度（消融实验用）",
    )
    # --- 损失函数 ---
    parser.add_argument(
        "--loss", choices=["triplet", "infonce"], default="triplet",
        help="triplet 每步只用 1 个负样本且满足 margin 后零梯度；"
             "infonce 用上 batch 内全部合格负样本，且自带难样本加权",
    )
    parser.add_argument(
        "--tau", type=float, default=0.07,
        help="InfoNCE 温度。越小越只盯最难的负样本（趋近 hardest mining）",
    )
    # --- 池化 ---
    parser.add_argument("--pooling", choices=["gap", "gem"], default="gap")
    parser.add_argument("--gem-p", type=float, default=3.0)
    parser.add_argument(
        "--seed", type=int, default=0,
        help="随机种子。验证集只有 10 帧，R@1 的粒度是 0.1，"
             "不固定种子的话不同次运行会差 1-2 个 query，消融对比就不可信了",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = TripletPlaceDataset(
        anchor_dir=args.anchor_dir,
        database_dir=args.database_dir,
        transform=get_default_transform() if args.no_augment else get_train_transform(),
        positive_tolerance=args.positive_tolerance,
        negative_gap=args.negative_gap,
        min_index=args.min_index,
        max_index=args.max_index,
        db_min_index=args.db_min_index,
        db_max_index=args.db_max_index,
        return_negative=(args.loss == "triplet"),
    )

    print(f"训练三元组: {len(dataset)} 个 anchor, {len(dataset.database_paths)} 张 database")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )

    model = ResNet18FeatureExtractor(
        pretrained=True, pooling=args.pooling, gem_p=args.gem_p
    ).to(device)

    criterion = torch.nn.TripletMarginLoss(margin=args.margin, p=2)

    # AdamW 而非 Adam：Adam 把 weight decay 混进梯度里，会被自适应分母缩放，
    # 导致正则化强度在不同参数上失控。AdamW 把它从梯度中解耦，直接作用于参数。
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        None
        if args.no_schedule
        else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    )

    do_validate = args.val_min_index is not None or args.val_max_index is not None
    best_recall = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def save(epoch, val_metrics):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epochs": epoch + 1,
                "lr": args.lr,
                "margin": args.margin,
                "val_metrics": val_metrics,
            },
            output_path,
        )

    for epoch in range(args.epochs):
        stats = train_one_epoch(
            model=model,
            loader=loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            loss_type=args.loss,
            tau=args.tau,
            negative_gap=args.negative_gap,
        )

        line = f"Epoch {epoch + 1}/{args.epochs} - loss: {stats['loss']:.4f}"
        if args.loss == "triplet":
            line += f" | 零梯度三元组 {stats['zero_grad_ratio']:.0%}"
        else:
            line += (f" | 有效负样本 {stats['valid_negatives']:.1f}"
                     f" (梯度>1%: {stats['effective_negs']:.1f})"
                     f" | in-batch top1 {stats['top1_acc']:.2f}")
        val_metrics = None

        if do_validate:
            val_metrics = evaluate_model(
                model=model,
                database_dir=args.database_dir,
                query_dir=args.anchor_dir,
                device=device,
                min_index=args.val_min_index,
                max_index=args.val_max_index,
                tolerance=args.tolerance,
                batch_size=args.batch_size,
            )
            recall1 = val_metrics["recall@1"]
            line += f" | val R@1: {recall1:.4f}  R@5: {val_metrics['recall@5']:.4f}"

            if recall1 > best_recall:
                best_recall, best_epoch = recall1, epoch
                epochs_without_improvement = 0
                save(epoch, val_metrics)
                line += "  <- best, saved"
            else:
                epochs_without_improvement += 1

        if scheduler is not None:
            scheduler.step()

        print(line)

        if (
            args.early_stop_patience > 0
            and epochs_without_improvement >= args.early_stop_patience
        ):
            print(f"验证指标连续 {args.early_stop_patience} 个 epoch 未提升，提前停止")
            break

    if do_validate:
        print(f"\n最佳 checkpoint 来自 epoch {best_epoch + 1}，验证 R@1 = {best_recall:.4f}")
        if best_epoch != args.epochs - 1:
            print("（注意：最佳并非最后一个 epoch —— 模型选择起作用了）")
    else:
        # 没有验证段时保持旧行为：直接保存最后一个 epoch
        save(args.epochs - 1, None)

    print(f"Saved checkpoint to {output_path}")


if __name__ == "__main__":
    main()
