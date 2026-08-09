import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import ImageFolderDataset, get_default_transform
from src.models import ResNet18FeatureExtractor


def build_model(
    checkpoint_path: str | None,
    device: str,
    pooling: str = "gap",
    gem_p: float = 3.0,
):
    use_pretrained = checkpoint_path is None

    model = ResNet18FeatureExtractor(
        pretrained=use_pretrained, pooling=pooling, gem_p=gem_p
    )

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        # GeM 的 pool.p 可能不在旧 checkpoint 里，用 strict=False 容忍它，
        # 但【其余任何缺失都要报错】——否则会退化成"随机权重静默生效"。
        missing, unexpected = model.load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )
        unexpected_keys = list(unexpected)
        missing_keys = [k for k in missing if k != "pool.p"]
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                f"checkpoint 与模型结构不匹配：缺失 {missing_keys}，多余 {unexpected_keys}"
            )

    model = model.to(device)
    model.eval()

    return model


def extract_with_model(
    model,
    image_dir: str | Path,
    device: str,
    batch_size: int = 16,
    num_workers: int = 2,
    show_progress: bool = True,
    image_size: int = 224,
):
    """用一个【已有的】模型抽特征。训练中的验证评测直接复用这个函数。

    注意 model.eval() + torch.no_grad() 两者都要：前者让 BatchNorm 用固定的
    running statistics（否则同一张图在不同 batch 里会得到不同特征），后者关掉
    autograd 省显存。二者正交，缺一不可。
    """
    dataset = ImageFolderDataset(
        root_dir=image_dir,
        transform=get_default_transform(image_size),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    was_training = model.training
    model.eval()

    all_features = []
    all_paths = []

    with torch.no_grad():
        iterator = tqdm(loader, desc=f"Extracting {image_dir}") if show_progress else loader
        for images, paths in iterator:
            images = images.to(device)
            all_features.append(model(images).cpu())
            all_paths.extend(paths)

    if was_training:
        model.train()

    return torch.cat(all_features, dim=0), all_paths


def extract_features(
    image_dir: str | Path,
    batch_size: int = 16,
    checkpoint_path: str | None = None,
    pooling: str = "gap",
    gem_p: float = 3.0,
    image_size: int = 224,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(
        checkpoint_path=checkpoint_path,
        device=device,
        pooling=pooling,
        gem_p=gem_p,
    )

    return extract_with_model(
        model=model,
        image_dir=image_dir,
        device=device,
        batch_size=batch_size,
        image_size=image_size,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--pooling", choices=["gap", "gem"], default="gap",
        help="全局池化方式。gem 会放大高响应区域，抑制天空/路面的稀释",
    )
    parser.add_argument("--gem-p", type=float, default=3.0)
    parser.add_argument(
        "--image-size", type=int, default=224,
        help="输入分辨率。backbone 是全卷积+自适应池化，改这个不需要动模型代码",
    )
    args = parser.parse_args()

    features, paths = extract_features(
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint,
        pooling=args.pooling,
        gem_p=args.gem_p,
        image_size=args.image_size,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 存元数据：早期版本只存 {features, paths}，导致必须靠文件名后缀去猜
    # "这份特征是哪个 checkpoint、什么分辨率抽的"。这里把配置一起记下来。
    torch.save(
        {
            "features": features,
            "paths": paths,
            "meta": {
                "backbone": "resnet18",
                "pooling": args.pooling,
                "gem_p": args.gem_p if args.pooling == "gem" else None,
                "image_size": args.image_size,
                "checkpoint": args.checkpoint,
                "feature_dim": int(features.shape[1]),
            },
        },
        output_path,
    )

    print(f"Saved {len(paths)} features to {output_path}")
    print(f"Feature shape: {features.shape}")
    print(f"Config: pooling={args.pooling}, image_size={args.image_size}, "
          f"checkpoint={args.checkpoint}")


if __name__ == "__main__":
    main()