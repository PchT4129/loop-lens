import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import ImageFolderDataset, get_default_transform
from src.models import ResNet18FeatureExtractor


def build_model(checkpoint_path: str | None, device: str):
    use_pretrained = checkpoint_path is None

    model = ResNet18FeatureExtractor(pretrained=use_pretrained)

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

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
):
    """用一个【已有的】模型抽特征。训练中的验证评测直接复用这个函数。

    注意 model.eval() + torch.no_grad() 两者都要：前者让 BatchNorm 用固定的
    running statistics（否则同一张图在不同 batch 里会得到不同特征），后者关掉
    autograd 省显存。二者正交，缺一不可。
    """
    dataset = ImageFolderDataset(
        root_dir=image_dir,
        transform=get_default_transform(),
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
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    return extract_with_model(
        model=model,
        image_dir=image_dir,
        device=device,
        batch_size=batch_size,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    features, paths = extract_features(
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "features": features,
            "paths": paths,
        },
        output_path,
    )

    print(f"Saved {len(paths)} features to {output_path}")
    print(f"Feature shape: {features.shape}")


if __name__ == "__main__":
    main()